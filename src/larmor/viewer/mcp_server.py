"""Serves a running viewer's actions to agents over MCP, on local HTTP."""

import asyncio
import collections.abc
import concurrent.futures
import dataclasses
import errno
import hmac
import inspect
import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import typing
from collections.abc import Awaitable, Callable
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from larmor.viewer.model.item import Item
from larmor.viewer.model.settings import config_folder
from larmor.viewer.model.viewer import AGENT_ACTION, Viewer

if TYPE_CHECKING:
    import uvicorn
    from mcp.server.mcpserver import MCPServer

HOST = "127.0.0.1"
PATH = "/mcp"  # the endpoint, as the mcp package serves it
TOKEN_FILE = "mcp-token"
# The longest a tool call waits for the window: reading a large file takes a while.
CALL_TIMEOUT = 300.0
# Images an agent sees, beside the path, when an action returns their file.
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

type Post = Callable[[Callable[[], None]], None]
type Asgi = Callable[
    [dict[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]


_UNSET = object()

# Called with an action's result, or why it failed.
type Done = Callable[[object, str | None], None]
# Starts an action on the main thread with the agent's arguments and `then`, a Done.
type Background = Callable[..., None]


class PortInUseError(OSError):
    """Raised when the MCP server's port is taken by another program."""


def token_path() -> Path:
    """Returns the file that keeps the MCP server's token, beside the settings."""
    return config_folder() / TOKEN_FILE


def read_token(path: Path | None = None) -> str:
    """Returns the saved token, or makes and saves a new one if there is none."""
    path = path if path is not None else token_path()
    return saved_token(path) or new_token(path)


def saved_token(path: Path) -> str:
    """Returns the saved token, or "" if the file is missing or empty.

    It never writes a token, so the server's guard uses it: a request must never
    create one.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def new_token(path: Path | None = None) -> str:
    """Makes a new random token and saves it, readable by the user only."""
    path = path if path is not None else token_path()
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside it and moved into place in one step, so a reader never sees
    # an empty file. mkstemp makes it readable by the user only on macOS and Linux;
    # on Windows its access list is restricted before the write.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        if sys.platform == "win32":
            _restrict_to_user(Path(temporary))
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(token + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return token


def _restrict_to_user(path: Path) -> None:
    """Lets only the current user open path, through its Windows access list.

    Raises:
        PermissionError: If the access list cannot be set.
    """
    user = os.environ.get("USERNAME", "")
    domain = os.environ.get("USERDOMAIN", "")
    account = f"{domain}\\{user}" if domain else user
    icacls = Path(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icacls.exe")
    # Removes the entries inherited from the folder and grants the user alone.
    command = [str(icacls), str(path), "/inheritance:r", "/grant:r", f"{account}:F"]
    if not user:
        raise PermissionError(f"Cannot restrict the token file {path}: no USERNAME.")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise PermissionError(
            f"Cannot restrict the token file {path} to the user: {error}"
        ) from error
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise PermissionError(
            f"Cannot restrict the token file {path} to the user: {message}"
        )


def actions(viewer: Viewer) -> dict[str, Callable[..., Any]]:
    """Returns the viewer's methods marked for agents, by name."""
    return {
        name: getattr(viewer, name)
        for name, member in inspect.getmembers(type(viewer), inspect.isfunction)
        if getattr(member, AGENT_ACTION, False)
    }


SCENE_DESCRIPTION = (
    "The items in the scene, in scene tree order (id, category, name, "
    "source file, number of volumes, value range and options), and the "
    "state of the views (cursor, main volume, selection, cameras, ...). "
    "Other tools name items by these ids."
)


def description(method: Callable[..., Any]) -> str:
    """Returns a method's docstring without its Python example, for a tool."""
    doc = inspect.getdoc(method) or ""
    return doc.split("\nExample:")[0].strip()


def to_json(value: object) -> object:
    """Converts an action's result to JSON values, items without their data."""
    if isinstance(value, Item):
        return {
            "id": value.id,
            "category": value.category.value,
            "name": value.name,
            "source": None if value.source is None else str(value.source),
            "volumes": value.volume_count,
            "value_range": to_json(value.value_range),
            "options": to_json(value.options),
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_json(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    return value


def scene_summary(viewer: Viewer) -> dict[str, object]:
    """Returns the items, in scene tree order, and the state of the views."""
    return {
        "items": [to_json(item) for item in viewer.scene.items],
        "view": to_json(viewer.scene.view),
    }


class McpServer:
    """Serves the viewer's actions to agents over Streamable HTTP, on 127.0.0.1.

    Every request needs the token as `Authorization: Bearer <token>`, a `Host`
    of this machine at the server's port, and no `Origin` but this server's.
    Tool calls run on the main thread through `post`, as the scene and the
    drawing allow no other. An action named in `background` is started through
    its function instead, and its tool answers once that tells the result.
    """

    def __init__(
        self,
        viewer: Viewer,
        post: Post,
        token_file: Path | None = None,
        background: dict[str, Background] | None = None,
    ) -> None:
        """Creates a server, not yet started, that runs each call through post."""
        self.viewer = viewer
        self.post = post
        self.background = background or {}
        self.token_file = token_file if token_file is not None else token_path()
        self.port: int | None = None  # while running
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._pending: set[concurrent.futures.Future[Any]] = set()

    @property
    def running(self) -> bool:
        """Returns whether the server is listening."""
        return self._server is not None

    @property
    def token(self) -> str:
        """Returns the token requests must carry, read from its file or made."""
        # Read each time, so a new token made by another viewer applies at once.
        return read_token(self.token_file)

    def url(self, port: int | None = None) -> str:
        """Returns the endpoint agents connect to, at the running or the given port."""
        return f"http://{HOST}:{port or self.port}{PATH}"

    def renew_token(self) -> str:
        """Makes and saves a new token; requests with the old one are refused."""
        return new_token(self.token_file)

    def start(self, port: int) -> None:
        """Starts listening on port, in a thread of its own.

        Raises:
            PortInUseError: If another program listens on the port; a server
                already running keeps serving on its port.
        """
        read_token(self.token_file)  # made now if there is none, for the agents
        if self.running and port == self.port:
            self.stop()  # the port is this server's own
        # Bound here, so a taken port fails now, with a message, not in the thread;
        # and before the old server stops, which keeps serving if this fails.
        sock = _bind(port)
        self.stop()

        import uvicorn

        app = self._app(port)
        config = uvicorn.Config(
            app,
            log_level="warning",
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="larmor-mcp",
            daemon=True,
        )
        self._server, self._thread, self.port = server, thread, port
        thread.start()

    def stop(self) -> None:
        """Stops listening; calls still waiting for the main thread are dropped.

        Runs on the main thread, so no action is mid-run while it stops.
        """
        if self._server is None:
            return
        for future in list(self._pending):
            # A call that runs on this thread has finished; one still running is a
            # background load, which then can no longer tell its item.
            if not future.cancel() and not future.done():
                future.set_exception(
                    RuntimeError("The viewer closed before the file was read.")
                )
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = self._thread = self.port = None

    def _app(self, port: int) -> Asgi:
        """Returns the web application that checks requests, then serves MCP."""
        from mcp.server.transport_security import TransportSecuritySettings

        mcp = self._mcp_server()
        hosts = [f"{HOST}:{port}", f"localhost:{port}"]
        app = mcp.streamable_http_app(
            streamable_http_path=PATH,
            host=HOST,
            # The mcp package's own check of Host and Origin, behind ours.
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=hosts,
                allowed_origins=[f"http://{h}" for h in hosts],
            ),
        )
        token_file = self.token_file
        return _Guard(app, hosts, lambda: saved_token(token_file))

    def _mcp_server(self) -> MCPServer:
        """Returns an MCP server with one tool per action, and one listing the scene."""
        from mcp.server.mcpserver import MCPServer

        from larmor.viewer import APP_NAME

        # The mcp package configures the root logger as it is made; Larmor's logging is
        # put back as it was.
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        mcp: MCPServer = MCPServer(
            APP_NAME,
            instructions=(
                "Drives the Larmor viewer the user is looking at. Call `scene` "
                "for the items' ids; positions are world mm, RAS+."
            ),
            log_level="WARNING",
        )
        root.handlers[:] = handlers
        root.setLevel(level)
        for name, method in actions(self.viewer).items():
            mcp.add_tool(
                self._tool(method, self.background.get(name)),
                name=name,
                description=description(method),
                structured_output=False,
            )
        viewer = self.viewer

        def scene() -> dict[str, object]:
            """Returns the scene summary."""
            return scene_summary(viewer)

        mcp.add_tool(
            self._tool(scene),
            name="scene",
            description=SCENE_DESCRIPTION,
            structured_output=False,
        )
        return mcp

    def _tool(
        self, method: Callable[..., Any], background: Background | None = None
    ) -> Callable[..., Awaitable[object]]:
        """Returns a coroutine with method's signature that runs it on the main thread.

        With background, the call starts background with the agent's arguments
        instead, and waits until it tells the result.
        """
        from mcp.server.mcpserver.exceptions import ToolError

        pending = self._pending
        post = self.post
        defaults = {
            p.name: p.default
            for p in inspect.signature(method).parameters.values()
            if p.default is not p.empty
        }

        async def call(**kwargs: object) -> object:
            """Posts the action to the main thread and returns its result as content."""
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            pending.add(future)
            try:
                post(partial(_run_action, future, method, background, defaults, kwargs))
                return _content(await _result(future, ToolError))
            finally:
                pending.discard(future)

        hints = inspect.get_annotations(method, eval_str=True)
        params = [
            p.replace(annotation=_schema_type(hints.get(p.name, Any)))
            for p in tool_parameters(method)
        ]
        call.__signature__ = inspect.signature(method).replace(  # type: ignore[attr-defined]
            parameters=params, return_annotation=Any
        )
        call.__name__ = method.__name__
        return call


def _run_action(
    future: concurrent.futures.Future[Any],
    method: Callable[..., Any],
    background: Background | None,
    defaults: dict[str, object],
    kwargs: dict[str, object],
) -> None:
    """Runs or starts an action on the main thread, and sets future to its outcome."""
    if not future.set_running_or_notify_cancel():
        return  # dropped while it waited
    try:
        if background is not None:
            # mcp passes the action's defaults too; background has its own, such
            # as the window's read limit, so those are left out.
            given = {k: v for k, v in kwargs.items() if v != defaults.get(k, _UNSET)}
            background(**given, then=partial(_tell, future))
            return
        result = method(**kwargs)
        # Converted here too, as the scene changes on this thread only.
        future.set_result(to_json(result))
    except Exception as error:
        if not future.done():  # not told already, nor stopped
            future.set_exception(error)


def _tell(
    future: concurrent.futures.Future[Any], result: object, error: str | None
) -> None:
    """Sets future to a background action's result, or to its error."""
    if future.done():
        return  # the server stopped first
    if error is not None:
        future.set_exception(RuntimeError(error))
        return
    try:
        future.set_result(to_json(result))
    except Exception as failure:
        future.set_exception(failure)


async def _result(
    future: concurrent.futures.Future[Any], tool_error: type[Exception]
) -> object:
    """Waits for a call's result.

    Args:
        future: The call's future.
        tool_error: mcp's ToolError class.

    Returns:
        The action's result, as JSON values.

    Raises:
        tool_error: If the viewer does not answer in time, closes first, or the
            action fails.
    """
    waiting = asyncio.wrap_future(future)
    done, _ = await asyncio.wait({waiting}, timeout=CALL_TIMEOUT)
    if not done:
        future.cancel()
        raise tool_error(f"The viewer did not answer within {CALL_TIMEOUT:g} s.")
    try:
        return waiting.result()
    except asyncio.CancelledError:
        raise tool_error("The viewer closed before the call ran.") from None
    except Exception as error:
        # The message, as the window would show it; mcp hides other errors.
        raise tool_error(str(error)) from error


def _bind(port: int) -> socket.socket:
    """Returns a socket listening at 127.0.0.1:port.

    Raises:
        PortInUseError: If another program listens on the port.
    """  # noqa: DOC501 (the OSError raised inside is caught below)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            # Windows binds 127.0.0.1 on a port another program holds at every
            # address; binding every address first, alone, refuses that.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("0.0.0.0", port))
            # SO_REUSEADDR would share a port another program listens on; this
            # refuses that, and Windows binds a port in TIME_WAIT without it.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # On macOS SO_REUSEADDR also lets 127.0.0.1 share a port another program
            # listens on at every address; such a program accepts at 127.0.0.1.
            if sys.platform != "linux" and _accepts(port):
                raise OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE))
            # Binds the port while the connections this viewer closed are in
            # TIME_WAIT, as after a restart.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, port))
        sock.listen(64)
    except OSError as error:
        sock.close()
        raise PortInUseError(
            f"Cannot serve agents on port {port}: {error.strerror}. Another "
            "program may be using it; choose another port with --mcp-port or "
            "in Settings."
        ) from error
    return sock


def _accepts(port: int) -> bool:
    """Returns whether a program accepts connections at 127.0.0.1:port."""
    try:
        with socket.create_connection((HOST, port), timeout=1.0):
            return True
    except OSError:
        return False


def tool_parameters(method: Callable[..., Any]) -> list[inspect.Parameter]:
    """Returns the parameters of a bound action that its tool takes.

    A callback with a default is left out, as no agent can send a function.
    """
    hints = inspect.get_annotations(method, eval_str=True)
    return [
        p
        for p in inspect.signature(method).parameters.values()
        if not _is_callback(hints.get(p.name)) or p.default is p.empty
    ]


def _is_callback(hint: object) -> bool:
    """Returns whether a type hint is a function or an optional one."""
    for part in typing.get_args(hint) or (hint,):
        value = getattr(part, "__value__", part)  # a `type` alias's value
        if typing.get_origin(value) is collections.abc.Callable:
            return True
    return False


def _schema_type(hint: object) -> object:
    """Returns a type hint for the input schema, with `object` read as any value."""
    return Any if hint is object else hint


def _content(result: object) -> list[object]:
    """Returns a result as MCP content, JSON text after the images it names."""
    from mcp.server.mcpserver import Image

    paths = result if isinstance(result, list) else [result]
    images: list[object] = [
        Image(path=p)
        for p in paths
        if isinstance(p, str) and Path(p).suffix.lower() in _IMAGE_SUFFIXES
    ]
    text = "Done." if result is None else json.dumps(result)
    return [*images, text]


class _Guard:
    """Refuses requests with a foreign Host or Origin (403) or no valid token (401)."""

    def __init__(self, app: Asgi, hosts: list[str], token: Callable[[], str]) -> None:
        """Checks requests before app sees them; token gives the current one, or ""."""
        self.app = app
        self.hosts = set(hosts)
        self.origins = {f"http://{h}" for h in hosts}
        self.token = token

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Answers a refused request itself; passes the others on."""
        if scope["type"] != "http":  # the server's start and stop
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        origin = headers.get("origin")
        if headers.get("host") not in self.hosts:
            await _refuse(send, 403, "This server answers only at 127.0.0.1.")
        elif origin is not None and origin not in self.origins:
            await _refuse(send, 403, "Requests from web pages are refused.")
        elif not (token := self.token()) or not hmac.compare_digest(
            headers.get("authorization", "").encode(), f"Bearer {token}".encode()
        ):
            await _refuse(send, 401, "A valid token is needed.", bearer=True)
        else:
            await self.app(scope, receive, send)


async def _refuse(
    send: Callable[[Any], Awaitable[None]], status: int, text: str, bearer: bool = False
) -> None:
    """Sends a plain-text refusal."""
    headers = [(b"content-type", b"text/plain; charset=utf-8")]
    if bearer:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": text.encode()})
