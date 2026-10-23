---
myst:
  html_meta:
    description: "How BrainViewer's MCP server is protected: it listens at 127.0.0.1 only and every request needs a token. Also what a connected agent can do."
---

# Security

The viewer's MCP server lets a program drive the viewer, so it accepts requests only from agents you have given
its token, on your own computer.

## What the server accepts

- **It is off until you start it**, with `lar bv --mcp`, with `--mcp-port`, or with **Serve agents** in Settings.
- **It listens at `127.0.0.1` only**, the local address, so other computers on the network cannot reach it. An
  agent on another computer reaches it through an SSH forward, as
  [Connecting an agent](connecting.md#on-a-remote-computer) shows.
- **Every request needs the token**, sent as the header `Authorization: Bearer <token>`. A request without it, or
  with an old one, is refused with 401. The token is 32 random bytes, kept in the file `mcp-token` in your settings
  folder. Only you can read that file (on Windows, through the file's access list), so other users of the same
  computer cannot drive your viewer.
- **The request must be addressed to the viewer itself.** Its `Host` must be `127.0.0.1` or `localhost` at the
  viewer's own port, and a request that names a web page as its `Origin` is refused. Both are refused with 403. A
  web page open in your browser therefore cannot drive the viewer, even one that points a name of its own at
  `127.0.0.1`.

## Keeping the token safe

The token stays the same from one start to the next, so an agent you added once keeps working. If someone else
may have seen the token, click **New token** under **Agents (MCP)** in Settings. It replaces the token at once,
and agents with the old one are refused until you give them the new one.

Never commit the token. In a configuration file shared with a project, refer to it through an environment
variable, such as `LARMOR_MCP_TOKEN`, as [Connecting an agent](connecting.md#3-connect-the-agent) shows.

## What a connected agent can do

An agent with the token acts as you in the viewer. It can load any file you can read, and write screenshots,
videos and sessions wherever you can write, on the computer the viewer runs on. It can change the defaults and the
convention of the running session, but not the saved settings, and it cannot open other windows or run other
programs. Connect only agents you trust with your files, and watch what they do in the viewer.
