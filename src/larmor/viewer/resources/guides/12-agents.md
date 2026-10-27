# Driving the viewer from an agent

An agent, such as Claude Code, can drive the viewer through MCP (the Model Context Protocol). It calls the same
actions as the window: it loads files, changes options, moves the cursor, turns the cameras and takes
screenshots, and you watch it happen.

## 1. Start the server

From the command line, run `lar bv --mcp`. The server listens on port 8765 unless you changed the port in
Settings. `lar bv --mcp-port 9000` uses another port (from 1024 to 65535) and starts the server even without
`--mcp`.

From the window, open [Settings](09-settings.md) and turn on **Serve agents** under **Agents (MCP)**. The **Port**
field above it sets the port.

The server is off when the viewer starts, unless `--mcp` or `--mcp-port` is given.

## 2. Find the URL and the token

The viewer prints both to the terminal:

```
Serving agents (MCP) at http://127.0.0.1:8765/mcp
Token: ...
```

The Settings panel shows them too, under **Agents (MCP)**, each with a **Copy** button. The token stays the same
from one start to the next, and is kept in the file `mcp-token`, which only you can read:

- macOS: `cat ~/Library/Application\ Support/larmor/mcp-token`
- Linux: `cat ~/.config/larmor/mcp-token` (or under `$XDG_CONFIG_HOME`)
- Windows (PowerShell): `type $env:APPDATA\larmor\mcp-token`

**New token** in Settings replaces it, and agents with the old one are refused until given the new one.

## 3. Connect the agent

For Claude Code, run the command below with the token in place of `<token>`:

```
claude mcp add --scope user --transport http larmor http://127.0.0.1:8765/mcp --header "Authorization: Bearer <token>"
```

With `--scope user` the server is there in every folder you start Claude Code in; without it, only in the
folder where you ran the command.

A client that reads its servers from a JSON file with an `mcpServers` list, such as Claude Code's `.mcp.json`,
takes the entry below. Such a file is often committed with a project, so the entry reads the token from the
environment variable `LARMOR_MCP_TOKEN`. Never commit the token itself.

```json
{
  "mcpServers": {
    "larmor": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {"Authorization": "Bearer ${LARMOR_MCP_TOKEN}"}
    }
  }
}
```

## 4. Check that it works

In Claude Code, `/mcp` lists the servers, and `larmor` should be shown as connected. Then ask "What is loaded in
the viewer?". The agent calls the `scene` tool and lists the items and the state of the views.

## What an agent can do

Each tool is one of the viewer's actions, the same ones the window and the Python Viewer class use. They fall
into these groups:

- **Items**: `load`, `load_matlab`, `remove`, `save_item`, `save_session`, `load_session`.
- **Options**: `set`, `set_default`, `labels`, `item_choices`, `data_choices`.
- **Tractogram regions**: `add_mask`, `add_sphere`, `clear_regions`, `kept_streamlines`.
- **Views**: `select`, `move_cursor`, `step_slice`, `select_view`, `step_volume`, `zoom`, `pan`,
  `reset_camera`, `rotate_camera`, `set_camera`, `set_convention`, `view_centre`, `readout`,
  `toggle_maximised`, `toggle_overlay`, `toggle_crosshair`, `set_tool`.
- **Phantoms**: `new_phantom`, `close_phantom`, `set_phantom_grid`, `set_tissue`, `set_compartments`,
  `remove_tissue`, `set_tissue_map`, `set_background_tissue`, `set_scan`, `start_bundle`, `add_bundle_point`,
  `remove_bundle_point`, `set_bundle_points`, `finish_bundle`, `cancel_bundle`, `edit_bundle`, `use_as_bundle`,
  `set_bundle`, `show_proxy`, `show_grid_box`, `check_phantom`, `set_run_folder`, `save_phantom`, `load_phantom`,
  `run_phantom`, `open_simulation`.
- **Images**: `screenshot`, `screenshot_views`, `record_rotation`. The agent sees the images it takes.
- **Scene**: `scene` lists the items, with the ids the other tools name them by, and the state of the views.

Paths are on the viewer's computer, not the agent's, so the agent loads files and writes screenshots there. When
a large file is read, the agent waits for it for up to five minutes. An agent cannot change the settings of the
Settings panel, open other windows, or reach files the viewer cannot read.

## Requests to try

- "Load `/data/sub-01/t1.nii.gz` as a volume."
- "Show the T1 with the hot colormap."
- "Move the cursor to (0, -20, 10)."
- "Show the 3D view from the side and take a screenshot to `/tmp/side.png`."
- "Save the session to `/data/sub-01/view.lar.json`."

## On a remote computer

When the viewer runs on a server and the agent on your computer, forward the port over SSH, with the same port
on both sides:

```
ssh -L 8765:127.0.0.1:8765 user@server
```

Then connect the agent to `http://127.0.0.1:8765/mcp` on your computer, with the token from the server. The ports
must be the same, because the viewer answers only requests addressed to `127.0.0.1` or `localhost` at its own
port, so a forward such as `-L 9000:127.0.0.1:8765` is refused.

## When it fails

- **The port is in use.** The viewer says "Cannot serve agents on port 8765" and opens without the server.
  Choose another port with `--mcp-port` or in Settings, and use it in the agent's URL.
- **401.** The token is missing or old, as after **New token**. Give the agent the current token (remove the
  server with `claude mcp remove --scope user larmor` and add it again).
- **403.** The request reached the viewer at another host or port than its own, as with an SSH forward to
  another port, or came from a web page. Use the same port on both sides and `127.0.0.1` in the URL.
- **"The viewer did not answer within 300 s".** A call took longer than five minutes, such as a very large file
  being read. The file still appears in the viewer once it is read.

## Why it is safe on a shared computer

The server listens at `127.0.0.1` only, so other computers cannot reach it. Every request must carry the token,
which only you can read, so other users of the same computer cannot drive your viewer.
