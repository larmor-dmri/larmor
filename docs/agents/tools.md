---
myst:
  html_meta:
    description: "Every tool that BrainViewer offers agents through MCP, with its arguments and description."
---

# Tools

This page lists the tools the viewer offers an agent, with the descriptions the agent is given. Each tool is one
of the viewer's actions, the same method that the window and Python call, so `load` here is `Viewer.load` in
Python. The arguments are sent as JSON. An enum is sent as its text (`"volumes"`), a position as a list of numbers
in world coordinates (mm, RAS+), and an item by the id that `scene` gives it. Paths are on the computer the viewer
runs on.

```{eval-rst}
.. agent-tools::
```
