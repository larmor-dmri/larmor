---
myst:
  html_meta:
    description: "How an agent can drive Larmor's viewer, BrainViewer, through the Model Context Protocol (MCP), to load files, set options, move the cameras and take screenshots."
---

# Agents

An agent, such as Claude Code or any other client of the Model Context Protocol (MCP), can drive a running
viewer. It can load files, change options, move the cursor, turn the cameras and take screenshots. Each tool the
viewer offers is one of the actions its window uses, so the agent works on what you see.

- [Connecting an agent](connecting.md) shows how to start the viewer's server and give its URL and token to the
  agent. The viewer's Help panel shows the same page.
- [Tools](tools.md) lists every tool the viewer offers, with its arguments, as the agent sees it.
- [Security](security.md) explains which requests the server accepts, and how to keep the token safe.

```{toctree}
:hidden:

Connecting an agent <connecting>
tools
security
```
