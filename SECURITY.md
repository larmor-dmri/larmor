# Security policy

## Reporting a vulnerability

Please report a security problem privately, in one of two ways:

- by e-mail to security@larmor.org;
- through GitHub's private vulnerability reporting: on the repository's **Security** tab, choose **Report a
  vulnerability**. Only the maintainers see the report, and you can follow and discuss it there until a fix is
  released.

Please do not open a public issue for it.

Say what the problem is, how to reproduce it, and which Larmor version and operating system you use
(`lar --version` prints it). Once a fix is released, the report is
published as a security advisory, crediting you if you wish.

## Supported versions

Fixes go into the next release. Until 1.0, only the latest release is supported.

## What to look at

Most of Larmor reads and writes files on the user's own computer. The part that accepts requests from other
programs is the viewer's MCP server, which lets an agent drive BrainViewer:

- it is off until the user starts it (`lar bv --mcp`, or **Serve agents** in Settings);
- it listens at `127.0.0.1` only;
- every request needs a 32-byte random token, kept in a file readable by the user only;
- requests whose `Host` is not the viewer's own address, or that come from a web page (`Origin`), are refused;
- an agent with the token can read any file the user can read, by loading it, and write screenshots, videos and
  sessions wherever the user can write; it cannot run other programs.

The [security page of the documentation](https://larmor.org/agents/security/) describes this in full. A way around
any of these limits is a vulnerability; so is a file that makes a reader run code or write outside the paths it was
given.
