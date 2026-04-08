# MCP Integration — Claude Code

## Setup

Run the MCP server:

```bash
python -m psa.mcp_server
```

Or add it to Claude Code:

```bash
claude mcp add psa -- python -m psa.mcp_server
```

## Available Tools

The server exposes the full PSA MCP toolset. Common entry points include:

- **psa_status** — palace stats (wings, rooms, drawer counts)
- **psa_search** — semantic search across all memories
- **psa_list_wings** — list all projects in the palace

## Usage in Claude Code

Once configured, Claude Code can search your memories directly during conversations.
