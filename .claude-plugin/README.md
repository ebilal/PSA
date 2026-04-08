# PSA Claude Code Plugin

A Claude Code plugin that gives your AI a persistent memory system. Mine projects and conversations into a searchable palace backed by ChromaDB, with 19 MCP tools, auto-save hooks, and 5 guided skills.

## Prerequisites

- Python 3.9+

## Installation

### Claude Code Marketplace

```bash
claude plugin marketplace add milla-jovovich/psa
claude plugin install --scope user psa
```

### Local Clone

```bash
claude plugin add /path/to/psa
```

## Post-Install Setup

After installing the plugin, run the init command to complete setup (pip install, MCP configuration, etc.):

```
/psa:init
```

## Available Slash Commands

| Command | Description |
|---------|-------------|
| `/psa:help` | Show available tools, skills, and architecture |
| `/psa:init` | Set up PSA -- install, configure MCP, onboard |
| `/psa:search` | Search your memories across the palace |
| `/psa:mine` | Mine projects and conversations into the palace |
| `/psa:status` | Show palace overview -- wings, rooms, drawer counts |

## Hooks

PSA registers two hooks that run automatically:

- **Stop** -- Saves conversation context every 15 messages.
- **PreCompact** -- Preserves important memories before context compaction.

Set the `PSA_DIR` environment variable to a directory path to automatically run `psa mine` on that directory during each save trigger.

## MCP Server

The plugin automatically configures a local MCP server with 19 tools for storing, searching, and managing memories. No manual MCP setup is required -- `/psa:init` handles everything.

## Full Documentation

See the main [README](../README.md) for complete documentation, architecture details, and advanced usage.
