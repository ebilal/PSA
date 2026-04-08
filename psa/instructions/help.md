# MemPalace

AI memory system. Store everything, find anything. Local, free, no API key.

---

## Slash Commands

| Command              | Description                    |
|----------------------|--------------------------------|
| /psa:init      | Install and set up MemPalace   |
| /psa:search    | Search your memories           |
| /psa:mine      | Mine projects and conversations|
| /psa:status    | Palace overview and stats      |
| /psa:help      | This help message              |

---

## MCP Tools (19)

### Palace (read)
- psa_status -- Palace status and stats
- psa_list_wings -- List all wings
- psa_list_rooms -- List rooms in a wing
- psa_get_taxonomy -- Get the full taxonomy tree
- psa_search -- Search memories by query
- psa_check_duplicate -- Check if a memory already exists
- psa_get_aaak_spec -- Get the AAAK specification

### Palace (write)
- psa_add_drawer -- Add a new memory (drawer)
- psa_delete_drawer -- Delete a memory (drawer)

### Knowledge Graph
- psa_kg_query -- Query the knowledge graph
- psa_kg_add -- Add a knowledge graph entry
- psa_kg_invalidate -- Invalidate a knowledge graph entry
- psa_kg_timeline -- View knowledge graph timeline
- psa_kg_stats -- Knowledge graph statistics

### Navigation
- psa_traverse -- Traverse the palace structure
- psa_find_tunnels -- Find cross-wing connections
- psa_graph_stats -- Graph connectivity statistics

### Agent Diary
- psa_diary_write -- Write a diary entry
- psa_diary_read -- Read diary entries

---

## CLI Commands

    psa init <dir>                  Initialize a new palace
    psa mine <dir>                  Mine a project (default mode)
    psa mine <dir> --mode convos    Mine conversation exports
    psa search "query"              Search your memories
    psa split <dir>                 Split large transcript files
    psa wake-up                     Load palace into context
    psa compress                    Compress palace storage
    psa status                      Show palace status
    psa repair                      Rebuild vector index
    psa hook run                    Run hook logic (for harness integration)
    psa instructions <name>         Output skill instructions

---

## Auto-Save Hooks

- Stop hook -- Automatically saves memories every 15 messages. Counts human
  messages in the session transcript (skipping command-messages). When the
  threshold is reached, blocks the AI with a save instruction. Uses
  ~/.psa/hook_state/ to track save points per session. If
  stop_hook_active is true, passes through to prevent infinite loops.

- PreCompact hook -- Emergency save before context compaction. Always blocks
  with a comprehensive save instruction because compaction means the AI is
  about to lose detailed context.

Hooks read JSON from stdin and output JSON to stdout. They can be invoked via:

    echo '{"session_id":"abc","stop_hook_active":false,"transcript_path":"..."}' | psa hook run --hook stop --harness claude-code

---

## Architecture

    Wings (projects/people)
      +-- Rooms (topics)
            +-- Closets (summaries)
                  +-- Drawers (verbatim memories)

    Halls connect rooms within a wing.
    Tunnels connect rooms across wings.

The palace is stored locally using ChromaDB for vector search and SQLite for
metadata. No cloud services or API keys required.

---

## Getting Started

1. /psa:init -- Set up your palace
2. /psa:mine -- Mine a project or conversation
3. /psa:search -- Find what you stored
