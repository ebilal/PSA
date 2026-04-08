---
name: psa
description: PSA — mine projects and conversations into a searchable memory palace. Use when asked about psa, memory palace, mining memories, searching memories, or palace setup.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# PSA

A searchable memory palace for AI — mine projects and conversations, then search them semantically.

## Prerequisites

Ensure `psa` is installed:

```bash
psa --version
```

If not installed:

```bash
pip install psa
```

## Usage

PSA provides dynamic instructions via the CLI. To get instructions for any operation:

```bash
psa instructions <command>
```

Where `<command>` is one of: `help`, `init`, `mine`, `search`, `status`.

Run the appropriate instructions command, then follow the returned instructions step by step.
