#!/bin/bash
# PSA Stop Hook — thin wrapper calling Python CLI
# All logic lives in psa.hooks_cli for cross-harness extensibility
INPUT=$(cat)
echo "$INPUT" | python3 -m psa hook run --hook stop --harness claude-code
