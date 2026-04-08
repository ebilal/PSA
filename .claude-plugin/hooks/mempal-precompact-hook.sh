#!/bin/bash
# PSA PreCompact Hook — thin wrapper calling Python CLI
# All logic lives in psa.hooks_cli for cross-harness extensibility
INPUT=$(cat)
echo "$INPUT" | python3 -m psa hook run --hook precompact --harness claude-code
