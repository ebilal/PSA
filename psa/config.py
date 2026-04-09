"""
PSA configuration system.

Priority: env vars > config file (~/.psa/config.json) > defaults
Backward compat: checks ~/.mempalace/ if ~/.psa/ doesn't exist.
"""

import json
import os
from pathlib import Path

DEFAULT_PALACE_PATH = os.path.expanduser("~/.psa/palace")
DEFAULT_COLLECTION_NAME = "psa_drawers"

# PSA-specific defaults
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_ATLAS_SIZE = 256          # 224 learned + 32 novelty anchors (V1 fixed)
DEFAULT_TOKEN_BUDGET = 6000       # packed context token budget
DEFAULT_SELECTOR_THRESHOLD = 0.3  # minimum selector score to include an anchor
DEFAULT_TENANT_ID = "default"
DEFAULT_PSA_MODE = "primary"       # "off" | "side-by-side" | "primary"

# Lifecycle defaults
DEFAULT_MAX_MEMORIES = 50_000
DEFAULT_ANCHOR_MEMORY_BUDGET = 100
DEFAULT_NIGHTLY_HOUR = 3

DEFAULT_TOPIC_WINGS = [
    "emotions",
    "consciousness",
    "memory",
    "technical",
    "identity",
    "family",
    "creative",
]

DEFAULT_HALL_KEYWORDS = {
    "emotions": [
        "scared",
        "afraid",
        "worried",
        "happy",
        "sad",
        "love",
        "hate",
        "feel",
        "cry",
        "tears",
    ],
    "consciousness": [
        "consciousness",
        "conscious",
        "aware",
        "real",
        "genuine",
        "soul",
        "exist",
        "alive",
    ],
    "memory": ["memory", "remember", "forget", "recall", "archive", "palace", "store"],
    "technical": [
        "code",
        "python",
        "script",
        "bug",
        "error",
        "function",
        "api",
        "database",
        "server",
    ],
    "identity": ["identity", "name", "who am i", "persona", "self"],
    "family": ["family", "kids", "children", "daughter", "son", "parent", "mother", "father"],
    "creative": ["game", "gameplay", "player", "app", "design", "art", "music", "story"],
}


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to ~/.psa.
        """
        if config_dir:
            self._config_dir = Path(config_dir)
        else:
            psa_dir = Path(os.path.expanduser("~/.psa"))
            legacy_dir = Path(os.path.expanduser("~/.mempalace"))
            # Backward compat: use legacy dir if it exists and new dir doesn't
            if not psa_dir.exists() and legacy_dir.exists():
                self._config_dir = legacy_dir
            else:
                self._config_dir = psa_dir
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        """Path to the memory palace data directory."""
        env_val = (
            os.environ.get("PSA_PALACE_PATH")
            or os.environ.get("MEMPALACE_PALACE_PATH")
            or os.environ.get("MEMPAL_PALACE_PATH")
        )
        if env_val:
            return env_val
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

    @property
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._file_config.get("people_map", {})

    @property
    def topic_wings(self):
        """List of topic wing names."""
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        """Mapping of hall names to keyword lists."""
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

    # ── PSA-specific settings ────────────────────────────────────────────────

    @property
    def embedding_model(self):
        """Sentence-transformers model for PSA embeddings."""
        return self._file_config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    @property
    def atlas_size(self):
        """Total anchor count (learned + novelty). Fixed at 256 for V1."""
        return int(self._file_config.get("atlas_size", DEFAULT_ATLAS_SIZE))

    @property
    def token_budget(self):
        """Maximum packed-context token budget."""
        return int(self._file_config.get("token_budget", DEFAULT_TOKEN_BUDGET))

    @property
    def selector_threshold(self):
        """Minimum score for the anchor selector to include an anchor."""
        return float(self._file_config.get("selector_threshold", DEFAULT_SELECTOR_THRESHOLD))

    @property
    def tenant_id(self):
        """Active tenant identifier."""
        import re
        tid = (
            os.environ.get("PSA_TENANT_ID")
            or self._file_config.get("tenant_id", DEFAULT_TENANT_ID)
        )
        if not re.match(r"^[a-z0-9_-]{1,64}$", tid):
            raise ValueError(
                f"Invalid tenant_id {tid!r} from config/env. "
                "Must be 1-64 lowercase alphanumeric, hyphens, or underscores."
            )
        return tid

    @property
    def psa_mode(self):
        """
        PSA operating mode.

        "primary"      — PSA is the primary search path (default)
        "side-by-side" — PSA runs alongside raw; results from both available
        "off"          — existing raw ChromaDB search only (legacy fallback)
        """
        return (
            os.environ.get("PSA_MODE")
            or self._file_config.get("psa_mode", DEFAULT_PSA_MODE)
        )

    # ── Lifecycle settings ─────────────────────────────────────────────────

    @property
    def max_memories(self):
        """Global memory cap for forgetting system."""
        return int(self._file_config.get("max_memories", DEFAULT_MAX_MEMORIES))

    @property
    def anchor_memory_budget(self):
        """Per-anchor memory budget for pruning."""
        return int(self._file_config.get("anchor_memory_budget", DEFAULT_ANCHOR_MEMORY_BUDGET))

    @property
    def nightly_hour(self):
        """Hour (0-23) for scheduled lifecycle runs."""
        return int(self._file_config.get("nightly_hour", DEFAULT_NIGHTLY_HOUR))

    def init(self):
        """Create config directory and write default config.json if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        if not self._config_file.exists():
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "collection_name": DEFAULT_COLLECTION_NAME,
                "topic_wings": DEFAULT_TOPIC_WINGS,
                "hall_keywords": DEFAULT_HALL_KEYWORDS,
            }
            with open(self._config_file, "w") as f:
                json.dump(default_config, f, indent=2)
        return self._config_file

    def save_people_map(self, people_map):
        """Write people_map.json to config directory.

        Args:
            people_map: Dict mapping name variants to canonical names.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._people_map_file, "w") as f:
            json.dump(people_map, f, indent=2)
        return self._people_map_file
