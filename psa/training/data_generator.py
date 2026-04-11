"""
data_generator.py — Training data generation for the PSA anchor selector.

Generates (query, anchor_card) pairs with binary labels for cross-encoder
fine-tuning. Data is sourced from oracle-labeled queries.

Query families:
  - single_anchor: factual retrieval, one anchor covers the answer
  - contrastive: boundary questions between two similar anchors
  - compositional: requires combining 2 anchors
  - bridge: 3 anchors form a chain
  - experience: "what worked last time?" episodic retrieval

Data mix per epoch:
  60% synthetic internal queries (from oracle labels)
  20% hard-negative augmented
  20% adversarial rewrites

Output: JSONL file with records:
  {"query": "...", "anchor_card": "...", "label": 0 or 1}
"""

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("psa.training.data_generator")

# ── Query family definitions ──────────────────────────────────────────────────

QUERY_FAMILIES = [
    "single_anchor",  # factual, one anchor
    "contrastive",  # boundary between two similar anchors
    "compositional",  # two-anchor combination
    "bridge",  # three-anchor chain
    "experience",  # episodic "what worked?" queries
]

# ── Data mix ──────────────────────────────────────────────────────────────────

MIX_SYNTHETIC = 0.60
MIX_HARD_NEG = 0.20
MIX_ADVERSARIAL = 0.20

MIN_POSITIVE_RATIO = 0.25

# ── Adversarial rewrite patterns ──────────────────────────────────────────────

_ADVERSARIAL_TRANSFORMS = [
    # Lexical compression
    lambda q: re.sub(r"\b(authentication|auth)\b", "login", q, flags=re.I),
    lambda q: re.sub(r"\b(database|db)\b", "storage", q, flags=re.I),
    lambda q: re.sub(r"\bvector\b", "embedding", q, flags=re.I),
    # Conversational phrasing
    lambda q: f"hey can you remind me about {q.lower().rstrip('?.')}?",
    lambda q: f"what do we know about {q.lower().rstrip('?.')}",
    # Distractor insertion
    lambda q: q + " (not counting the recent refactor)",
    lambda q: q.replace("?", " from last quarter?") if "?" in q else q + " from our last sprint?",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class TrainingExample:
    query: str
    anchor_card: str
    label: int  # 1 = positive (oracle anchor), 0 = negative
    anchor_id: int
    query_family: str
    example_type: str  # "positive" | "hard_negative" | "easy_negative" | "adversarial"
    source_query_id: Optional[str] = None


# ── DataGenerator ─────────────────────────────────────────────────────────────


class DataGenerator:
    """
    Generates cross-encoder training examples from oracle labels.

    Usage::

        gen = DataGenerator(oracle_labels_path, anchor_cards)
        gen.generate(output_path, n_examples=12000)
    """

    def __init__(
        self,
        oracle_labels_path: str,
        anchor_cards: Dict[int, str],  # anchor_id → card text
        seed: int = 42,
    ):
        self.oracle_labels_path = oracle_labels_path
        self.anchor_cards = anchor_cards
        self.rng = random.Random(seed)
        self._labels = self._load_labels()

    def _load_labels(self) -> list:
        if not os.path.exists(self.oracle_labels_path):
            return []
        labels = []
        with open(self.oracle_labels_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        labels.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return labels

    def generate(
        self,
        output_path: str,
        n_examples: int = 12000,
        query_family_counts: Optional[Dict[str, int]] = None,
    ) -> int:
        """
        Generate training examples and write them to a JSONL file.

        Returns the number of examples written.
        """
        if not self._labels:
            logger.warning("No oracle labels found at %s", self.oracle_labels_path)
            return 0

        # Compute split sizes
        n_synthetic = int(n_examples * MIX_SYNTHETIC)
        n_hard_neg = int(n_examples * MIX_HARD_NEG)
        n_adversarial = n_examples - n_synthetic - n_hard_neg

        examples: List[TrainingExample] = []

        # 1. Synthetic positives from oracle labels
        examples.extend(self._generate_positives(n_synthetic // 2))
        examples.extend(self._generate_easy_negatives(n_synthetic // 2))

        # 2. Hard negatives
        examples.extend(self._generate_hard_negatives(n_hard_neg))

        # 3. Adversarial rewrites
        examples.extend(self._generate_adversarial(n_adversarial))

        # 4. Enforce minimum positive ratio
        positives = [e for e in examples if e.label == 1]
        negatives = [e for e in examples if e.label == 0]
        pos_ratio = len(positives) / len(examples) if examples else 0

        if pos_ratio < 0.10:
            logger.warning(
                "Very low positive ratio (%.1f%%) — oracle labels may lack "
                "winning_oracle_set entries. Training quality will be poor.",
                pos_ratio * 100,
            )

        if positives and pos_ratio < MIN_POSITIVE_RATIO:
            target_pos = int(len(negatives) * MIN_POSITIVE_RATIO / (1 - MIN_POSITIVE_RATIO))
            extra_needed = target_pos - len(positives)
            if extra_needed > 0:
                oversampled = [self.rng.choice(positives) for _ in range(extra_needed)]
                examples = positives + oversampled + negatives
                logger.info(
                    "Oversampled %d positives to reach %.0f%% positive ratio.",
                    extra_needed,
                    MIN_POSITIVE_RATIO * 100,
                )

        self.rng.shuffle(examples)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        written = 0
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(
                    json.dumps(
                        {
                            "query": ex.query,
                            "anchor_card": ex.anchor_card,
                            "label": ex.label,
                            "anchor_id": ex.anchor_id,
                            "query_family": ex.query_family,
                            "example_type": ex.example_type,
                            "source_query_id": ex.source_query_id,
                        }
                    )
                    + "\n"
                )
                written += 1

        logger.info("Generated %d training examples at %s", written, output_path)
        return written

    def _infer_family(self, label: dict) -> str:
        """Infer query family from winning oracle set size."""
        winning = label.get("winning_oracle_set", [])
        n = len(winning)
        if label.get("is_high_complexity"):
            return "bridge"
        if n == 1:
            return self.rng.choice(["single_anchor", "experience"])
        if n == 2:
            return self.rng.choice(["compositional", "contrastive"])
        return "bridge"

    def _generate_positives(self, n: int) -> List[TrainingExample]:
        """Create positive examples from oracle-winning anchor sets."""
        examples = []
        for _ in range(n):
            label = self.rng.choice(self._labels)
            query = label["query"]
            winning_ids = label.get("winning_oracle_set", [])
            if not winning_ids:
                continue
            anchor_id = self.rng.choice(winning_ids)
            card_text = self.anchor_cards.get(anchor_id, "")
            if not card_text:
                continue
            examples.append(
                TrainingExample(
                    query=query,
                    anchor_card=card_text,
                    label=1,
                    anchor_id=anchor_id,
                    query_family=self._infer_family(label),
                    example_type="positive",
                    source_query_id=label.get("query_id"),
                )
            )
        return examples

    def _generate_easy_negatives(self, n: int) -> List[TrainingExample]:
        """Create easy negatives: random anchors not in the oracle set."""
        all_anchor_ids = list(self.anchor_cards.keys())
        examples = []
        for _ in range(n):
            label = self.rng.choice(self._labels)
            query = label["query"]
            winning_ids = set(label.get("winning_oracle_set", []))
            negatives = [aid for aid in all_anchor_ids if aid not in winning_ids]
            if not negatives:
                continue
            anchor_id = self.rng.choice(negatives)
            card_text = self.anchor_cards.get(anchor_id, "")
            if not card_text:
                continue
            examples.append(
                TrainingExample(
                    query=query,
                    anchor_card=card_text,
                    label=0,
                    anchor_id=anchor_id,
                    query_family=self._infer_family(label),
                    example_type="easy_negative",
                    source_query_id=label.get("query_id"),
                )
            )
        return examples

    def _generate_hard_negatives(self, n: int) -> List[TrainingExample]:
        """
        Hard negatives: shortlisted anchors that were NOT in the oracle set.

        These are the anchors the retriever found relevant but the oracle
        rejected — the hardest negatives.
        """
        examples = []
        for _ in range(n):
            label = self.rng.choice(self._labels)
            query = label["query"]
            winning_ids = set(label.get("winning_oracle_set", []))
            candidate_ids = label.get("candidate_anchor_ids", [])
            hard_neg_ids = [aid for aid in candidate_ids if aid not in winning_ids]
            if not hard_neg_ids:
                continue
            anchor_id = self.rng.choice(hard_neg_ids)
            card_text = self.anchor_cards.get(anchor_id, "")
            if not card_text:
                continue
            examples.append(
                TrainingExample(
                    query=query,
                    anchor_card=card_text,
                    label=0,
                    anchor_id=anchor_id,
                    query_family=self._infer_family(label),
                    example_type="hard_negative",
                    source_query_id=label.get("query_id"),
                )
            )
        return examples

    def _generate_adversarial(self, n: int) -> List[TrainingExample]:
        """
        Adversarial rewrites of positive examples.

        Apply random lexical transforms to queries to make them harder.
        """
        positives = self._generate_positives(n)
        examples = []
        for ex in positives:
            transform = self.rng.choice(_ADVERSARIAL_TRANSFORMS)
            try:
                rewritten = transform(ex.query)
            except Exception:
                rewritten = ex.query
            examples.append(
                TrainingExample(
                    query=rewritten,
                    anchor_card=ex.anchor_card,
                    label=ex.label,
                    anchor_id=ex.anchor_id,
                    query_family=ex.query_family,
                    example_type="adversarial",
                    source_query_id=ex.source_query_id,
                )
            )
        return examples

    def query_family_summary(self) -> Dict[str, int]:
        """Count oracle labels per inferred query family."""
        counts: Dict[str, int] = {f: 0 for f in QUERY_FAMILIES}
        for label in self._labels:
            family = self._infer_family(label)
            if family in counts:
                counts[family] += 1
        return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSA training data generator")
    parser.add_argument("--tenant", default="default", help="Tenant ID (default: default)")
    parser.add_argument("--labels", required=True, help="Path to oracle_labels.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path for training data")
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Total samples to generate (default: 1000)"
    )
    args = parser.parse_args()

    from psa.atlas import AtlasManager
    from psa.tenant import TenantManager

    tm = TenantManager()
    tenant = tm.get_or_create(args.tenant)
    atlas_mgr = AtlasManager(tenant_dir=tenant.root_dir, tenant_id=args.tenant)
    atlas = atlas_mgr.get_atlas()
    if atlas is None:
        print(f"No atlas for tenant '{args.tenant}'. Run 'psa atlas build' first.")
        raise SystemExit(1)

    # Build anchor_cards dict: {anchor_id: card_text}
    anchor_cards = {c.anchor_id: c.to_stable_card_text() for c in atlas.cards}

    gen = DataGenerator(oracle_labels_path=args.labels, anchor_cards=anchor_cards)
    n_written = gen.generate(output_path=args.output, n_examples=args.n_samples)
    print(f"Generated {n_written} training samples → {args.output}")
