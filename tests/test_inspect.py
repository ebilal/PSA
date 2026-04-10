"""Tests for psa.inspect — InspectResult construction and query_log."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from psa.anchor import AnchorCard, AnchorIndex
from psa.atlas import Atlas
from psa.embeddings import EmbeddingModel
from psa.inspect import (
    CandidateTrace,
    InspectResult,
    PackerSectionTrace,
    _run_id,
    inspect_query,
    load_log,
)
from psa.memory_object import MemoryStore
from psa.pipeline import PSAPipeline, QueryTiming
from psa.retriever import AnchorCandidate
from psa.selector import AnchorSelector, SelectedAnchor


def _make_card(anchor_id: int, name: str = "auth") -> AnchorCard:
    return AnchorCard(
        anchor_id=anchor_id,
        name=name,
        meaning=f"Anchor about {name}",
        memory_types=["EPISODIC"],
        include_terms=[name],
        exclude_terms=[],
        prototype_examples=[],
        near_but_different=[],
        centroid=[1.0, 0.0],
        memory_count=2,
        is_novelty=False,
    )


def _make_pipeline(tmp_path):
    card = _make_card(1)
    index = AnchorIndex(dim=2)
    card.centroid = [1.0, 0.0]
    index.build([card])

    atlas = MagicMock(spec=Atlas)
    atlas.cards = [card]
    atlas.anchor_index = index
    atlas.version = 1

    store = MagicMock(spec=MemoryStore)
    store.query_by_anchor.return_value = []

    em = MagicMock(spec=EmbeddingModel)
    em.embed.return_value = [1.0, 0.0]

    return PSAPipeline(
        store=store,
        atlas=atlas,
        embedding_model=em,
        selector=AnchorSelector.cosine(max_k=2),
        token_budget=1000,
        tenant_id="test",
    )


def test_run_id_format():
    rid = _run_id("my query")
    parts = rid.split("_")
    assert len(parts) == 2
    assert len(parts[0]) == 15   # YYYYMMDDTHHmmSS
    assert len(parts[1]) == 6


def test_run_id_same_hash_for_same_query():
    rid1 = _run_id("hello world")
    rid2 = _run_id("hello world")
    assert rid1.split("_")[1] == rid2.split("_")[1]


def test_inspect_query_returns_inspect_result(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    assert isinstance(result, InspectResult)
    assert result.query == "test query"
    assert result.tenant_id == "default"


def test_inspect_result_has_context_text(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    assert isinstance(result.context_text, str)
    assert len(result.context_text) > 0


def test_inspect_result_selected_is_subset_of_candidates(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    selected_ids = set(result.selected_anchor_ids)
    candidate_ids = {c.anchor_id for c in result.candidates}
    assert selected_ids.issubset(candidate_ids)


def test_inspect_result_to_dict_has_required_keys(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("test query", base_dir=str(tmp_path), write_log=False)
    d = result.to_dict()
    for key in ("run_id", "query", "tenant_id", "context_text", "tokens_used",
                 "token_budget", "sections", "selected_anchor_ids", "candidates", "timing"):
        assert key in d, f"Missing key: {key}"


def test_render_brief_contains_query(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("auth query", base_dir=str(tmp_path), write_log=False)
    brief = result.render_brief()
    assert "auth query" in brief
    assert "tokens" in brief.lower()


def test_render_verbose_contains_candidates_section(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        result = inspect_query("auth query", base_dir=str(tmp_path), write_log=False)
    verbose = result.render_verbose()
    assert "ANCHOR CANDIDATES" in verbose


def test_inspect_writes_to_log(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        inspect_query("log test query", base_dir=str(tmp_path), write_log=True)
    log_path = tmp_path / "tenants" / "default" / "query_log.jsonl"
    assert log_path.exists()
    line = log_path.read_text().strip()
    entry = json.loads(line)
    assert entry["query"] == "log test query"


def test_load_log_returns_newest_first(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    with patch("psa.inspect.PSAPipeline.from_tenant", return_value=pipeline):
        inspect_query("first query", base_dir=str(tmp_path), write_log=True)
        inspect_query("second query", base_dir=str(tmp_path), write_log=True)
    entries = load_log(tenant_id="default", base_dir=str(tmp_path))
    assert len(entries) == 2
    assert entries[0]["query"] == "second query"
    assert entries[1]["query"] == "first query"


def test_load_log_empty_if_no_file(tmp_path):
    entries = load_log(tenant_id="no_such_tenant", base_dir=str(tmp_path))
    assert entries == []
