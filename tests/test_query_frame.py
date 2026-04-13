"""
test_query_frame.py — Tests for query frame extraction.
"""

from psa.query_frame import extract_query_frame


class TestPatternMatcher:
    def test_procedure_query(self):
        frame = extract_query_frame("How do I configure JWT tokens?")
        assert frame.answer_target == "procedure"
        assert frame.retrieval_mode == "single_hop"

    def test_failure_query(self):
        frame = extract_query_frame("What went wrong with the auth deployment?")
        assert frame.answer_target == "failure"

    def test_temporal_change_query(self):
        frame = extract_query_frame("How has our auth pattern changed over time?")
        assert frame.answer_target == "temporal_change"
        assert frame.retrieval_mode == "compare_over_time"

    def test_speaker_constraint(self):
        frame = extract_query_frame("What did Alice say about the database?")
        assert frame.entity_constraint == "Alice"
        assert frame.answer_target == "prior_statement"

    def test_entity_extraction(self):
        frame = extract_query_frame("Why did we switch to GraphQL?")
        assert "GraphQL" in frame.entities

    def test_time_constraint(self):
        frame = extract_query_frame("What auth changes were made last week?")
        assert frame.time_constraint is not None
        assert frame.answer_target == "temporal_change"

    def test_simple_fact_query(self):
        frame = extract_query_frame("What database do we use?")
        assert frame.answer_target == "fact"
        assert frame.retrieval_mode == "single_hop"

    def test_confidence_above_threshold(self):
        frame = extract_query_frame("How do I set up the GraphQL server?")
        assert frame.confidence >= 0.6

    def test_default_frame_for_ambiguous(self):
        frame = extract_query_frame("hmm")
        assert frame.answer_target == "fact"
        assert frame.retrieval_mode == "single_hop"
