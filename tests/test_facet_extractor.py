"""Tests for the heuristic facet extractor."""


from psa.facet_extractor import extract_facets


class TestEntityExtraction:
    def test_extracts_camel_case(self):
        result = extract_facets("We migrated to GraphQL")
        assert "GraphQL" in result.entities

    def test_extracts_quoted_strings(self):
        result = extract_facets('The "auth-service" endpoint')
        assert "auth-service" in result.entities

    def test_extracts_file_paths(self):
        result = extract_facets("Check src/auth/jwt.py")
        assert "src/auth/jwt.py" in result.entities


class TestTemporalExtraction:
    def test_extracts_date_pattern(self):
        result = extract_facets("happened in 2026-03")
        assert result.mentioned_at is not None
        assert "2026-03" in result.mentioned_at

    def test_extracts_relative_time(self):
        result = extract_facets("changed this last week")
        assert result.mentioned_at is not None


class TestSpeakerExtraction:
    def test_user_marker(self):
        result = extract_facets("> Can you fix the auth module?")
        assert result.speaker_role == "user"

    def test_assistant_default(self):
        result = extract_facets("I recommend using JWT for authentication.")
        assert result.speaker_role == "assistant"


class TestStanceExtraction:
    def test_switched(self):
        result = extract_facets("switched from REST to GraphQL")
        assert result.stance == "switched"

    def test_deprecated(self):
        result = extract_facets("old auth module was deprecated")
        assert result.stance == "deprecated"

    def test_no_stance(self):
        result = extract_facets("JWT tokens expire after 24 hours")
        assert result.stance is None


class TestActorEntities:
    def test_extracts_named_person(self):
        result = extract_facets("Alice said we should use GraphQL")
        assert "Alice" in result.actor_entities

    def test_no_actors(self):
        result = extract_facets("The system uses JWT")
        assert result.actor_entities == []
