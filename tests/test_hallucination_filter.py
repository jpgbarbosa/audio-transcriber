"""Tests for the hallucination filter."""

import copy

from transcribe import filter_hallucinations


def _make_word(word, probability=0.8):
    """Helper to create a word dict with probability."""
    return {"word": word, "start": 0.0, "end": 0.5, "probability": probability}


def _make_word_score(word, score=0.8):
    """Helper to create a word dict with score key (WhisperX style)."""
    return {"word": word, "start": 0.0, "end": 0.5, "score": score}


def _make_segment(words=None, text=None, start=0.0, end=1.0):
    """Helper to create a segment dict."""
    seg = {"start": start, "end": end}
    if words is not None:
        seg["words"] = words
        if text is None:
            seg["text"] = " ".join(w["word"].strip() for w in words)
        else:
            seg["text"] = text
    elif text is not None:
        seg["text"] = text
    return seg


class TestLowProbabilityFiltering:
    def test_low_probability_words_removed(self):
        words = [
            _make_word("Hello", 0.9),
            _make_word("world", 0.02),
            _make_word("today", 0.8),
        ]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        assert remaining == ["Hello", "today"]

    def test_high_probability_words_kept(self):
        words = [_make_word("Hello", 0.9), _make_word("world", 0.5)]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        assert remaining == ["Hello", "world"]

    def test_score_key_handled(self):
        words = [
            _make_word_score("Hello", 0.9),
            _make_word_score("um", 0.03),
            _make_word_score("world", 0.7),
        ]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        assert remaining == ["Hello", "world"]

    def test_word_without_probability_kept(self):
        words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        assert len(filtered["segments"][0]["words"]) == 1


class TestRepetitionCollapsing:
    def test_consecutive_identical_words_collapsed(self):
        words = [_make_word(w) for w in ["the", "the", "the", "the", "the", "end"]]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        assert remaining == ["the", "the", "the", "end"]

    def test_words_within_limit_kept(self):
        words = [_make_word(w) for w in ["the", "the", "end"]]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        assert remaining == ["the", "the", "end"]

    def test_case_insensitive_collapsing(self):
        words = [_make_word(w) for w in ["The", "the", "THE", "the", "the", "end"]]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        # First 3 kept (max_word_repetitions=3), rest collapsed
        assert remaining == ["The", "the", "THE", "end"]


class TestMixedFiltering:
    def test_probability_and_repetition_combined(self):
        words = [
            _make_word("good", 0.8),
            _make_word("um", 0.02),  # low prob → removed
            _make_word("yes", 0.8),
            _make_word("yes", 0.8),
            _make_word("yes", 0.8),
            _make_word("yes", 0.8),  # 4th repetition → collapsed
            _make_word("ok", 0.7),
        ]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        remaining = [w["word"] for w in filtered["segments"][0]["words"]]
        assert remaining == ["good", "yes", "yes", "yes", "ok"]


class TestEmptySegmentRemoval:
    def test_empty_segment_after_filtering_removed(self):
        words = [_make_word("um", 0.01), _make_word("uh", 0.02)]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        assert filtered["segments"] == []

    def test_empty_segments_list(self):
        result = {"segments": []}
        filtered = filter_hallucinations(result)
        assert filtered["segments"] == []


class TestSegmentsWithoutWords:
    def test_regex_repetition_collapse(self):
        text = "hello hello hello hello hello world"
        result = {"segments": [_make_segment(text=text)]}
        filtered = filter_hallucinations(result)
        assert filtered["segments"][0]["text"] == "hello world"

    def test_no_repetition_preserved(self):
        text = "hello world today"
        result = {"segments": [_make_segment(text=text)]}
        filtered = filter_hallucinations(result)
        assert filtered["segments"][0]["text"] == "hello world today"

    def test_empty_text_segment_removed(self):
        result = {"segments": [_make_segment(text="")]}
        filtered = filter_hallucinations(result)
        assert filtered["segments"] == []


class TestDeepCopy:
    def test_original_result_not_mutated(self):
        words = [_make_word("hello", 0.9), _make_word("um", 0.01)]
        result = {"segments": [_make_segment(words)]}
        original = copy.deepcopy(result)
        filter_hallucinations(result)
        assert result == original


class TestEdgeCases:
    def test_segment_with_no_words_key_and_no_text(self):
        result = {"segments": [{"start": 0.0, "end": 1.0}]}
        filtered = filter_hallucinations(result)
        # No words key and no text → segment has no text to check, words is None
        # The segment goes through the else branch, text is "" → stripped → empty → removed
        assert filtered["segments"] == []

    def test_all_words_filtered(self):
        words = [_make_word("um", 0.01), _make_word("uh", 0.02), _make_word("ah", 0.03)]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        assert filtered["segments"] == []

    def test_text_rebuilt_from_remaining_words(self):
        words = [_make_word("Hello", 0.9), _make_word("beautiful", 0.01), _make_word("world", 0.8)]
        result = {"segments": [_make_segment(words)]}
        filtered = filter_hallucinations(result)
        assert filtered["segments"][0]["text"] == "Hello world"
