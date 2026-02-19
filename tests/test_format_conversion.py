from unittest.mock import MagicMock, patch

from transcribe import TranscriptionPipeline


class TestTranscribeWithTransformers:
    def _make_pipeline(self):
        """Create a TranscriptionPipeline with mocked internals."""
        with patch("transcribe.torch") as mock_torch, patch("transcribe.console"):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            p = TranscriptionPipeline(device="cpu", backend="transformers")
        return p

    def test_converts_hf_chunks_to_segments(self):
        p = self._make_pipeline()
        p.hf_pipe = MagicMock()
        p.hf_pipe.return_value = {
            "text": "Hello world. How are you?",
            "chunks": [
                {"text": "Hello world.", "timestamp": (0.0, 2.5)},
                {"text": " How are you?", "timestamp": (2.5, 5.0)},
            ],
        }

        result = p._transcribe_with_transformers("fake.mp3", language="en")

        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello world."
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.5
        assert result["segments"][1]["text"] == " How are you?"
        assert result["language"] == "en"

    def test_skips_chunks_with_none_timestamps(self):
        p = self._make_pipeline()
        p.hf_pipe = MagicMock()
        p.hf_pipe.return_value = {
            "text": "Hello",
            "chunks": [
                {"text": "Hello", "timestamp": (0.0, 1.0)},
                {"text": " world", "timestamp": None},
                {"text": " end", "timestamp": (2.0, None)},
            ],
        }

        result = p._transcribe_with_transformers("fake.mp3")
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello"

    def test_empty_chunks(self):
        p = self._make_pipeline()
        p.hf_pipe = MagicMock()
        p.hf_pipe.return_value = {"text": "", "chunks": []}

        result = p._transcribe_with_transformers("fake.mp3")
        assert result["segments"] == []

    def test_no_chunks_key(self):
        p = self._make_pipeline()
        p.hf_pipe = MagicMock()
        p.hf_pipe.return_value = {"text": "Hello"}

        result = p._transcribe_with_transformers("fake.mp3")
        assert result["segments"] == []

    def test_language_auto_when_not_specified(self):
        p = self._make_pipeline()
        p.hf_pipe = MagicMock()
        p.hf_pipe.return_value = {"text": "", "chunks": []}

        result = p._transcribe_with_transformers("fake.mp3")
        assert result["language"] == "auto"

    def test_passes_language_to_generate_kwargs(self):
        p = self._make_pipeline()
        p.hf_pipe = MagicMock()
        p.hf_pipe.return_value = {"text": "", "chunks": []}

        p._transcribe_with_transformers("fake.mp3", language="pt")
        call_kwargs = p.hf_pipe.call_args
        assert call_kwargs[1]["generate_kwargs"] == {"language": "pt"}
