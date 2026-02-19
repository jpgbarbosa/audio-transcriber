from unittest.mock import patch

from transcribe import TranscriptionPipeline


class TestPipelineInit:
    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_auto_device_cuda(self, mock_torch, mock_console):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        p = TranscriptionPipeline(device="auto")
        assert p.device == "cuda"
        assert p.compute_type == "float16"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_auto_device_mps_whisperx(self, mock_torch, mock_console):
        """WhisperX backend on MPS falls back to CPU (CTranslate2 doesn't support MPS)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        p = TranscriptionPipeline(device="auto", backend="whisperx")
        assert p.device == "cpu"
        assert p.compute_type == "int8"
        assert p.diarize_device == "mps"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_auto_device_mps_transformers(self, mock_torch, mock_console):
        """Transformers backend uses MPS directly."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        p = TranscriptionPipeline(device="auto", backend="transformers")
        assert p.device == "mps"
        assert p.compute_type == "float32"
        assert p.diarize_device == "mps"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_auto_device_mps_mlx(self, mock_torch, mock_console):
        """MLX backend on MPS uses CPU for whisperx (MLX handles its own device)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        p = TranscriptionPipeline(device="auto", backend="mlx")
        assert p.device == "cpu"
        assert p.compute_type == "int8"
        assert p.diarize_device == "mps"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_auto_device_cpu_only(self, mock_torch, mock_console):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        p = TranscriptionPipeline(device="auto")
        assert p.device == "cpu"
        assert p.compute_type == "int8"
        assert p.diarize_device == "cpu"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_explicit_cpu_device(self, mock_torch, mock_console):
        mock_torch.backends.mps.is_available.return_value = False

        p = TranscriptionPipeline(device="cpu")
        assert p.device == "cpu"
        assert p.compute_type == "int8"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_hf_token_from_env(self, mock_torch, mock_console):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("os.environ", {"HF_TOKEN": "test-token-123"}):
            p = TranscriptionPipeline(device="cpu")
        assert p.hf_token == "test-token-123"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_hf_token_explicit_overrides_env(self, mock_torch, mock_console):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            p = TranscriptionPipeline(device="cpu", hf_token="explicit-token")
        assert p.hf_token == "explicit-token"

    @patch("transcribe.console")
    @patch("transcribe.torch")
    def test_model_defaults(self, mock_torch, mock_console):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        p = TranscriptionPipeline(device="cpu")
        assert p.model_size == "large-v3"
        assert p.backend == "whisperx"
        assert p.model is None
        assert p.hf_pipe is None
        assert p.diarize_model is None
