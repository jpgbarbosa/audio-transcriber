from unittest.mock import patch

import pytest

from transcribe import validate_audio_file


class TestValidateAudioFile:
    def test_valid_mp3(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.touch()
        with patch("transcribe.console"):
            result = validate_audio_file(str(f))
        assert result == f

    def test_valid_wav(self, tmp_path):
        f = tmp_path / "test.wav"
        f.touch()
        with patch("transcribe.console"):
            result = validate_audio_file(str(f))
        assert result == f

    def test_missing_file_exits(self, tmp_path):
        with patch("transcribe.console"), pytest.raises(SystemExit) as exc_info:
            validate_audio_file(str(tmp_path / "nonexistent.mp3"))
        assert exc_info.value.code == 1

    def test_unsupported_extension_warns(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.touch()
        with patch("transcribe.console") as mock_console:
            result = validate_audio_file(str(f))
        assert result == f
        # Should have printed a warning
        mock_console.print.assert_any_call("[yellow]Warning: .xyz may not be supported.[/yellow]")

    @pytest.mark.parametrize("ext", [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"])
    def test_all_supported_extensions(self, tmp_path, ext):
        f = tmp_path / f"test{ext}"
        f.touch()
        with patch("transcribe.console") as mock_console:
            result = validate_audio_file(str(f))
        assert result == f
        # No warning should be printed about unsupported format
        for call in mock_console.print.call_args_list:
            assert "may not be supported" not in str(call)
