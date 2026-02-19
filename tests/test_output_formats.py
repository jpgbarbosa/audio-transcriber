import json
from unittest.mock import patch

from transcribe import save_json, save_rttm, save_txt


class TestSaveRttm:
    @patch("transcribe.console")
    def test_writes_rttm_lines(self, mock_console, sample_result, tmp_path):
        output = tmp_path / "test.rttm"
        save_rttm(sample_result, str(output))

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("SPEAKER audio 1 0.000 2.500")
        assert "SPEAKER_00" in lines[0]

    @patch("transcribe.console")
    def test_default_speaker_when_missing(self, mock_console, sample_result_no_speakers, tmp_path):
        output = tmp_path / "test.rttm"
        save_rttm(sample_result_no_speakers, str(output))

        lines = output.read_text().strip().split("\n")
        for line in lines:
            assert "SPEAKER_00" in line

    @patch("transcribe.console")
    def test_empty_segments(self, mock_console, tmp_path):
        output = tmp_path / "test.rttm"
        save_rttm({"segments": []}, str(output))
        assert output.read_text() == ""

    @patch("transcribe.console")
    def test_missing_segments_key(self, mock_console, tmp_path):
        output = tmp_path / "test.rttm"
        save_rttm({}, str(output))
        assert output.read_text() == ""


class TestSaveTxt:
    @patch("transcribe.console")
    def test_writes_speaker_transcript(self, mock_console, sample_result, tmp_path):
        output = tmp_path / "test.txt"
        save_txt(sample_result, str(output))

        content = output.read_text()
        assert "SPEAKER_00 [00:00]:" in content
        assert "Hello, welcome to the meeting." in content
        assert "SPEAKER_01 [00:03]:" in content

    @patch("transcribe.console")
    def test_groups_consecutive_segments(self, mock_console, tmp_path):
        result = {
            "segments": [
                {"text": " Part one.", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"text": " Part two.", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
                {"text": " Response.", "start": 2.5, "end": 3.5, "speaker": "SPEAKER_01"},
            ]
        }
        output = tmp_path / "test.txt"
        save_txt(result, str(output))

        content = output.read_text()
        # SPEAKER_00 header should appear only once
        assert content.count("SPEAKER_00") == 1
        assert content.count("SPEAKER_01") == 1

    @patch("transcribe.console")
    def test_default_speaker_when_missing(self, mock_console, sample_result_no_speakers, tmp_path):
        output = tmp_path / "test.txt"
        save_txt(sample_result_no_speakers, str(output))

        content = output.read_text()
        assert "Unknown [00:00]:" in content


class TestSaveJson:
    @patch("transcribe.console")
    def test_writes_valid_json(self, mock_console, sample_result, tmp_path):
        output = tmp_path / "test.json"
        save_json(sample_result, str(output))

        loaded = json.loads(output.read_text())
        assert loaded == sample_result

    @patch("transcribe.console")
    def test_preserves_unicode(self, mock_console, tmp_path):
        result = {"segments": [{"text": "Olá, tudo bem?", "start": 0.0, "end": 1.0}]}
        output = tmp_path / "test.json"
        save_json(result, str(output))

        loaded = json.loads(output.read_text())
        assert loaded["segments"][0]["text"] == "Olá, tudo bem?"
