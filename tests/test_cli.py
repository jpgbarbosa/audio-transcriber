from click.testing import CliRunner

from transcribe import main


class TestCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Transcribe audio files" in result.output

    def test_help_shows_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--language" in result.output
        assert "--model" in result.output
        assert "--backend" in result.output
        assert "--device" in result.output
        assert "--format" in result.output
        assert "--output" in result.output

    def test_missing_audio_file(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_nonexistent_audio_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/path/file.mp3"])
        assert result.exit_code != 0
