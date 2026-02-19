import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock heavy ML dependencies before transcribe.py is imported by any test.
# pytest processes conftest.py before collecting test modules, so these mocks
# are in place by the time `from transcribe import ...` runs.
# ---------------------------------------------------------------------------
_HEAVY_MODULES = [
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
    "whisperx",
    "soundfile",
    "pandas",
    "dotenv",
    "pyannote",
    "pyannote.audio",
    "rich",
    "rich.console",
    "rich.progress",
]

for _mod in _HEAVY_MODULES:
    sys.modules.setdefault(_mod, MagicMock())


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_result():
    """Sample transcription result with speaker diarization."""
    return {
        "segments": [
            {
                "text": " Hello, welcome to the meeting.",
                "start": 0.0,
                "end": 2.5,
                "speaker": "SPEAKER_00",
            },
            {
                "text": " Thank you for having me.",
                "start": 3.0,
                "end": 5.0,
                "speaker": "SPEAKER_01",
            },
            {
                "text": " Let's get started with the agenda.",
                "start": 5.5,
                "end": 8.0,
                "speaker": "SPEAKER_00",
            },
        ],
        "language": "en",
    }


@pytest.fixture
def sample_result_no_speakers():
    """Sample transcription result without speaker labels."""
    return {
        "segments": [
            {
                "text": " Hello, welcome to the meeting.",
                "start": 0.0,
                "end": 2.5,
            },
            {
                "text": " Thank you for having me.",
                "start": 3.0,
                "end": 5.0,
            },
        ],
    }
