# Meeting Transcriber

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A command-line tool for transcribing audio recordings with automatic speaker diarization (identifying who spoke when). Optimized for meeting recordings with multiple speakers from a single microphone.

## Features

- **High-Accuracy Transcription**: Uses OpenAI's Whisper (via WhisperX) for state-of-the-art speech recognition
- **Speaker Diarization**: Automatically identifies and labels different speakers using pyannote.audio
- **Multi-Language Support**: Excellent support for Portuguese and English (and 97 other languages)
- **Fast Processing**: WhisperX provides 70x realtime transcription speed with GPU/M1 acceleration
- **Multiple Output Formats**: RTTM (standard diarization), human-readable text, and JSON
- **Local Processing**: All processing happens on your machine - no cloud services required

## Requirements

- Python 3.8 or higher
- MacBook M1/M2/M3 (MPS acceleration) OR NVIDIA GPU (CUDA) OR CPU (slower)
- ~5GB disk space for models
- HuggingFace account (free) for speaker diarization models

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/transcriber.git
cd transcriber
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get HuggingFace Token (Required for Speaker Diarization)

1. Create a free account at [HuggingFace](https://huggingface.co/join)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)
4. Accept the terms for pyannote models:
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Agree and access repository"
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Click "Agree and access repository"

### 5. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token
```

Your `.env` file should look like:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Usage

### Basic Usage

Transcribe an audio file with automatic language detection:

```bash
python transcribe.py meeting.mp3
```

This will create three files in the same directory as your audio:
- `meeting.rttm` - Standard diarization format (who spoke when)
- `meeting.txt` - Human-readable transcript with speaker labels
- `meeting.json` - Complete data with timestamps and metadata

### Specify Language

For faster processing and better accuracy, specify the language:

```bash
# Portuguese
python transcribe.py meeting.mp3 --language pt

# English
python transcribe.py meeting.mp3 --language en
```

### Specify Number of Speakers

If you know how many speakers are in the recording:

```bash
# Exactly 3 speakers
python transcribe.py meeting.mp3 --min-speakers 3 --max-speakers 3

# Between 2 and 5 speakers
python transcribe.py meeting.mp3 --min-speakers 2 --max-speakers 5
```

### Choose Model Size

Larger models are more accurate but slower and require more memory:

```bash
# Fastest (least accurate) - good for testing
python transcribe.py meeting.mp3 --model tiny

# Balanced
python transcribe.py meeting.mp3 --model medium

# Best accuracy (default, recommended)
python transcribe.py meeting.mp3 --model large-v3
```

**Model Comparison:**
| Model | Size | Speed | Accuracy | M1 32GB Compatible |
|-------|------|-------|----------|-------------------|
| tiny | 39M | Fastest | Good | ✓ |
| base | 74M | Very Fast | Good | ✓ |
| small | 244M | Fast | Better | ✓ |
| medium | 769M | Medium | Great | ✓ |
| large-v2 | 1.5GB | Slower | Excellent | ✓ |
| large-v3 | 1.5GB | Slower | Best | ✓ |

### Custom Output Location

```bash
python transcribe.py meeting.mp3 --output ./transcripts/
```

### Specific Output Formats

```bash
# Only RTTM
python transcribe.py meeting.mp3 --format rttm

# RTTM and text only
python transcribe.py meeting.mp3 --format rttm --format txt

# All formats (default)
python transcribe.py meeting.mp3 --format all
```

### Complete Example

```bash
python transcribe.py meeting_recording.mp3 \
  --language pt \
  --model large-v3 \
  --min-speakers 2 \
  --max-speakers 4 \
  --output ./transcripts/ \
  --format all
```

## Output Formats

### RTTM (Rich Transcription Time Marked)

Standard format for speaker diarization. Each line represents when a speaker was talking:

```
SPEAKER audio 1 0.000 5.123 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio 1 5.123 3.456 <NA> <NA> SPEAKER_01 <NA> <NA>
```

Format: `SPEAKER <file> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>`

### TXT (Human-Readable)

Easy-to-read transcript with speaker labels and timestamps:

```
SPEAKER_00 [00:00]:
Hello everyone, welcome to today's meeting.

SPEAKER_01 [00:05]:
Thanks for having me. I'd like to discuss the project timeline.

SPEAKER_00 [00:12]:
Of course, let's start with the current status.
```

### JSON (Complete Data)

Full transcription data including word-level timestamps, confidence scores, and all metadata:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.123,
      "text": "Hello everyone, welcome to today's meeting.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.95},
        ...
      ]
    }
  ]
}
```

## Supported Audio Formats

- MP3 (`.mp3`)
- WAV (`.wav`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG (`.ogg`)
- OPUS (`.opus`)
- WebM (`.webm`)

## Performance Tips

### For M1 Mac (Your Setup)

Your M1 Pro with 32GB RAM is excellent for this task:

- **Recommended model**: `large-v3` (best accuracy, still fast on M1)
- **Expected speed**: ~5-10x realtime (a 30-minute recording takes 3-6 minutes)
- **Memory usage**: ~4-6GB for large-v3 model

### For CPU-Only Systems

If you don't have a GPU:

- Use `--model medium` or `--model small` for faster processing
- Expect 0.5-2x realtime speed (a 30-minute recording may take 15-60 minutes)

### For NVIDIA GPU

If you have CUDA available:

- Will automatically use GPU acceleration
- `large-v3` can process at 50-70x realtime speed

## Troubleshooting

### "HF_TOKEN not found" Warning

You need to set up your HuggingFace token for speaker diarization:
1. Follow step 4 in Installation
2. Create a `.env` file with your token
3. Make sure you've accepted the pyannote model agreements

### Out of Memory Errors

- Try a smaller model: `--model medium` or `--model small`
- Close other applications
- Process shorter audio segments

### Poor Diarization Results

- Specify the number of speakers: `--min-speakers 2 --max-speakers 3`
- Ensure audio quality is good (clear speech, minimal background noise)
- Try processing with `--language` specified instead of auto-detection

### Slow Processing

- Verify GPU/MPS acceleration is being used (check console output)
- Use a smaller model: `--model medium`
- Ensure no other heavy applications are running

## Advanced Usage

### Batch Processing

Process multiple files:

```bash
for file in recordings/*.mp3; do
  python transcribe.py "$file" --language pt --output transcripts/
done
```

### Integration with Other Tools

The JSON output can be easily parsed by other programs:

```python
import json

with open('meeting.json', 'r') as f:
    data = json.load(f)

for segment in data['segments']:
    print(f"{segment['speaker']}: {segment['text']}")
```

## Technical Details

### Architecture

1. **Audio Loading**: Loads and preprocesses audio to 16kHz mono
2. **Transcription**: WhisperX transcribes speech to text
3. **Alignment**: Forced alignment provides precise word-level timestamps
4. **Diarization**: pyannote.audio identifies different speakers
5. **Assignment**: Speaker labels are assigned to transcribed segments

### Models Used

- **Whisper**: OpenAI's open-source speech recognition model
- **WhisperX**: Optimized pipeline for faster processing and better timestamps
- **pyannote.audio 3.1**: State-of-the-art speaker diarization
- **Wav2Vec2**: Forced alignment for precise timestamps

## Privacy & Security

- All processing happens locally on your machine
- No audio data is sent to external services
- HuggingFace token is only used to download models (one-time)
- Models are cached locally after first download

## License

This project uses several open-source components:
- WhisperX: BSD License
- pyannote.audio: MIT License
- OpenAI Whisper: MIT License

## Contributing

Suggestions and improvements are welcome! Feel free to open issues or submit pull requests.

## Credits

Built with:
- [WhisperX](https://github.com/m-bain/whisperX) by Max Bain
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) by Hervé Bredin
- [OpenAI Whisper](https://github.com/openai/whisper) by OpenAI
