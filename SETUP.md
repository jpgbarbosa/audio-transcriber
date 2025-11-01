# Quick Setup Guide

Follow these steps to get the transcriber running on your M1 Mac.

## 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages (this may take a few minutes)
pip install -r requirements.txt
```

## 2. Get HuggingFace Token

### Create Account & Token
1. Go to https://huggingface.co/join
2. Create a free account
3. Go to https://huggingface.co/settings/tokens
4. Click "New token"
5. Give it a name (e.g., "transcriber")
6. Select "Read" permission
7. Copy the token (starts with `hf_...`)

### Accept Model Terms
1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
2. Click "Agree and access repository"
3. Visit https://huggingface.co/pyannote/segmentation-3.0
4. Click "Agree and access repository"

## 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and paste your token
# Change this line:
#   HF_TOKEN=your_huggingface_token_here
# To something like:
#   HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890
```

## 4. Test Installation

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # If not already activated

# Test with a small audio file (if you have one)
python transcribe.py your_audio.mp3 --model small

# Or check the help
python transcribe.py --help
```

## Expected First Run

The first time you run the transcriber:
1. It will download models (~5GB total)
2. This can take 10-30 minutes depending on your internet
3. Models are cached locally - subsequent runs are much faster
4. You'll see download progress in the terminal

## Common Issues

### "No module named 'whisperx'"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall if needed
pip install -r requirements.txt
```

### "HF_TOKEN not found"
```bash
# Check your .env file exists and has the token
cat .env

# Should show:
# HF_TOKEN=hf_your_actual_token_here
```

### "pyannote model access denied"
- Make sure you accepted the terms for BOTH models:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

### "Out of memory"
```bash
# Use a smaller model
python transcribe.py audio.mp3 --model medium
# or
python transcribe.py audio.mp3 --model small
```

## Quick Reference

```bash
# Basic usage (auto-detect language)
python transcribe.py meeting.mp3

# Specify Portuguese
python transcribe.py meeting.mp3 --language pt

# Specify English
python transcribe.py meeting.mp3 --language en

# With speaker count
python transcribe.py meeting.mp3 --min-speakers 2 --max-speakers 4

# Custom output location
python transcribe.py meeting.mp3 --output ./transcripts/

# Use smaller/faster model
python transcribe.py meeting.mp3 --model medium
```

## Your M1 Mac Specifications

- **Device**: MacBook M1 Pro with 32GB RAM ✓
- **Acceleration**: MPS (Metal Performance Shaders) ✓
- **Recommended model**: `large-v3` (best accuracy)
- **Expected speed**: 5-10x realtime
- **Example**: 30-minute meeting → 3-6 minutes processing time

## Next Steps

Once setup is complete:
1. Read the full README.md for detailed usage
2. Try transcribing a test audio file
3. Experiment with different models and options
4. Check the example_batch.sh for batch processing

## Support

If you encounter issues:
1. Check the Troubleshooting section in README.md
2. Ensure all steps above were completed
3. Try with a smaller model first (`--model small`)
4. Check that your audio file is in a supported format
