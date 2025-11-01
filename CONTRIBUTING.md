# Contributing to Meeting Transcriber

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include your Python version, OS, and hardware specs
- Provide a minimal reproducible example
- Share relevant error messages and logs

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider implementation complexity and scope

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly on your hardware
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/transcriber.git
cd transcriber

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your HuggingFace token to .env
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for classes and functions
- Keep functions focused and concise

## Testing

- Test on different hardware (M1/M2, NVIDIA GPU, CPU-only)
- Test with various audio formats and languages
- Verify output formats (RTTM, TXT, JSON)

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Reference issues when applicable (#123)
- Example: "Fix speaker diarization for short audio files (#42)"

## Questions?

Feel free to open an issue for any questions about contributing!
