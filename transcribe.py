#!/usr/bin/env python3
"""
Meeting Transcriber CLI
Transcribes audio files with speaker diarization using WhisperX and pyannote.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import click
import torch
import whisperx
import soundfile as sf
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from pyannote.audio import Pipeline

# Load environment variables
load_dotenv()

# Initialize console for pretty output
console = Console()


class TranscriptionPipeline:
    """Handles the complete transcription and diarization pipeline."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        hf_token: Optional[str] = None
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                # MPS doesn't support float16 well, use float32
                self.compute_type = "float32"
            else:
                self.device = "cpu"
                # INT8 quantization for CPU: 2x faster with <1% accuracy loss
                self.compute_type = "int8"
        else:
            self.device = device
            # Set optimal compute type for each device
            if self.device == "cpu":
                self.compute_type = "int8"  # Optimized for CPU
            elif self.device == "mps":
                self.compute_type = "float32"  # MPS requirement

        console.print(f"[cyan]Using device: {self.device}[/cyan]")

        # Optimize CPU threading for better performance
        cpu_count = os.cpu_count() or 8
        if self.device == "cpu":
            # Configure PyTorch threading (used by alignment models)
            torch.set_num_threads(cpu_count)  # Use all cores
            torch.set_num_interop_threads(2)  # Balance inter/intra-op parallelism

            # Store optimal thread count for WhisperX/CTranslate2
            # M1 Mac has 6 performance cores + 2 efficiency cores
            # Using 6 threads focuses on performance cores
            self.cpu_threads = min(cpu_count, 6)
            console.print(f"[cyan]CPU threads: PyTorch={cpu_count}, CTranslate2={self.cpu_threads}[/cyan]")
        else:
            self.cpu_threads = 4  # Default for non-CPU devices

        self.model = None
        self.diarize_model = None
        self.align_model = None
        self.align_metadata = None

    def load_models(self, language: Optional[str] = None):
        """Load WhisperX and diarization models."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Load Whisper model
            task = progress.add_task("[cyan]Loading Whisper model...", total=None)
            # Add threads parameter for CPU to enable CTranslate2 multi-threading
            model_kwargs = {
                "compute_type": self.compute_type,
                "language": language
            }
            if self.device == "cpu":
                model_kwargs["threads"] = self.cpu_threads

            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                **model_kwargs
            )
            progress.update(task, completed=True)
            console.print("[green]✓[/green] Whisper model loaded")

            # Load diarization model
            if self.hf_token:
                task = progress.add_task("[cyan]Loading diarization model...", total=None)
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                # Move to device
                if self.device != "cpu":
                    self.diarize_model.to(torch.device(self.device))
                progress.update(task, completed=True)
                console.print("[green]✓[/green] Diarization model loaded")
            else:
                console.print("[yellow]⚠[/yellow] HF_TOKEN not found. Speaker diarization will be skipped.")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_cache: bool = True
    ) -> dict:
        """
        Transcribe audio file with speaker diarization.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'pt') or None for auto-detect
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            use_cache: Whether to use cached intermediate results

        Returns:
            Dictionary containing transcription results
        """
        # Cache file paths
        cache_dir = Path(audio_path).parent / ".transcribe_cache"
        cache_dir.mkdir(exist_ok=True)
        audio_name = Path(audio_path).stem
        transcription_cache = cache_dir / f"{audio_name}_transcription.json"
        aligned_cache = cache_dir / f"{audio_name}_aligned.json"
        final_cache = cache_dir / f"{audio_name}_final.json"

        # Check if final result is cached
        if use_cache and final_cache.exists():
            console.print("[cyan]Loading complete transcription from cache...[/cyan]")
            with open(final_cache, 'r', encoding='utf-8') as f:
                result = json.load(f)
            console.print("[green]✓[/green] Complete transcription loaded from cache")
            return result

        # Load models
        self.load_models(language=language)

        # Load audio
        console.print(f"[cyan]Loading audio file: {audio_path}[/cyan]")
        audio = whisperx.load_audio(audio_path)

        # Transcribe (with caching)
        if use_cache and transcription_cache.exists():
            console.print("[cyan]Loading cached transcription...[/cyan]")
            with open(transcription_cache, 'r', encoding='utf-8') as f:
                result = json.load(f)
            detected_language = result.get("language", language)
            console.print(f"[green]✓[/green] Loaded from cache (language: {detected_language})")
        else:
            console.print("[cyan]Transcribing audio...[/cyan]")
            # Determine optimal batch size based on model size
            batch_size_map = {
                "tiny": 64,
                "base": 48,
                "small": 32,
                "medium": 24,
                "large-v2": 20,
                "large-v3": 20
            }
            batch_size = batch_size_map.get(self.model_size, 16)

            result = self.model.transcribe(audio, batch_size=batch_size)
            detected_language = result.get("language", language)

            # Save to cache
            if use_cache:
                with open(transcription_cache, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            console.print(f"[green]✓[/green] Transcription complete (language: {detected_language})")

        # Align whisper output (with caching)
        if use_cache and aligned_cache.exists():
            console.print("[cyan]Loading cached alignment...[/cyan]")
            with open(aligned_cache, 'r', encoding='utf-8') as f:
                result = json.load(f)
            console.print("[green]✓[/green] Loaded from cache")
        else:
            console.print("[cyan]Aligning timestamps...[/cyan]")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )

            # Save to cache
            if use_cache:
                with open(aligned_cache, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            console.print("[green]✓[/green] Timestamp alignment complete")

        # Diarization
        if self.diarize_model:
            console.print("[cyan]Performing speaker diarization...[/cyan]")
            diarize_params = {}
            if min_speakers:
                diarize_params["min_speakers"] = min_speakers
            if max_speakers:
                diarize_params["max_speakers"] = max_speakers

            # Create temporary WAV file for diarization (pyannote doesn't support all formats)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                # Write audio to temporary WAV file
                sf.write(temp_wav_path, audio, 16000)

            try:
                # Run diarization
                diarization = self.diarize_model(temp_wav_path, **diarize_params)

                # Convert pyannote Annotation to whisperx format (DataFrame)
                diarize_segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    diarize_segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })

                # Convert to DataFrame (required by whisperx.assign_word_speakers)
                diarize_df = pd.DataFrame(diarize_segments)

                # Assign speakers to words
                result = whisperx.assign_word_speakers(diarize_df, result)
                console.print("[green]✓[/green] Speaker diarization complete")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)

        # Save final result to cache
        if use_cache:
            with open(final_cache, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result


def validate_audio_file(audio_path: str) -> Path:
    """Validate audio file exists and has supported extension."""
    path = Path(audio_path)

    if not path.exists():
        console.print(f"[red]Error: File not found: {audio_path}[/red]")
        sys.exit(1)

    supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
    if path.suffix.lower() not in supported_extensions:
        console.print(f"[yellow]Warning: {path.suffix} may not be supported.[/yellow]")
        console.print(f"Supported formats: {', '.join(supported_extensions)}")

    return path


def save_rttm(result: dict, output_path: str):
    """Save diarization results in RTTM format."""
    with open(output_path, 'w') as f:
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "SPEAKER_00")
            start = segment.get("start", 0)
            duration = segment.get("end", 0) - start

            # RTTM format: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            f.write(f"SPEAKER audio 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

    console.print(f"[green]✓[/green] RTTM saved to: {output_path}")


def save_txt(result: dict, output_path: str):
    """Save human-readable transcript with speaker labels."""
    with open(output_path, 'w', encoding='utf-8') as f:
        current_speaker = None

        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)

            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n\n")
                timestamp = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
                f.write(f"{speaker} {timestamp}:\n")
                current_speaker = speaker

            f.write(f"{text}\n")

    console.print(f"[green]✓[/green] Transcript saved to: {output_path}")


def save_json(result: dict, output_path: str):
    """Save complete results as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓[/green] JSON saved to: {output_path}")


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option(
    '--language', '-l',
    type=click.Choice(['en', 'pt', 'auto'], case_sensitive=False),
    default='auto',
    help='Language of the audio (en=English, pt=Portuguese, auto=detect)'
)
@click.option(
    '--model', '-m',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']),
    default='large-v3',
    help='Whisper model size (larger = more accurate but slower)'
)
@click.option(
    '--min-speakers',
    type=int,
    default=None,
    help='Minimum number of speakers (optional)'
)
@click.option(
    '--max-speakers',
    type=int,
    default=None,
    help='Maximum number of speakers (optional)'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default=None,
    help='Output directory (default: same as input file)'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['rttm', 'txt', 'json', 'all'], case_sensitive=False),
    multiple=True,
    default=['all'],
    help='Output format(s)'
)
@click.option(
    '--device',
    type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
    default='auto',
    help='Device to use for computation'
)
def main(
    audio_file: str,
    language: str,
    model: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    output: Optional[str],
    format: tuple,
    device: str
):
    """
    Transcribe audio files with speaker diarization.

    Example usage:

        # Basic transcription (auto-detect language)
        python transcribe.py meeting.mp3

        # Specify language and model
        python transcribe.py meeting.mp3 --language pt --model large-v3

        # Specify number of speakers
        python transcribe.py meeting.mp3 --min-speakers 2 --max-speakers 4

        # Custom output location
        python transcribe.py meeting.mp3 --output ./transcripts/

        # Specific output formats
        python transcribe.py meeting.mp3 --format rttm --format txt
    """
    # Banner
    console.print("\n[bold cyan]═══ Meeting Transcriber ═══[/bold cyan]\n")

    # Validate audio file
    audio_path = validate_audio_file(audio_file)

    # Set output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = audio_path.parent

    # Output base name
    output_base = output_dir / audio_path.stem

    # Convert language
    lang = None if language == 'auto' else language

    try:
        # Initialize pipeline
        pipeline = TranscriptionPipeline(
            model_size=model,
            device=device
        )

        # Transcribe
        start_time = datetime.now()
        result = pipeline.transcribe(
            str(audio_path),
            language=lang,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        console.print(f"\n[green]✓[/green] Transcription completed in {elapsed:.1f}s")

        # Determine output formats
        formats = set(format)
        if 'all' in formats:
            formats = {'rttm', 'txt', 'json'}

        # Save outputs
        console.print("\n[cyan]Saving outputs...[/cyan]")

        if 'rttm' in formats:
            save_rttm(result, f"{output_base}.rttm")

        if 'txt' in formats:
            save_txt(result, f"{output_base}.txt")

        if 'json' in formats:
            save_json(result, f"{output_base}.json")

        console.print("\n[bold green]✓ Done![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
