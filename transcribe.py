#!/usr/bin/env python3
"""
Meeting Transcriber CLI
Transcribes audio files with speaker diarization.

Backends:
  - whisperx: CPU with INT8 quantization (CTranslate2, no MPS support)
  - mlx: Apple GPU via MLX (~7x faster than CPU, native word timestamps)
  - transformers: MPS GPU via HuggingFace Transformers pipeline
"""

import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import soundfile as sf
import torch
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Initialize console for pretty output
console = Console()

# Model ID mappings per backend
HF_MODEL_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}

MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

# CTranslate2 model overrides for WhisperX (only non-standard mappings)
WHISPERX_MODEL_MAP = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


class TranscriptionPipeline:
    """Handles the complete transcription and diarization pipeline."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        hf_token: Optional[str] = None,
        backend: str = "whisperx",
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.backend = backend

        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.compute_type = "float16"
            elif torch.backends.mps.is_available():
                if self.backend == "transformers":
                    # HF Transformers supports MPS natively
                    self.device = "mps"
                    self.compute_type = "float32"  # MPS requires float32 for Whisper
                else:
                    # WhisperX/CTranslate2 doesn't support MPS; MLX handles its own device
                    self.device = "cpu"
                    self.compute_type = "int8"
            else:
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = device
            if self.device == "cpu":
                self.compute_type = "int8"
            elif self.device == "mps":
                self.compute_type = "float32" if self.backend == "transformers" else "float32"

        # Hybrid device strategy: Use MPS for diarization even when transcription uses CPU
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.diarize_device = "mps"
            if self.backend == "mlx":
                console.print("[cyan]Using MLX (Apple GPU) for transcription, MPS for diarization[/cyan]")
            else:
                console.print("[cyan]Using hybrid devices: Transcription=CPU (INT8), Diarization=MPS (GPU)[/cyan]")
        elif self.device == "mps":
            self.diarize_device = "mps"
            console.print("[cyan]Using MPS (GPU) for all stages[/cyan]")
        else:
            self.diarize_device = self.device
            console.print(f"[cyan]Using device: {self.device}[/cyan]")

        # Optimize CPU threading for better performance
        cpu_count = os.cpu_count() or 8
        if self.device == "cpu":
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(2)
            # M1 Mac has 6 performance cores + 2 efficiency cores
            self.cpu_threads = min(cpu_count, 6)
            if self.backend == "whisperx":
                console.print(f"[cyan]CPU threads: PyTorch={cpu_count}, CTranslate2={self.cpu_threads}[/cyan]")
        else:
            self.cpu_threads = 4

        console.print(f"[cyan]Backend: {self.backend} | Model: {self.model_size}[/cyan]")

        self.model = None
        self.hf_pipe = None
        self.diarize_model = None
        self.align_model = None
        self.align_metadata = None

    def load_models(self, language: Optional[str] = None):
        """Load transcription and diarization models."""
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            if self.backend == "mlx":
                # MLX models are loaded lazily by mlx_whisper.transcribe()
                task = progress.add_task("[cyan]Checking MLX-Whisper...", total=None)
                try:
                    import mlx_whisper  # noqa: F401
                except ImportError:
                    raise RuntimeError(
                        "mlx-whisper is required for the 'mlx' backend. Install it with: pip install mlx-whisper"
                    )
                model_id = MLX_MODEL_MAP.get(self.model_size, f"mlx-community/whisper-{self.model_size}")
                progress.update(task, completed=True)
                console.print(f"[green]✓[/green] MLX-Whisper ready (model: {model_id})")

            elif self.backend == "transformers":
                task = progress.add_task("[cyan]Loading Whisper model (HF Transformers)...", total=None)
                try:
                    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                    from transformers import pipeline as hf_pipeline
                except ImportError:
                    raise RuntimeError(
                        "transformers is required for the 'transformers' backend. "
                        "Install it with: pip install transformers"
                    )

                model_id = HF_MODEL_MAP.get(self.model_size, f"openai/whisper-{self.model_size}")
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                model.to(self.device)

                processor = AutoProcessor.from_pretrained(model_id)

                batch_size_map = {
                    "tiny": 64,
                    "base": 48,
                    "small": 40,
                    "medium": 32,
                    "large-v2": 32,
                    "large-v3": 32,
                    "large-v3-turbo": 32,
                }
                batch_size = batch_size_map.get(self.model_size, 16)

                self.hf_pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=self.device,
                    chunk_length_s=30,
                    batch_size=batch_size,
                )

                progress.update(task, completed=True)
                console.print(f"[green]✓[/green] Whisper model loaded ({model_id} on {self.device.upper()})")

            else:
                # WhisperX backend (original)
                task = progress.add_task("[cyan]Loading Whisper model...", total=None)
                model_kwargs = {
                    "compute_type": self.compute_type,
                    "language": language,
                }
                if self.device == "cpu":
                    model_kwargs["threads"] = self.cpu_threads

                model_name = WHISPERX_MODEL_MAP.get(self.model_size, self.model_size)

                self.model = whisperx.load_model(model_name, self.device, **model_kwargs)
                progress.update(task, completed=True)
                console.print("[green]✓[/green] Whisper model loaded")

            # Load diarization model (same for all backends)
            if self.hf_token:
                task = progress.add_task("[cyan]Loading diarization model...", total=None)
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
                )
                if self.diarize_device != "cpu":
                    self.diarize_model.to(torch.device(self.diarize_device))
                    console.print(f"[green]✓[/green] Diarization model loaded on {self.diarize_device.upper()}")
                else:
                    console.print("[green]✓[/green] Diarization model loaded on CPU")
                progress.update(task, completed=True)
            else:
                console.print("[yellow]⚠[/yellow] HF_TOKEN not found. Speaker diarization will be skipped.")

    def _detect_language_hf(self, audio_path: str) -> str:
        """Detect language using the Whisper model (transformers backend)."""
        audio = whisperx.load_audio(audio_path)
        sample = audio[: 16000 * 30]  # First 30 seconds
        input_features = self.hf_pipe.feature_extractor(
            sample, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.hf_pipe.device, dtype=self.hf_pipe.torch_dtype)

        predicted_ids = self.hf_pipe.model.generate(input_features, max_new_tokens=1)
        token_str = self.hf_pipe.tokenizer.decode(predicted_ids[0], skip_special_tokens=False)

        # Parse language code from tokens like "<|startoftranscript|><|pt|>"
        lang_match = re.search(r"<\|([a-z]{2})\|>", token_str)
        if lang_match:
            detected = lang_match.group(1)
            console.print(f"[green]✓[/green] Detected language: {detected}")
            return detected

        console.print("[yellow]⚠[/yellow] Could not detect language, defaulting to 'en'")
        return "en"

    def _transcribe_with_mlx(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Transcribe using MLX-Whisper (Apple GPU accelerated, native word timestamps)."""
        import mlx_whisper

        model_id = MLX_MODEL_MAP.get(self.model_size, f"mlx-community/whisper-{self.model_size}")

        kwargs = {
            "path_or_hf_repo": model_id,
            "word_timestamps": True,
            "verbose": False,
        }
        if language:
            kwargs["language"] = language

        result = mlx_whisper.transcribe(audio_path, **kwargs)
        # Returns {"text": "...", "segments": [...], "language": "pt"}
        # Segments include "words" with start/end/probability — compatible with whisperx
        return result

    def _transcribe_with_transformers(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Transcribe using HuggingFace Transformers pipeline (MPS GPU)."""
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language

        hf_result = self.hf_pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        # Convert HF output to WhisperX-compatible segment format
        # HF returns: {"text": "...", "chunks": [{"text": "...", "timestamp": (start, end)}, ...]}
        # WhisperX expects: {"segments": [{"text": "...", "start": float, "end": float}, ...]}
        segments = []
        for chunk in hf_result.get("chunks", []):
            ts = chunk.get("timestamp")
            if ts is None:
                continue
            start, end = ts
            if start is None or end is None:
                continue
            segments.append(
                {
                    "text": chunk["text"],
                    "start": float(start),
                    "end": float(end),
                }
            )

        return {
            "segments": segments,
            "language": language or "auto",
        }

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_cache: bool = True,
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
        # Cache file paths (include backend + model in cache key)
        cache_dir = Path(audio_path).parent / ".transcribe_cache"
        cache_dir.mkdir(exist_ok=True)
        audio_name = Path(audio_path).stem
        cache_prefix = f"{audio_name}_{self.backend}_{self.model_size}"
        transcription_cache = cache_dir / f"{cache_prefix}_transcription.json"
        aligned_cache = cache_dir / f"{cache_prefix}_aligned.json"
        final_cache = cache_dir / f"{cache_prefix}_final.json"

        # Check if final result is cached
        if use_cache and final_cache.exists():
            console.print("[cyan]Loading complete transcription from cache...[/cyan]")
            with open(final_cache, "r", encoding="utf-8") as f:
                result = json.load(f)
            console.print("[green]✓[/green] Complete transcription loaded from cache")
            return result

        # Load models
        self.load_models(language=language)

        # For transformers backend, detect language if not specified
        if language is None and self.backend == "transformers":
            language = self._detect_language_hf(audio_path)

        # Load audio (needed for alignment and diarization)
        console.print(f"[cyan]Loading audio file: {audio_path}[/cyan]")
        audio = whisperx.load_audio(audio_path)

        # Transcribe (with caching)
        if use_cache and transcription_cache.exists():
            console.print("[cyan]Loading cached transcription...[/cyan]")
            with open(transcription_cache, "r", encoding="utf-8") as f:
                result = json.load(f)
            detected_language = result.get("language", language)
            console.print(f"[green]✓[/green] Loaded from cache (language: {detected_language})")
        else:
            console.print(f"[cyan]Transcribing audio ({self.backend} backend)...[/cyan]")

            if self.backend == "mlx":
                result = self._transcribe_with_mlx(audio_path, language)
            elif self.backend == "transformers":
                result = self._transcribe_with_transformers(audio_path, language)
            else:
                # WhisperX backend (original)
                batch_size_map = {
                    "tiny": 64,
                    "base": 48,
                    "small": 40,
                    "medium": 32,
                    "large-v2": 32,
                    "large-v3": 32,
                    "large-v3-turbo": 32,
                }
                batch_size = batch_size_map.get(self.model_size, 16)
                result = self.model.transcribe(audio, batch_size=batch_size)

            detected_language = result.get("language", language)

            # Save to cache
            if use_cache:
                with open(transcription_cache, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            console.print(f"[green]✓[/green] Transcription complete (language: {detected_language})")

        # Align whisper output — skip for mlx (native word timestamps via cross-attention)
        if self.backend != "mlx":
            if use_cache and aligned_cache.exists():
                console.print("[cyan]Loading cached alignment...[/cyan]")
                with open(aligned_cache, "r", encoding="utf-8") as f:
                    result = json.load(f)
                console.print("[green]✓[/green] Loaded from cache")
            else:
                console.print("[cyan]Aligning timestamps...[/cyan]")
                detected_language = result.get("language", language)
                # Use CPU for alignment when on MPS (more reliable across torchaudio models)
                align_device = "cpu" if self.device == "mps" else self.device

                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=align_device,
                )
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    align_device,
                    return_char_alignments=False,
                )

                # Save to cache
                if use_cache:
                    with open(aligned_cache, "w", encoding="utf-8") as f:
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
                sf.write(temp_wav_path, audio, 16000)

            try:
                # Run diarization
                diarization = self.diarize_model(temp_wav_path, **diarize_params)

                # Convert pyannote Annotation to whisperx format (DataFrame)
                diarize_segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    diarize_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

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
            with open(final_cache, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result


def validate_audio_file(audio_path: str) -> Path:
    """Validate audio file exists and has supported extension."""
    path = Path(audio_path)

    if not path.exists():
        console.print(f"[red]Error: File not found: {audio_path}[/red]")
        sys.exit(1)

    supported_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
    if path.suffix.lower() not in supported_extensions:
        console.print(f"[yellow]Warning: {path.suffix} may not be supported.[/yellow]")
        console.print(f"Supported formats: {', '.join(supported_extensions)}")

    return path


def save_rttm(result: dict, output_path: str):
    """Save diarization results in RTTM format."""
    with open(output_path, "w") as f:
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "SPEAKER_00")
            start = segment.get("start", 0)
            duration = segment.get("end", 0) - start

            # RTTM format: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            f.write(f"SPEAKER audio 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")

    console.print(f"[green]✓[/green] RTTM saved to: {output_path}")


def save_txt(result: dict, output_path: str):
    """Save human-readable transcript with speaker labels."""
    with open(output_path, "w", encoding="utf-8") as f:
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓[/green] JSON saved to: {output_path}")


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "pt", "auto"], case_sensitive=False),
    default="auto",
    help="Language of the audio (en=English, pt=Portuguese, auto=detect)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]),
    default="large-v3",
    help="Whisper model size (larger = more accurate but slower)",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["whisperx", "mlx", "transformers"], case_sensitive=False),
    default="whisperx",
    help="Transcription backend: whisperx (CPU), mlx (Apple GPU, fastest), transformers (MPS GPU)",
)
@click.option("--min-speakers", type=int, default=None, help="Minimum number of speakers (optional)")
@click.option("--max-speakers", type=int, default=None, help="Maximum number of speakers (optional)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory (default: same as input file)")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["rttm", "txt", "json", "all"], case_sensitive=False),
    multiple=True,
    default=["all"],
    help="Output format(s)",
)
@click.option(
    "--device", type=click.Choice(["auto", "cuda", "mps", "cpu"]), default="auto", help="Device to use for computation"
)
def main(
    audio_file: str,
    language: str,
    model: str,
    backend: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    output: Optional[str],
    format: tuple,
    device: str,
):
    """
    Transcribe audio files with speaker diarization.

    Example usage:

        # Basic transcription (auto-detect language)
        python transcribe.py meeting.mp3

        # GPU-accelerated with MLX (fastest on Apple Silicon)
        python transcribe.py meeting.mp3 -l pt -b mlx

        # Fast turbo model on Apple GPU
        python transcribe.py meeting.mp3 -l pt -m large-v3-turbo -b mlx

        # HF Transformers backend (MPS GPU fallback)
        python transcribe.py meeting.mp3 -l pt -b transformers

        # Specify number of speakers
        python transcribe.py meeting.mp3 --min-speakers 2 --max-speakers 4

        # Custom output location
        python transcribe.py meeting.mp3 --output ./transcripts/
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
    lang = None if language == "auto" else language

    try:
        # Initialize pipeline
        pipeline = TranscriptionPipeline(
            model_size=model,
            device=device,
            backend=backend,
        )

        # Transcribe
        start_time = datetime.now()
        result = pipeline.transcribe(
            str(audio_path), language=lang, min_speakers=min_speakers, max_speakers=max_speakers
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        console.print(f"\n[green]✓[/green] Transcription completed in {elapsed:.1f}s")

        # Determine output formats
        formats = set(format)
        if "all" in formats:
            formats = {"rttm", "txt", "json"}

        # Save outputs
        console.print("\n[cyan]Saving outputs...[/cyan]")

        if "rttm" in formats:
            save_rttm(result, f"{output_base}.rttm")

        if "txt" in formats:
            save_txt(result, f"{output_base}.txt")

        if "json" in formats:
            save_json(result, f"{output_base}.json")

        console.print("\n[bold green]✓ Done![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
