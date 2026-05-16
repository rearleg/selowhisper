"""Headless CLI test for SeloWhisper-ko-disfluency.

Usage:
    python examples/transcribe.py path/to/audio.wav
    python examples/transcribe.py path/to/audio.wav --no-meta-strip

The first run downloads the model (~3 GB) from
https://huggingface.co/rearleg/SeloWhisper-ko-disfluency.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

HF_MODEL_ID = "rearleg/SeloWhisper-ko-disfluency"
DISFLUENCY_TOKENS: tuple[str, ...] = (
    "<ah>", "<uh>", "<um>", "<gue>", "<jeo>",
    "<mwo>", "<mak>", "<repeat>", "<laugh>", "<other>",
)
WHISPER_META_RE = re.compile(r"<\|[^|]+\|>")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    return waveform, sr


def strip_whisper_meta(text: str) -> str:
    return WHISPER_META_RE.sub("", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file (wav/flac/ogg/mp3).")
    parser.add_argument(
        "--no-meta-strip",
        action="store_true",
        help="Keep Whisper meta tokens (<|ko|>, <|transcribe|>, …) in the output.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    device = pick_device() if args.device == "auto" else args.device
    print(f"[info] device={device}, model={HF_MODEL_ID}", file=sys.stderr)

    processor = WhisperProcessor.from_pretrained(HF_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(HF_MODEL_ID).eval()
    model.config.forced_decoder_ids = None
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None
        model.generation_config.begin_suppress_tokens = None
    model.to(device)

    waveform, sr = load_waveform(args.audio)
    inputs = processor(
        waveform.squeeze().cpu().numpy(),
        sampling_rate=sr,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            inputs["input_features"],
            max_length=448,
            num_beams=1,
            do_sample=False,
        )
    raw = processor.batch_decode(generated, skip_special_tokens=False)[0]

    print(raw if args.no_meta_strip else strip_whisper_meta(raw))

    counts = {tok: raw.count(tok) for tok in DISFLUENCY_TOKENS if raw.count(tok) > 0}
    if counts:
        print(
            "\n[disfluencies] " + ", ".join(f"{k}={v}" for k, v in counts.items()),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
