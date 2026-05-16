"""SeloWhisper-ko-disfluency — Streamlit demo (downloads model from HF Hub).

Run:
    pip install -r requirements.txt
    streamlit run app.py

On first launch this downloads ~3 GB of model weights from
https://huggingface.co/rearleg/SeloWhisper-ko-disfluency to the local
Hugging Face cache (~/.cache/huggingface/hub/). Subsequent runs reuse it.
"""

from __future__ import annotations

import os

# Silence transformers' optional-dependency advisory (we don't use torchvision).
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import io
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

HF_MODEL_ID = "rearleg/SeloWhisper-ko-disfluency"
HERE = Path(__file__).parent
SAMPLE_AUDIO = HERE / "samples" / "명지대1길9.wav"

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


@st.cache_resource(show_spinner="모델 다운로드 / 로드 중... (첫 실행 시 ~3 GB)")
def load_model() -> tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
    processor = WhisperProcessor.from_pretrained(HF_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(HF_MODEL_ID).eval()

    # Defensive: clear any Whisper-base suppression even though the hub model
    # already ships a patched generation_config.
    model.config.forced_decoder_ids = None
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None
        model.generation_config.begin_suppress_tokens = None

    device = pick_device()
    model.to(device)
    return processor, model, device


def load_waveform(
    file_bytes: bytes | None, path: Path | None
) -> tuple[torch.Tensor, int]:
    """Decode audio to a mono 16 kHz [1, frames] float32 tensor via soundfile."""
    source: str | io.BytesIO
    if path is not None:
        source = str(path)
    else:
        assert file_bytes is not None
        source = io.BytesIO(file_bytes)

    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    return waveform, sr


@torch.inference_mode()
def transcribe(
    audio: torch.Tensor,
    sr: int,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
) -> str:
    inputs = processor(
        audio.squeeze().cpu().numpy(),
        sampling_rate=sr,
        return_tensors="pt",
    ).to(device)
    generated = model.generate(
        inputs["input_features"],
        max_length=448,
        num_beams=1,
        do_sample=False,
    )
    return processor.batch_decode(generated, skip_special_tokens=False)[0]


def strip_whisper_meta(text: str) -> str:
    return WHISPER_META_RE.sub("", text).strip()


def count_disfluencies(text: str) -> dict[str, int]:
    return {tok: text.count(tok) for tok in DISFLUENCY_TOKENS if text.count(tok) > 0}


def render_highlighted(text: str) -> str:
    escaped = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
    for tok in DISFLUENCY_TOKENS:
        escaped_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
        highlighted = (
            f'<span style="background:#ffe680;border-radius:4px;'
            f'padding:1px 6px;margin:0 2px;font-weight:600;'
            f'font-family:monospace;">{escaped_tok}</span>'
        )
        escaped = escaped.replace(escaped_tok, highlighted)
    return escaped


def main() -> None:
    st.set_page_config(
        page_title="SeloWhisper-ko-disfluency",
        page_icon="🎙️",
        layout="centered",
    )
    st.title("SeloWhisper-ko-disfluency")
    st.caption(
        "Whisper Large v3 Turbo 기반 한국어 비유창성 탐지 모델 데모 · "
        f"[모델 카드](https://huggingface.co/{HF_MODEL_ID})"
    )

    processor, model, device = load_model()
    st.success(f"모델 로드 완료 — device: `{device}`")

    st.divider()
    st.subheader("1. 입력 오디오")

    col1, col2 = st.columns([3, 2])
    audio_bytes: bytes | None = None
    audio_path: Path | None = None

    with col1:
        uploaded = st.file_uploader(
            "WAV / FLAC / OGG / MP3 업로드",
            type=["wav", "flac", "ogg", "mp3"],
        )
        if uploaded is not None:
            audio_bytes = uploaded.read()
            st.session_state["use_sample"] = False

    with col2:
        st.write("")
        st.write("")
        if SAMPLE_AUDIO.exists():
            if st.button("🎧 샘플 오디오 사용", use_container_width=True):
                st.session_state["use_sample"] = True
        else:
            st.caption(f"샘플 없음: `{SAMPLE_AUDIO.name}`")

    if audio_bytes is None and st.session_state.get("use_sample"):
        if SAMPLE_AUDIO.exists():
            audio_path = SAMPLE_AUDIO

    if audio_bytes is None and audio_path is None:
        st.info("오디오를 업로드하거나 샘플 버튼을 눌러주세요.")
        return

    if audio_path is not None:
        st.audio(str(audio_path))
        st.caption(f"샘플 파일: `{audio_path.name}`")
    else:
        st.audio(audio_bytes)

    st.divider()
    st.subheader("2. 전사 결과")

    with st.spinner("전사 중..."):
        waveform, sr = load_waveform(audio_bytes, audio_path)
        raw = transcribe(waveform, sr, processor, model, device)

    cleaned = strip_whisper_meta(raw)

    st.markdown(
        f'<div style="font-size:1.15em;line-height:1.8;'
        f'padding:14px 18px;background:#f6f7f9;border-radius:8px;">'
        f"{render_highlighted(cleaned)}"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("3. 검출된 비유창성 토큰")
    counts = count_disfluencies(cleaned)
    if counts:
        st.metric("총 비유창성 토큰 수", sum(counts.values()))
        st.table({"token": list(counts.keys()), "count": list(counts.values())})
    else:
        st.info("비유창성 토큰이 검출되지 않았습니다.")

    with st.expander("Raw output (special tokens 포함)"):
        st.code(raw or "(empty)", language="text")
    with st.expander("정제 후 텍스트 (Whisper meta 토큰 제거)"):
        st.code(cleaned or "(empty)", language="text")


if __name__ == "__main__":
    main()
