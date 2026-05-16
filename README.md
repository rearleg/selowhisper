# SeloWhisper · Demo

Run [**SeloWhisper-ko-disfluency**](https://huggingface.co/rearleg/SeloWhisper-ko-disfluency) — a Korean ASR model that transcribes speech with **inline disfluency detection** (음 / 어 / 그 / 저 / 막 / 뭐 / 아, repetitions, laughter) — locally with a Streamlit UI or a one-shot CLI.

Fine-tuned from `openai/whisper-large-v3-turbo`. The first run downloads model weights (~3 GB) from the Hugging Face Hub; afterwards everything is offline.

> 모델 가중치와 카드: <https://huggingface.co/rearleg/SeloWhisper-ko-disfluency>
> 학습 방법론 / 데이터 / 평가 디테일은 논문 출간 전이라 공개하지 않습니다.

---

## Features

- 🎙️ Streamlit web UI — drag & drop audio, get transcription + disfluency highlights
- ⚡ One-shot CLI for headless / scripted runs
- 🏷️ Inline disfluency tags via 10 dedicated special tokens
- 📊 Per-token detection counts
- 🚀 CPU · CUDA · Apple Silicon (MPS) — auto-detected
- 🔐 Local cache — model downloads once, then runs fully offline

---

## Quick Start

```bash
git clone https://github.com/rearleg/selowhisper.git
cd selowhisper

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

streamlit run app.py
```

Browser opens at `http://localhost:8501`. Use the **"🎧 샘플 오디오 사용"** button to test the bundled sample (`samples/demo.wav`), or upload your own clip.

> First launch: pulls ~3 GB of model weights into `~/.cache/huggingface/hub/`. Re-runs are instant.

---

## CLI Usage

```bash
python examples/transcribe.py samples/demo.wav
```

Output (transcription to stdout, diagnostics to stderr):

```
그<gue> 그래서 <um> 제가 그 학교 정문에서 만나기로 했는데 <laugh>
[disfluencies] <gue>=1, <um>=1, <laugh>=1
```

Flags:

| Flag | Meaning |
|---|---|
| `--no-meta-strip` | Keep Whisper meta tokens (`<\|ko\|>`, `<\|transcribe\|>`, …) in stdout |
| `--device {auto,cpu,cuda,mps}` | Force a specific compute device (default: auto-detect) |

---

## Disfluency Tokens

| Token | Korean cue | Meaning |
|---|---|---|
| `<ah>`  | 아 | filler "ah" |
| `<uh>`  | 어 | filler "uh" |
| `<um>`  | 음 | filler "um" |
| `<gue>` | 그 | filler "geu" |
| `<jeo>` | 저 | filler "jeo" |
| `<mwo>` | 뭐 | filler "mwo" |
| `<mak>` | 막 | filler "mak" |
| `<repeat>` | — | repeated word / syllable |
| `<laugh>`  | — | laughter |
| `<other>`  | — | other disfluency |

When decoding, pass `skip_special_tokens=False` to keep these visible. Whisper meta tokens (`<|ko|>`, `<|transcribe|>`, …) can be stripped with a regex — `app.py` and `examples/transcribe.py` do this automatically.

---

## Python API (without the demo wrappers)

```python
import torch, soundfile as sf, numpy as np, torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "rearleg/SeloWhisper-ko-disfluency"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()

audio, sr = sf.read("samples/demo.wav", dtype="float32", always_2d=True)
audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
wav = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

inputs = processor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
with torch.inference_mode():
    ids = model.generate(inputs["input_features"], max_length=448)
print(processor.batch_decode(ids, skip_special_tokens=False)[0])
```

---

## Requirements

- Python **3.10+**
- ~5 GB free disk (model cache)
- macOS / Linux / Windows
- Optional: NVIDIA GPU (CUDA) or Apple Silicon (MPS) for faster inference

The pinned dependency list is in `requirements.txt` — `transformers`, `torch`, `torchaudio`, `streamlit`, `huggingface_hub`, `soundfile`, etc.

---

## Project Layout

```
selowhisper/
├── app.py                  # Streamlit demo (entry point)
├── examples/
│   └── transcribe.py       # Headless CLI
├── samples/
│   └── demo.wav        # Bundled audio sample for quick testing
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

---

## Troubleshooting

**`ImportError: TorchCodec is required for load_with_torchcodec`**
Use `app.py` / `examples/transcribe.py` as-is — they bypass torchaudio decoders and use `soundfile`. If you wrote your own loader, switch to `soundfile.read()` or install `torchcodec` + `ffmpeg`.

**Model not downloading / `401 Unauthorized`**
The HF repo is public, so anonymous access works. If you have a stale HF token cached, log out: `hf auth logout`.

**Slow first run**
~3 GB download. Set `HF_HOME=/path/to/big/disk` before running to point the cache elsewhere.

**MPS runs out of memory**
Force CPU: `python examples/transcribe.py … --device cpu`, or in `app.py` set `device = "cpu"` in `pick_device()`.

---

## License

This repo (demo code) is licensed under **MIT** — see [LICENSE](LICENSE).

The released model weights are also under MIT — see the [Hugging Face model card](https://huggingface.co/rearleg/SeloWhisper-ko-disfluency).

---

## Citation

```bibtex
@misc{cheon2025selowhisper,
  title        = {SeloWhisper-ko-disfluency: Korean ASR with Inline Disfluency Detection},
  author       = {Cheon, Changhyun},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/rearleg/SeloWhisper-ko-disfluency}}
}
```

---

## Acknowledgments

- Built on [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo).
- Inference UI powered by [Streamlit](https://streamlit.io/).
