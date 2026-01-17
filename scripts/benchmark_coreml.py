#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from unicodedata import normalize

import numpy as np
import coremltools as ct
from coremltools.models import MLModel
import onnx
from onnx import numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import convert_onnx_coreml as conv  # noqa: E402


MODEL_DIR = "models/supertonic-2/coreml"
ONNX_DIR = "models/supertonic-2/onnx"
VOICE_DIR = "models/supertonic-2/voice_styles"


def preprocess_text(text: str, lang: str) -> str:
    text = normalize("NFKD", text)
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\U0001f1e6-\U0001f1ff]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    replacements = {
        "–": "-",
        "‑": "-",
        "—": "-",
        "_": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "´": "'",
        "`": "'",
        "[": " ",
        "]": " ",
        "|": " ",
        "/": " ",
        "#": " ",
        "→": " ",
        "←": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    expr_replacements = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
    for k, v in expr_replacements.items():
        text = text.replace(k, v)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r" ;", ";", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" '", "'", text)
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    while "``" in text:
        text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>" + text + f"</{lang}>"


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pattern = (
            r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
            r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
            r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
            r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        )
        sentences = re.split(pattern, paragraph)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
    return chunks


def length_to_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    max_len = int(latent_lengths.max())
    return length_to_mask(latent_lengths, max_len)


def build_text_inputs(
    text_list: list[str], lang_list: list[str], max_text_len: int
) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(ONNX_DIR, "unicode_indexer.json"), "r", encoding="utf-8") as f:
        indexer = json.load(f)
    processed = [preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)]
    lengths = np.array([len(t) for t in processed], dtype=np.int64)
    if lengths.max() > max_text_len:
        raise ValueError(f"Text length {lengths.max()} exceeds max_text_len={max_text_len}")
    text_ids = np.zeros((len(processed), max_text_len), dtype=np.int32)
    for i, text in enumerate(processed):
        ids = [indexer[ord(ch)] for ch in text]
        if any(j < 0 for j in ids):
            raise ValueError("Text contains unsupported characters.")
        text_ids[i, : len(ids)] = np.array(ids, dtype=np.int32)
    text_mask = length_to_mask(lengths, max_text_len)
    return text_ids, text_mask


def load_embedding_weight(onnx_path: str) -> np.ndarray:
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        if "char_embedder.weight" in init.name and len(init.dims) == 2:
            return numpy_helper.to_array(init).astype(np.float32)
    raise RuntimeError(f"char_embedder.weight not found in {onnx_path}")


def build_text_embed(text_ids: np.ndarray, onnx_path: str) -> np.ndarray:
    weight = load_embedding_weight(onnx_path)
    embed = weight[text_ids]  # (B, T, C)
    return np.transpose(embed, (0, 2, 1)).astype(np.float32)


def load_voice(voice: str) -> tuple[np.ndarray, np.ndarray]:
    voice_path = os.path.join(VOICE_DIR, f"{voice}.json")
    data = json.loads(Path(voice_path).read_text())
    ttl_dims = data["style_ttl"]["dims"]
    dp_dims = data["style_dp"]["dims"]
    ttl_data = np.array(data["style_ttl"]["data"], dtype=np.float32).flatten()
    dp_data = np.array(data["style_dp"]["data"], dtype=np.float32).flatten()
    style_ttl = ttl_data.reshape(ttl_dims[0], ttl_dims[1], ttl_dims[2])
    style_dp = dp_data.reshape(dp_dims[0], dp_dims[1], dp_dims[2])
    return style_ttl, style_dp


def get_cfgs(cfg_dir: str) -> dict:
    return conv.load_cfgs(cfg_dir)


def get_compute_units(name: str):
    mapping = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu+gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }
    return mapping[name]


def get_rss_mb() -> float:
    try:
        out = subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        kb = float(out.strip())
        return kb / 1024.0
    except Exception:
        return 0.0


def get_peak_rss_mb() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio).astype(np.float32).reshape(-1)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1e-6:
        audio = audio * (0.95 / max_abs)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Supertonic-2 CoreML ML Program stack.")
    parser.add_argument("--text", default="Hello world. This is a CoreML benchmark.")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--voice", default="F1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-text-len", type=int, default=300)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--silence", type=float, default=0.3)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--format", choices=("mlprogram", "nn"), default="mlprogram")
    parser.add_argument("--compute", choices=("cpu", "cpu+gpu", "all"), default="cpu")
    parser.add_argument("--out-wav", default="outputs/tts_sample.wav")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    os.makedirs(".coremltmp", exist_ok=True)
    os.environ["TMPDIR"] = os.path.abspath(".coremltmp")

    if args.format == "mlprogram":
        dp_path = os.path.join(args.model_dir, "duration_predictor_mlprogram.mlpackage")
        te_path = os.path.join(args.model_dir, "text_encoder_mlprogram.mlpackage")
        ve_path = os.path.join(args.model_dir, "vector_estimator_mlprogram.mlpackage")
        voc_path = os.path.join(args.model_dir, "vocoder_mlprogram.mlpackage")
    else:
        dp_path = os.path.join(args.model_dir, "duration_predictor_fixed.mlmodel")
        te_path = os.path.join(args.model_dir, "text_encoder_fixed.mlmodel")
        ve_path = os.path.join(args.model_dir, "vector_estimator_fixed.mlmodel")
        voc_path = os.path.join(args.model_dir, "vocoder_fixed.mlmodel")

    for p in (dp_path, te_path, ve_path, voc_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    compute_units = get_compute_units(args.compute)

    dp_model = MLModel(dp_path, compute_units=compute_units)
    te_model = MLModel(te_path, compute_units=compute_units)
    ve_model = MLModel(ve_path, compute_units=compute_units)
    voc_model = MLModel(voc_path, compute_units=compute_units)

    cfgs = get_cfgs(ONNX_DIR)
    sample_rate = int(cfgs["ae"]["sample_rate"])
    base_chunk_size = int(cfgs["ae"]["base_chunk_size"])
    chunk_compress_factor = int(cfgs["ttl"]["chunk_compress_factor"])
    shapes = conv.build_input_shapes(cfgs, args.max_text_len, args.max_seconds)
    latent_dim = int(shapes["vector_estimator.onnx"]["noisy_latent"][1])
    latent_len_max = int(shapes["vector_estimator.onnx"]["noisy_latent"][2])

    style_ttl, style_dp = load_voice(args.voice)

    def sample_noisy_latent(duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = int(np.max(duration) * sample_rate)
        wav_lengths = (duration * sample_rate).astype(np.int64)
        chunk_size = base_chunk_size * chunk_compress_factor
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        if latent_len > latent_len_max:
            latent_len = latent_len_max
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len_max).astype(np.float32)
        latent_mask = get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor)
        if latent_mask.shape[-1] > latent_len_max:
            latent_mask = latent_mask[:, :, :latent_len_max]
        elif latent_mask.shape[-1] < latent_len_max:
            pad = latent_len_max - latent_mask.shape[-1]
            latent_mask = np.pad(latent_mask, ((0, 0), (0, 0), (0, pad)), mode="constant")
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def run_once() -> dict:
        timings = {"duration_predictor": 0.0, "text_encoder": 0.0, "vector_estimator": 0.0, "vocoder": 0.0}
        wav_parts = []
        dur_parts = []

        max_len = 120 if args.lang == "ko" else args.max_text_len
        for chunk in chunk_text(args.text, max_len=max_len):
            text_ids, text_mask = build_text_inputs([chunk], [args.lang], args.max_text_len)
            text_embed_dp = build_text_embed(text_ids, os.path.join(ONNX_DIR, "duration_predictor.onnx"))
            text_embed_te = build_text_embed(text_ids, os.path.join(ONNX_DIR, "text_encoder.onnx"))

            t0 = time.perf_counter()
            dp_out = dp_model.predict(
                {"style_dp": style_dp, "text_mask": text_mask, "text_embed": text_embed_dp}
            )
            dur = dp_out.get("duration", next(iter(dp_out.values())))
            dur = (dur / args.speed).astype(np.float32)
            timings["duration_predictor"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            te_out = te_model.predict(
                {"style_ttl": style_ttl, "text_mask": text_mask, "text_embed": text_embed_te}
            )
            text_emb = te_out.get("text_emb", next(iter(te_out.values())))
            timings["text_encoder"] += time.perf_counter() - t0

            xt, latent_mask = sample_noisy_latent(dur)
            total_step_np = np.array([args.steps], dtype=np.float32)
            t0 = time.perf_counter()
            for step in range(args.steps):
                current_step = np.array([step], dtype=np.float32)
                ve_out = ve_model.predict(
                    {
                        "noisy_latent": xt,
                        "text_emb": text_emb.astype(np.float32),
                        "style_ttl": style_ttl,
                        "latent_mask": latent_mask,
                        "text_mask": text_mask,
                        "current_step": current_step,
                        "total_step": total_step_np,
                    }
                )
                xt = ve_out.get("denoised_latent", next(iter(ve_out.values())))
            timings["vector_estimator"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            voc_out = voc_model.predict({"latent": xt.astype(np.float32)})
            wav = voc_out.get("wav_tts", next(iter(voc_out.values())))
            timings["vocoder"] += time.perf_counter() - t0

            wav_parts.append(wav)
            dur_parts.append(dur)

        timings["total"] = sum(timings.values())

        wav_cat = None
        dur_cat = 0.0
        for wav, dur in zip(wav_parts, dur_parts):
            wav_trim = wav[:, : int(sample_rate * dur.item())]
            if wav_cat is None:
                wav_cat = wav_trim
            else:
                silence = np.zeros((1, int(args.silence * sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, silence, wav_trim], axis=1)
            dur_cat += float(dur.item()) + args.silence

        return {"timings": timings, "wav": wav_cat, "duration": dur_cat}

    # warmup
    for _ in range(args.warmup):
        run_once()

    # benchmark runs
    results = []
    rss_before = get_rss_mb()
    for _ in range(args.runs):
        results.append(run_once())
    rss_after = get_rss_mb()
    peak_rss = get_peak_rss_mb()

    # aggregate timings
    keys = results[0]["timings"].keys()
    print("\nTiming (seconds):")
    for k in keys:
        vals = [r["timings"][k] for r in results]
        avg = float(np.mean(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        print(f"- {k}: {avg:.4f} ± {std:.4f}")

    print("\nMemory (MB):")
    if rss_before and rss_after:
        print(f"- RSS before: {rss_before:.1f}")
        print(f"- RSS after:  {rss_after:.1f}")
    print(f"- Peak RSS:   {peak_rss:.1f}")

    # save sample from last run
    wav = results[-1]["wav"]
    save_wav(args.out_wav, wav, sample_rate)
    print(f"\nWAV saved: {args.out_wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
