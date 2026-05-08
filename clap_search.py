#!/usr/bin/env python3
"""
clap_search.py — Index'lenmiş kütüphanede tag araması.

Audio + filename hibrit skor (default):
  combined = α * audio_sim + (1-α) * filename_sim
  α default = 0.7 (audio ağırlıklı)
  --text_weight ile tersine çevir (örn 0.5 → filename daha güçlü)

Kullanım:
    python clap_search.py --tags "rain on tin roof" "thunder rumble"
    python clap_search.py --tags "footsteps gravel" --text_weight 0.5
    python clap_search.py --tags "dog barking" --text_weight 0  # sadece audio
    python clap_search.py --tags "rain" --text_weight 1.0       # sadece filename
"""

import argparse
import os
import sys
from pathlib import Path

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import laion_clap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clap_index import PRESETS, download_ckpt_if_needed


def load_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index yok: {path}. Önce: python clap_index.py --audio_dir ...")
    data = np.load(path, allow_pickle=True)
    return {
        "paths": data["paths"].tolist(),
        "audio": data["embeddings"],
        "text": data["text_embeddings"] if "text_embeddings" in data.files else None,
        "preset": str(data["preset"]) if "preset" in data.files else "natural",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="./index.npz")
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--tags_file")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--full_path", action="store_true")
    parser.add_argument(
        "--text_weight",
        type=float,
        default=0.3,
        help="Filename embedding ağırlığı (0=sadece audio, 1=sadece filename, default 0.3)",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--show_breakdown", action="store_true",
                        help="Her sonuç için audio_sim ve text_sim ayrı göster")
    args = parser.parse_args()

    if not args.tags and not args.tags_file:
        parser.error("--tags veya --tags_file gerekli")
    tags = args.tags or [l.strip() for l in open(args.tags_file, encoding="utf-8") if l.strip()]

    idx = load_index(args.index_path)
    cfg = PRESETS[idx["preset"]]
    has_text = idx["text"] is not None

    print(f"📦 Index: {len(idx['paths'])} dosya, preset={idx['preset']}")
    print(f"   audio={idx['audio'].shape}, filename_emb={'evet' if has_text else 'hayır'}")

    text_w = args.text_weight
    if not has_text and text_w > 0:
        print(f"⚠️  Index'te filename embedding yok, text_weight=0 yapılıyor")
        text_w = 0.0
    audio_w = 1.0 - text_w
    print(f"🏷️  {len(tags)} etiket | weights: audio={audio_w:.2f}, filename={text_w:.2f}\n")

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"⏳ Text encoder ({device})...")
    model = laion_clap.CLAP_Module(
        enable_fusion=cfg["fusion"], amodel=cfg["amodel"], device=device
    )
    if cfg["model_id"] is not None:
        model.load_ckpt(model_id=cfg["model_id"])
    else:
        ckpt = download_ckpt_if_needed(cfg["ckpt_url"], cfg["ckpt_name"])
        model.load_ckpt(ckpt)
    print("✅ Hazır\n")

    text_emb = model.get_text_embedding(tags, use_tensor=False)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

    # [n_tags, n_audio] iki similarity
    sim_audio = text_emb @ idx["audio"].T
    if has_text and text_w > 0:
        sim_text = text_emb @ idx["text"].T
        sim_combined = audio_w * sim_audio + text_w * sim_text
    else:
        sim_text = None
        sim_combined = sim_audio

    paths = idx["paths"]
    print("=" * 78)
    for i, tag in enumerate(tags):
        scores = sim_combined[i]
        order = np.argsort(-scores)
        print(f"\n🏷️  '{tag}'")
        print("-" * 78)
        shown = 0
        for j in order:
            if args.min_score is not None and scores[j] < args.min_score:
                break
            if shown >= args.top_k:
                break
            disp = paths[j] if args.full_path else Path(paths[j]).name
            if args.show_breakdown and sim_text is not None:
                print(f"  {shown+1}. {scores[j]:+.4f} "
                      f"(a={sim_audio[i,j]:+.3f} t={sim_text[i,j]:+.3f})  {disp}")
            else:
                print(f"  {shown+1}. {scores[j]:+.4f}  {disp}")
            shown += 1
        if shown == 0:
            print(f"  (min_score={args.min_score} üstünde sonuç yok)")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()