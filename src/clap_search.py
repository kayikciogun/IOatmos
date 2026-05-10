#!/usr/bin/env python3
"""
clap_search.py — Index'lenmiş kütüphanede hibrit arama.

Sadeleştirilmiş versiyon:
- text_weight = 0.65 sabit
- Negatif query, dynamic weight, UCS kategori mask kaldırıldı (etkisiz/ kullanılmıyordu)
"""

import argparse
import os
import sys
import re
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

    # Captions varsa BM25 corpus oluştur
    bm25 = None
    if "captions" in data.files:
        from rank_bm25 import BM25Okapi
        captions = data["captions"].tolist()
        tokenized = [cap.lower().split() for cap in captions]
        bm25 = BM25Okapi(tokenized)

    return {
        "paths":       data["paths"].tolist(),
        "audio":       data["embeddings"],
        "text":        data["text_embeddings"] if "text_embeddings" in data.files else None,
        "bm25":        bm25,
        "preset":      str(data["preset"]) if "preset" in data.files else "natural",
    }


def build_model(preset: str, device: str):
    cfg = PRESETS[preset]
    model = laion_clap.CLAP_Module(
        enable_fusion=cfg["fusion"], amodel=cfg["amodel"], device=device
    )
    if cfg["model_id"] is not None:
        model.load_ckpt(model_id=cfg["model_id"])
    else:
        ckpt = download_ckpt_if_needed(cfg["ckpt_url"], cfg["ckpt_name"])
        model.load_ckpt(ckpt)
    return model, cfg



def search(
    tags: list[str],
    idx: dict,
    model,
    top_k: int = 3,
    rrf_k: int = 60,
    min_score: float | None = None,
) -> list[dict]:
    """
    RRF Fusion: BM25 (lexical) + CLAP audio + CLAP text.
    Skor değil sıralama kullanır — ölçek sorunu yok.
    """
    has_text = idx["text"] is not None
    has_bm25 = idx.get("bm25") is not None
    n = len(idx["paths"])

    # Pozitif query embedding
    pos_emb = model.get_text_embedding(tags, use_tensor=False)
    pos_emb = pos_emb / np.linalg.norm(pos_emb, axis=1, keepdims=True)

    # Audio ve text similarity
    sim_audio = pos_emb @ idx["audio"].T
    sim_text = (pos_emb @ idx["text"].T) if has_text else None

    # Combined CLAP score (audio + text)
    if has_text:
        clap_score = 0.3 * sim_audio + 0.7 * sim_text
    else:
        clap_score = sim_audio

    output = []
    for i, tag in enumerate(tags):

        # CLAP rank (dosya_idx → rank)
        clap_order = np.argsort(-clap_score[i])
        clap_rank = np.empty(n, dtype=int)
        clap_rank[clap_order] = np.arange(n)

        # BM25 rank
        if has_bm25:
            bm25_scores = idx["bm25"].get_scores(tag.lower().split())
            bm25_order = np.argsort(-bm25_scores)
            bm25_rank = np.empty(n, dtype=int)
            bm25_rank[bm25_order] = np.arange(n)

        # RRF: 1/(k + rank) — her sistemden katkı
        if has_bm25:
            rrf = (1 / (rrf_k + clap_rank) + 1 / (rrf_k + bm25_rank))
        else:
            rrf = 1 / (rrf_k + clap_rank)

        top_indices = np.argsort(-rrf)
        results = []
        for j in top_indices:
            if min_score is not None and rrf[j] < min_score:
                continue
            if len(results) >= top_k:
                break
            results.append({
                "path": idx["paths"][j],
                "score": float(rrf[j]),
                "clap_score": float(clap_score[i, j]),
                "audio_sim": float(sim_audio[i, j]),
                "text_sim": float(sim_text[i, j]) if sim_text is not None else 0.0,
                "bm25_rank": int(bm25_rank[j]) if has_bm25 else -1,
                "clap_rank": int(clap_rank[j]),
            })
        output.append({"tag": tag, "results": results})

    return output


def smart_search(query, idx, model, top_k=3):
    """
    RRF Fusion arama: BM25 + CLAP audio + CLAP text.
    Re-rank kaldırıldı, sıralama tabanlı fusion kullanılıyor.
    """
    all_results = search(
        tags=[query],
        idx=idx,
        model=model,
        top_k=top_k,
    )
    candidates = all_results[0]["results"]

    # RRF skorları ~0.01-0.06 arasında, eşikler buna göre ayarlandı
    final_best_score = candidates[0]["score"] if candidates else 0

    return {
        "results": candidates,
        "source": "rrf",
        "confidence": "high" if final_best_score > 0.040 else "medium" if final_best_score > 0.030 else "low",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="./index.npz")
    parser.add_argument("--tags", nargs="+", required=False)
    parser.add_argument("--tags_file")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--full_path", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    args = parser.parse_args()

    if not args.tags and not args.tags_file:
        parser.error("--tags veya --tags_file gerekli")
    tags = args.tags or [l.strip() for l in open(args.tags_file, encoding="utf-8") if l.strip()]

    idx = load_index(args.index_path)
    has_text = idx["text"] is not None

    print(f"📦 Index: {len(idx['paths'])} dosya, preset={idx['preset']}")
    print(f"   audio={idx['audio'].shape}, text_emb={'evet' if has_text else 'hayır'}, bm25={'evet' if idx.get('bm25') is not None else 'hayır'}")
    print(f"   fusion=RRF (k=60), clap_w=0.7")
    print()

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"⏳ Model yükleniyor ({device})...")
    model, _ = build_model(idx["preset"], device)
    print("✅ Hazır\n")

    results = []
    for tag in tags:
        res = smart_search(
            query=tag,
            idx=idx,
            model=model,
            top_k=args.top_k
        )
        results.append({"tag": tag, **res})

    print("=" * 78)
    for item in results:
        print(f"\n🏷️  '{item['tag']}' [Source: {item['source']}, Conf: {item['confidence']}]")
        print("-" * 78)
        if not item["results"]:
            print(f"  (sonuç bulunamadı)")
            continue
        for rank, r in enumerate(item["results"], 1):
            disp = r["path"] if args.full_path else Path(r["path"]).name
            print(f"  {rank}. {r['score']:+.4f}  {disp}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
