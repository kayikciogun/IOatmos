#!/usr/bin/env python3
"""
clap_search.py — Index'lenmiş kütüphanede hibrit arama.

Cosine Similarity Based:
- Her query kendi percentile'larını kullanır (diğer query'ler etkilemez)
- 0-100 aralığında normalize edilmiş skorlar
- Per-query min/max normalizasyonu (global batch değil)
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


# ─── Cosine Search (raw similarity, no percentile tricks) ─────────────────

def _scale_to_100(raw_scores: np.ndarray) -> np.ndarray:
    """Ham cosine similarity'leri [0,1] aralığından kabaca [0,100] eşler.
    Basitçe: score = raw * 100, ama normalize edici hiçbir şey yapmaz.
    """
    return np.clip(raw_scores * 100.0, 0.0, 100.0)


def search_cosine_single(
    query: str,
    idx: dict,
    model,
    text_weight: float = 0.55,
) -> list[dict]:
    """Tek bir query string için arama yapar.
    Döndürür: [{path, score, raw_sim, audio_sim, text_sim}, ...]
    score: raw cosine * 100, gerçek eşleşmeyi yansıtır.
    """
    has_text = idx["text"] is not None

    # Single query embedding
    pos_emb = model.get_text_embedding([query], use_tensor=False)
    pos_emb = pos_emb / np.linalg.norm(pos_emb, axis=1, keepdims=True)

    # Cosine similarities (matrices already normalized from indexing)
    sim_audio = (pos_emb @ idx["audio"].T)[0]
    sim_text = None
    if has_text:
        sim_text = (pos_emb @ idx["text"].T)[0]

    # Hybrid score
    if has_text and sim_text is not None:
        hybrid = (1 - text_weight) * sim_audio + text_weight * sim_text
    else:
        hybrid = sim_audio

    # Scale to 0-100 WITHOUT percentile tricks (just multiply by 100)
    scaled = _scale_to_100(hybrid)

    # Sort descending
    order = np.argsort(-scaled)
    return [
        {
            "path": idx["paths"][j],
            "score": float(scaled[j]),
            "raw_sim": float(hybrid[j]),
            "audio_sim": float(sim_audio[j]),
            "text_sim": float(sim_text[j]) if sim_text is not None else 0.0,
        }
        for j in order
    ]


def smart_search(
    query,
    idx: dict,
    model,
    top_k: int = 3,
    text_weight: float = 0.55,
    min_score: float | None = None,
) -> dict:
    """Tekil arama veya multi-query arama.
    - query: str veya list[str]
    - Eğer liste ise: her query ayrı aranır, sonuçlar çeşitliliğe göre merge edilir.
    """
    if isinstance(query, str):
        queries = [query]
    elif isinstance(query, list):
        queries = [q for q in query if q and str(q).strip()]
    else:
        queries = [str(query)]

    if not queries:
        return {"results": [], "source": "cosine", "confidence": "poor", "best_score": 0}

    # Tek query → basit arama
    if len(queries) == 1:
        all_results = search_cosine_single(
            query=queries[0],
            idx=idx,
            model=model,
            text_weight=text_weight,
        )
        if min_score is not None:
            all_results = [r for r in all_results if r["score"] >= min_score]
        candidates = all_results[:top_k]
        best = candidates[0]["score"] if candidates else 0
        return {
            "results": candidates,
            "source": "cosine",
            "confidence": _score_to_confidence(best),
            "best_score": best,
        }

    # Multi-query: her query ayrı aranır, sonra çeşitlilik + skor ile merge
    per_query = {}
    for q in queries:
        per_query[q] = search_cosine_single(
            query=q, idx=idx, model=model, text_weight=text_weight,
        )

    # Merge: her dosyanın en yüksek skorunu al
    file_best = {}
    for q, results in per_query.items():
        for i, r in enumerate(results):
            path = r["path"]
            if path not in file_best or r["score"] > file_best[path]["score"]:
                file_best[path] = {
                    **r,
                    "best_query": q,
                    "query_rank": i + 1,
                }

    all_merged = sorted(file_best.values(), key=lambda x: x["score"], reverse=True)
    if min_score is not None:
        all_merged = [r for r in all_merged if r["score"] >= min_score]
    candidates = all_merged[:top_k]
    best = candidates[0]["score"] if candidates else 0

    return {
        "results": candidates,
        "source": "cosine_multi",
        "confidence": _score_to_confidence(best),
        "best_score": best,
        "num_queries": len(queries),
    }


def _score_to_confidence(score: float) -> str:
    """Score is raw cosine * 100.
    CLAP cosine scale: great matches ~0.80, okay ~0.55, weak ~0.35, unrelated ~<0.20.
    """
    if score > 80:
        return "excellent"
    if score > 65:
        return "high"
    if score > 45:
        return "medium"
    if score > 30:
        return "low"
    return "poor"


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
    print(f"   audio={idx['audio'].shape}, text_emb={'evet' if has_text else 'hayır'}, score=cosine(0-100)")
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
            top_k=args.top_k,
            min_score=args.min_score,
        )
        results.append({"tag": tag, **res})

    print("=" * 78)
    for item in results:
        print(f"\n🏷️  '{item['tag']}' [Conf: {item['confidence'].upper()}, Best: {item['best_score']:.1f}/100]")
        print("-" * 78)
        if not item["results"]:
            print(f"  (sonuç bulunamadı)")
            continue
        for rank, r in enumerate(item["results"], 1):
            disp = r["path"] if args.full_path else Path(r["path"]).name
            print(f"  {rank}. Score: {r['score']:.1f}/100  {disp}")
            print(f"     └─ raw={r['raw_sim']:.4f}  audio={r['audio_sim']:.4f}  text={r['text_sim']:.4f}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
