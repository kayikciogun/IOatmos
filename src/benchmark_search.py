#!/usr/bin/env python3
"""
CLAP arama ağırlıkları benchmark — VLM description sorguları üzerinde grid search.

Kullanım:
    python src/benchmark_search.py \\
        --analysis outputs/test1_frames/jsonlar/test1_sound_analysis.json \\
        --index_path ./index.npz

Proxy metrikler (ground truth yok):
  - mean_best: ortalama top-1 skor (0-100)
  - mean_margin: top1 - top2 farkı (kararlılık)
  - keyword_hit: sorgu anahtar kelimelerinin dosya adında geçme oranı
  - low_conf_pct: skor < 45 olan sahne oranı
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from clap_search import (
    load_index,
    smart_search,
    preprocess_query,
    build_model,
    DEFAULT_TEXT_WEIGHT,
    DEFAULT_BM25_BOOST_WEIGHT,
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "with", "in", "on", "at", "to", "for",
    "from", "into", "during", "through", "between", "inside", "outside", "nearby",
    "near", "faint", "soft", "gentle", "subtle", "quiet", "low", "steady",
    "constant", "light", "active", "continuous", "day", "daytime", "night",
}


def _query_keywords(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) > 3 and t not in STOPWORDS]


def _filename_hit(query: str, path: str) -> bool:
    stem = Path(path).stem.lower()
    keys = _query_keywords(query)
    if not keys:
        return False
    return any(k in stem for k in keys)


@dataclass
class SceneCase:
    scene_id: int
    query: str


@dataclass
class ConfigResult:
    text_weight: float
    bm25_boost_weight: float
    use_bm25: bool
    mean_best: float
    mean_margin: float
    keyword_hit: float
    low_conf_pct: float
    composite: float


def _scene_description(scene: dict) -> str:
    desc = (scene.get("description") or scene.get("sound_description") or "").strip()
    if desc:
        return desc
    parts = []
    for layer in scene.get("layers", []):
        q = (layer.get("query") or "").strip()
        if q:
            parts.append(q)
    return " ".join(parts).strip()


def load_cases(analysis_path: str) -> list[SceneCase]:
    with open(analysis_path, encoding="utf-8") as f:
        scenes = json.load(f)
    cases = []
    for scene in scenes:
        q = _scene_description(scene)
        if q:
            cases.append(SceneCase(scene["scene_id"], q))
    return cases


def evaluate_config(
    cases: list[SceneCase],
    idx: dict,
    model,
    text_weight: float,
    bm25_boost_weight: float,
    use_bm25: bool,
    top_k: int = 3,
) -> ConfigResult:
    bests, margins, hits, low = [], [], 0, 0

    for case in cases:
        res = smart_search(
            query=case.query,
            idx=idx,
            model=model,
            top_k=top_k,
            text_weight=text_weight,
            layer_type=None,
            bm25_boost_weight=bm25_boost_weight,
            use_bm25=use_bm25,
        )
        results = res["results"]
        if not results:
            low += 1
            continue

        best = results[0]["score"]
        bests.append(best)
        if best < 45:
            low += 1
        if len(results) > 1:
            margins.append(best - results[1]["score"])
        if _filename_hit(case.query, results[0]["path"]):
            hits += 1

    n = len(cases) or 1
    mean_best = float(np.mean(bests)) if bests else 0.0
    mean_margin = float(np.mean(margins)) if margins else 0.0
    keyword_hit = hits / n
    low_conf_pct = 100.0 * low / n
    composite = (
        0.40 * mean_best
        + 0.30 * keyword_hit * 100
        + 0.20 * mean_margin
        + 0.10 * (100.0 - low_conf_pct)
    )
    return ConfigResult(
        text_weight=text_weight,
        bm25_boost_weight=bm25_boost_weight,
        use_bm25=use_bm25,
        mean_best=mean_best,
        mean_margin=mean_margin,
        keyword_hit=keyword_hit,
        low_conf_pct=low_conf_pct,
        composite=composite,
    )


def print_detail(
    cases: list[SceneCase],
    idx: dict,
    model,
    cfg: ConfigResult,
    top_k: int = 3,
) -> None:
    print(f"\n{'='*90}")
    print(
        f"DETAY — text_weight={cfg.text_weight}  bm25={cfg.bm25_boost_weight}  "
        f"use_bm25={cfg.use_bm25}"
    )
    print(f"{'='*90}")
    for case in cases:
        res = smart_search(
            query=case.query,
            idx=idx,
            model=model,
            top_k=top_k,
            text_weight=cfg.text_weight,
            layer_type=None,
            bm25_boost_weight=cfg.bm25_boost_weight,
            use_bm25=cfg.use_bm25,
        )
        pq = preprocess_query(case.query, None)
        print(f"\n[S{case.scene_id:02d}]")
        print(f"  Q: {case.query[:90]}...")
        print(f"  → preprocess: {pq[:80]}")
        for i, r in enumerate(res["results"][:top_k], 1):
            name = Path(r["path"]).name
            hit = "✓" if _filename_hit(case.query, r["path"]) else " "
            print(
                f"  {i}. [{hit}] {r['score']:5.1f}  "
                f"a={r['audio_sim']:.3f} t={r['text_sim']:.3f}  {name[:70]}"
            )


def main():
    parser = argparse.ArgumentParser(description="CLAP search weight benchmark")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--index_path", default="./index.npz")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--detail", action="store_true", help="En iyi config için detay yazdır")
    parser.add_argument("--output", help="JSON sonuç dosyası")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    args = parser.parse_args()

    cases = load_cases(args.analysis)
    if not cases:
        print("❌ Analiz dosyasında description sorgusu yok.")
        return

    idx = load_index(args.index_path)
    print(f"📦 Index: {len(idx['paths'])} dosya, BM25={'evet' if idx['bm25'] else 'hayır'}")
    print(f"🎬 Test: {len(cases)} sahne sorgusu ({args.analysis})\n")

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"⏳ CLAP yükleniyor ({device})...")
    model, _ = build_model(idx["preset"], device)
    print("✅ Model hazır\n")

    text_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    bm25_weights = [0.0, 0.15, 0.25, 0.4]
    use_bm25_opts = [False, True] if idx["bm25"] else [False]

    results: list[ConfigResult] = []
    total = len(text_weights) * len(bm25_weights) * len(use_bm25_opts)
    n = 0
    for tw, bw, ub in product(text_weights, bm25_weights, use_bm25_opts):
        n += 1
        print(f"\r🔬 Grid {n}/{total}  tw={tw} bm25={bw} use={ub}   ", end="", flush=True)
        results.append(evaluate_config(cases, idx, model, tw, bw, ub, args.top_k))
    print()

    results.sort(key=lambda r: r.composite, reverse=True)

    print(f"\n{'='*100}")
    print(f"{'#':>2}  {'tw':>4}  {'bm25':>5}  {'BM25':>4}  "
          f"{'best':>6}  {'margin':>6}  {'kw%':>5}  {'low%':>5}  {'composite':>9}")
    print("-" * 100)
    for i, r in enumerate(results[:15], 1):
        print(
            f"{i:2d}  {r.text_weight:4.2f}  {r.bm25_boost_weight:5.2f}  "
            f"{'Y' if r.use_bm25 else 'N':>4}  "
            f"{r.mean_best:6.1f}  {r.mean_margin:6.2f}  "
            f"{r.keyword_hit*100:5.1f}  {r.low_conf_pct:5.1f}  {r.composite:9.1f}"
        )

    best = results[0]
    current = next(
        (
            r for r in results
            if r.text_weight == DEFAULT_TEXT_WEIGHT
            and r.bm25_boost_weight == DEFAULT_BM25_BOOST_WEIGHT
            and r.use_bm25
        ),
        None,
    )
    print(f"\n🏆 Önerilen: text_weight={best.text_weight}, bm25_boost_weight={best.bm25_boost_weight}, "
          f"use_bm25={best.use_bm25}")
    if current:
        print(
            f"📌 Mevcut ({DEFAULT_TEXT_WEIGHT}/{DEFAULT_BM25_BOOST_WEIGHT}/True): "
            f"composite={current.composite:.1f}  vs önerilen {best.composite:.1f}  "
            f"(Δ {best.composite - current.composite:+.1f})"
        )

    if args.output:
        out = {
            "analysis": args.analysis,
            "index_path": args.index_path,
            "num_cases": len(cases),
            "recommended": {
                "text_weight": best.text_weight,
                "bm25_boost_weight": best.bm25_boost_weight,
                "use_bm25": best.use_bm25,
            },
            "current_defaults": {
                "text_weight": DEFAULT_TEXT_WEIGHT,
                "bm25_boost_weight": DEFAULT_BM25_BOOST_WEIGHT,
                "use_bm25": True,
            },
            "ranking": [
                {
                    "text_weight": r.text_weight,
                    "bm25_boost_weight": r.bm25_boost_weight,
                    "use_bm25": r.use_bm25,
                    "mean_best": round(r.mean_best, 2),
                    "mean_margin": round(r.mean_margin, 2),
                    "keyword_hit_pct": round(r.keyword_hit * 100, 1),
                    "low_conf_pct": round(r.low_conf_pct, 1),
                    "composite": round(r.composite, 2),
                }
                for r in results
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Sonuçlar: {args.output}")

    if args.detail:
        print_detail(cases, idx, model, best, args.top_k)


if __name__ == "__main__":
    main()
