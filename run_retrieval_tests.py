#!/usr/bin/env python3
"""
run_retrieval_tests.py — Dinamik weight karşılaştırma testleri.

Üç senaryo, tek seferde çalışır:
  1. Sabit weight (eski davranış)
  2. Dinamik weight (yeni)
  3. Waterfall + min weight guarantee

Kullanım:
    python3 run_retrieval_tests.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import torch
from pathlib import Path

from clap_search import load_index, build_model, smart_search, search

INDEX_PATH = "./index.npz"


def fmt_result(r, rank):
    """Tek sonucu formatlı string olarak döndür."""
    tw = r.get("text_weight_used", 0.40)
    cap_len = r.get("cap_length", -1)
    return (
        f"  {rank}. {r['score']:+.4f}  {Path(r['path']).name}\n"
        f"     [a={r['audio_sim']:+.3f} t={r['text_sim']:+.3f} tw={tw:.2f} cap={cap_len}]"
    )


def run_test(name, tag, ucs_category, dynamic, text_w=0.4):
    """Bir test senaryosu çalıştır ve sonuçları yazdır."""
    print(f"\n{'='*78}")
    print(f"  TEST: {name}")
    print(f"  Query: '{tag}'  |  UCS: {ucs_category or 'None'}  |  Dynamic: {dynamic}")
    print(f"{'='*78}")

    t0 = time.time()
    if ucs_category:
        res = smart_search(tag, ucs_category, idx, model, top_k=5)
    else:
        res = search(
            tags=[tag], idx=idx, model=model,
            dynamic_weight=dynamic, text_weight=text_w, top_k=5
        )[0]

    elapsed = time.time() - t0

    for rank, r in enumerate(res["results"], 1):
        print(fmt_result(r, rank))
    print(f"  ({elapsed:.2f}s)")
    return res["results"]


def compare_queries(static_res, dyn_res):
    """Sabit vs dinamik sonuçları karşılaştır."""
    print(f"\n{'='*78}")
    print("  KARŞILAŞTIRMA: Sabit (0.40) vs Dinamik")
    print(f"{'='*78}")

    s_paths = [r["path"] for r in static_res]
    d_paths = [r["path"] for r in dyn_res]

    # Aynı dosyalar farklı sırada mı?
    common = set(s_paths) & set(d_paths)
    print(f"  Ortak dosya sayısı (top-5): {len(common)}")

    # #1 pozisyon değişti mi?
    if s_paths and d_paths:
        if s_paths[0] == d_paths[0]:
            print(f"  #1 aynı: {Path(s_paths[0]).name}")
        else:
            print(f"  #1 DEĞİŞTİ!")
            print(f"    Sabit   #1: {Path(s_paths[0]).name}  ({static_res[0]['score']:+.4f})")
            print(f"    Dinamik #1: {Path(d_paths[0]).name}  ({dyn_res[0]['score']:+.4f})")

    # Weight dağılımı
    weights = [r.get("text_weight_used", 0.40) for r in dyn_res]
    w_str = ", ".join(f"{w:.2f}" for w in weights)
    print(f"  Dinamik top-5 weight'ler: [{w_str}]")


if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        print(f"❌ Index bulunamadı: {INDEX_PATH}")
        sys.exit(1)

    print("📦 Index yükleniyor...")
    idx = load_index(INDEX_PATH)
    has_cap = idx.get("cap_lengths") is not None
    print(f"   {len(idx['paths'])} dosya, cap_lengths={'EVET' if has_cap else 'YOK'}")

    if not has_cap:
        print("\n⚠️  Index'te caption_lengths yok.")
        print("   Önce çalıştır: python3 src/update_caption_lengths.py")
        sys.exit(1)

    # Modeli bir kez yükle
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n⏳ Model yükleniyor ({device})...")
    model, _ = build_model(idx["preset"], device)
    print("✅ Hazır\n")

    # --- TEST 1: Sabit weight ---
    static_res = run_test(
        "SABİT WEIGHT (text_weight=0.40)",
        tag="car slow by crunchy snow tire hum",
        ucs_category=None,
        dynamic=False,
        text_w=0.4,
    )

    # --- TEST 2: Dinamik weight ---
    dyn_res = run_test(
        "DİNAMİK WEIGHT",
        tag="car slow by crunchy snow tire hum",
        ucs_category=None,
        dynamic=True,
    )

    # --- TEST 3: Waterfall + min weight ---
    waterfall_res = run_test(
        "WATERFALL + MIN WEIGHT GUARANTEE",
        tag="cascading waterfall crashing into pool",
        ucs_category="WATR-WATERFALL",
        dynamic=True,
    )

    # Karşılaştırma
    compare_queries(static_res, dyn_res)

    print(f"\n{'='*78}")
    print("  TÜM TESTLER TAMAMLANDI")
    print(f"{'='*78}\n")
