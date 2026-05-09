#!/usr/bin/env python3
"""
update_caption_lengths.py — Mevcut index.npz'ye caption uzunluklarını ekler.

Bir kez çalıştırılır, re-index gerekmez.
Kullanım:
    python src/update_caption_lengths.py --index_path ./index.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from clap_index import read_bext_description
from caption_enrichment import enrich_caption


def update_caption_lengths(index_path: str):
    data = np.load(index_path, allow_pickle=True)
    paths = data["paths"].tolist()
    catids = data["catids"].tolist() if "catids" in data.files else [""] * len(paths)

    print(f"📏 {len(paths)} dosya için caption uzunluğu hesaplanıyor...")

    cap_lengths = []
    for i, (p, c) in enumerate(zip(paths, catids)):
        bext = read_bext_description(p) if p.lower().endswith(".wav") else ""
        cap = enrich_caption(p, bext, c)
        cap_lengths.append(len(cap.split()))
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}...", end="\r")

    # Mevcut tüm array'leri koru, sadece caption_lengths ekle
    save_data = {key: data[key] for key in data.files}
    save_data["caption_lengths"] = np.array(cap_lengths, dtype=np.int16)

    np.savez(index_path, **save_data)

    from collections import Counter

    buckets = Counter(
        "short" if l < 15 else "medium" if l < 30 else "rich" for l in cap_lengths
    )
    print(f"\n✅ Tamamlandı")
    print(f"  short  (<15w):  {buckets['short']:4d} dosya")
    print(f"  medium (15-30w): {buckets['medium']:4d} dosya")
    print(f"  rich   (>30w):  {buckets['rich']:4d} dosya")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="./index.npz", help="Index dosyası yolu")
    args = parser.parse_args()

    if not Path(args.index_path).exists():
        print(f"❌ Index bulunamadı: {args.index_path}")
        sys.exit(1)

    update_caption_lengths(args.index_path)


if __name__ == "__main__":
    main()
