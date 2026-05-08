#!/usr/bin/env python3
import sys
from pathlib import Path

try:
    import aaf2
except Exception as e:
    print("pyaaf2 (aaf2) kütüphanesi gerekli. requirements.txt ile kurun.")
    raise


def describe_descriptor(desc):
    info = {}
    try:
        info["classdef"] = getattr(getattr(desc, "classdef", None), "name", str(type(desc)))
        info["sample_rate"] = getattr(desc, "sample_rate", None)
        info["length"] = getattr(desc, "length", None)
        info["channels"] = getattr(desc, "channels", None)
        info["locators_count"] = len(getattr(desc, "locators", []) or [])
        info["locator_paths"] = []
        for loc in getattr(desc, "locators", []) or []:
            # NetworkLocator / TextLocator may have different attrs
            path = getattr(loc, "path", None) or getattr(loc, "url", None) or getattr(loc, "value", None)
            info["locator_paths"].append(path)
    except Exception:
        pass
    return info


def inspect_aaf(aaf_path: Path):
    print(f"AAf inceleme: {aaf_path}")
    with aaf2.open(str(aaf_path), mode='r') as f:
        print("- Mobs:")
        for mob in f.content.mobs:
            mob_kind = mob.__class__.__name__
            print(f"  * {mob_kind}: name={mob.name}")
            try:
                desc = getattr(mob, "descriptor", None)
                if desc:
                    di = describe_descriptor(desc)
                    print(f"    - Descriptor: {di}")
            except Exception:
                pass
            for slot in getattr(mob, "slots", []) or []:
                try:
                    seg = slot.segment
                    seg_kind = getattr(seg, "classdef", None)
                    seg_kind = getattr(seg_kind, "name", seg.__class__.__name__)
                    comps = list(getattr(seg, "components", []) or [])
                    print(f"    - Slot: id={slot.slot_id} seg_kind={seg_kind} length={getattr(seg, 'length', None)} comps={len(comps)}")
                    for i, c in enumerate(comps[:10]):
                        # Heuristic classification to distinguish SourceClip vs Filler vs Sequence
                        try:
                            cname = getattr(getattr(c, "classdef", None), "name", c.__class__.__name__)
                            clen = getattr(c, "length", None)
                            # Identify SourceClip
                            if hasattr(c, 'source_slot') or hasattr(c, 'source_mob'):
                                cname = 'SourceClip'
                                # Try to print referenced mob name
                                ref_name = None
                                try:
                                    ref_mob = getattr(c, 'source_mob', None)
                                    if ref_mob:
                                        ref_name = getattr(ref_mob, 'name', None)
                                except Exception:
                                    pass
                                if ref_name:
                                    print(f"        • comp[{i}]: {cname} length={clen} => mob={ref_name}")
                                else:
                                    print(f"        • comp[{i}]: {cname} length={clen}")
                            elif hasattr(c, 'components'):
                                # Nested Sequence
                                cname = 'Sequence'
                                print(f"        • comp[{i}]: {cname} length={clen} (nested)")
                            else:
                                # Likely Filler or unknown
                                if cname == 'ClassDefinition':
                                    cname = 'Filler?'
                                print(f"        • comp[{i}]: {cname} length={clen}")
                        except Exception:
                            pass
                except Exception:
                    pass


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python tools/inspect_aaf.py <path/to/file.aaf>")
        sys.exit(1)
    aaf_path = Path(sys.argv[1]).expanduser()
    if not aaf_path.exists():
        print(f"Dosya bulunamadı: {aaf_path}")
        sys.exit(2)
    inspect_aaf(aaf_path)


if __name__ == "__main__":
    main()