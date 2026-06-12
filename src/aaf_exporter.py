"""
External-Linked AAF Writer (AMA + fallback, timeline uyumlu)
"""

try:
    import aaf2
    AAF2_AVAILABLE = True
except ImportError:
    AAF2_AVAILABLE = False

import json
from pathlib import Path
import wave
from fractions import Fraction
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil


def _resolve_rate(value, fallback=30.0):
    # Standard broadcast rates lookup (Exact numerators/denominators)
    BROADCAST_RATES = {
        23.976: (24000, 1001),
        29.97: (30000, 1001),
        59.94: (60000, 1001),
        47.952: (48000, 1001)
    }
    
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = float(fallback)
    if val <= 0:
        val = float(fallback)
        
    # 1. Check standard broadcast rates with tolerance
    for rate, (num, den) in BROADCAST_RATES.items():
        if abs(val - rate) < 0.01:
            exact_val = num / den
            return exact_val, num, den

    # 2. For integer rates
    if abs(val - round(val)) < 0.001:
        int_val = int(round(val))
        return float(int_val), int_val, 1

    # 3. Fallback
    frac = Fraction(val).limit_denominator(100000)
    return val, frac.numerator, frac.denominator


def _make_external_source_mob(f, media_path, edit_rate):
    """
    create_ama_link başarısız olduğunda manuel external mob zinciri yaratır.
    MasterMob → SourceMob yapısı oluşturur (DAW uyumluluğu için).
    ImportDescriptor kullanır — WAV, MP3, AIFF vs. tüm formatları destekler.
    """
    file_name = Path(media_path).stem  # uzantısız ad
    
    # 1. SourceMob (dosyaya referans veren mob)
    src_mob = f.create.SourceMob()
    src_mob.name = file_name

    descriptor = f.create.ImportDescriptor()
    
    # URL oluştur — file:// + absolute path (/ ile başlıyor)
    abs_path = Path(media_path).resolve()
    url = "file://" + abs_path.as_posix()  # file:///Users/... olur
    
    locator = f.create.NetworkLocator()
    locator["URLString"].value = url
    descriptor["Locator"].append(locator)

    src_mob.descriptor = descriptor
    f.content.mobs.append(src_mob)

    # SourceMob'a timeline slot + filler segment ekle
    src_slot = src_mob.create_timeline_slot(edit_rate=edit_rate)
    src_filler = f.create.Filler("sound", int(edit_rate * 3600))
    src_slot.segment = src_filler

    # 2. MasterMob (DAW'ların klipleri bulduğu mob)
    master_mob = f.create.MasterMob()
    master_mob.name = file_name
    f.content.mobs.append(master_mob)

    # MasterMob slot'u → SourceMob'a referans veren SourceClip
    master_slot = master_mob.create_timeline_slot(edit_rate=edit_rate)
    # SourceClip: SourceMob'a referans ver
    src_clip = src_mob.create_source_clip(
        slot_id=src_slot.slot_id,
        start=0,
        length=int(edit_rate * 3600)
    )
    master_slot.segment = src_clip

    return master_mob, master_slot.slot_id


def create_external_aaf(manifest_path, output_aaf_path=None, verbose=False):
    with open(manifest_path) as fjson:
        data = json.load(fjson)

    output_dir = Path(manifest_path).parent
    if output_aaf_path:
        aaf_path = Path(output_aaf_path)
    else:
        aaf_path = output_dir / "sound_design.aaf"

    project_name = data.get("project_name", "sound_design")
    scenes = data.get("scenes", [])
    fps, _, _ = _resolve_rate(data.get("video_fps", 30.0))  # timeline edit rate
    LAYER_NAMES = ["Ch1", "Ch2", "Ch3"]
    num_alternatives = 3  # Fixed 3-layer system

    # Toplam süreyi hesapla (Timecode track uzunluğu için)
    total_duration = 0.0
    for scene in scenes:
        try:
            t_start = scene.get("timeline_start", "00:00:00.000")
            h, m, s = t_start.split(":")
            start_sec = int(h) * 3600 + int(m) * 60 + float(s)
            dur = float(scene.get("duration", 0))
            if start_sec + dur > total_duration:
                total_duration = start_sec + dur
        except Exception:
            pass
    
    # En az 1 saat veya video süresi + 5 dk (güvenli marj)
    total_duration = max(total_duration + 300, 3600.0)
    tc_length = int(total_duration * fps)

    if verbose:
        print(f"\n🎯 External-Linked AAF Writer")
        print(f"   Proje: {project_name}")
        print(f"   FPS (edit rate): {fps:.3f}")
        print(f"   Katman sayısı: 3 (Ambience | Support | Spot FX)")
        print(f"   Timecode length: {tc_length} frames ({total_duration/3600:.1f}h)")

    with aaf2.open(str(aaf_path), "w") as f:
        # top comp
        comp = f.create.CompositionMob(project_name)
        comp.usage = "Usage_TopLevel"
        f.content.mobs.append(comp)

        # timecode
        tc_slot = comp.create_empty_sequence_slot(fps, media_kind="timecode")
        
        # Timecode FPS'i tam sayı olmalı (struct hatasını önlemek için)
        # Edit rate (fps) rasyonel kalabilir ama Timecode display rate tam sayıya yuvarlanır
        fps_int = int(round(fps))
        
        tc = f.create.Timecode(fps_int)
        tc.length = tc_length
        tc.start = 0
        tc_slot.segment.components.append(tc)

        # FIXED 3-LAYER AUDIO TRACKS: Ambience / Support / Spot FX
        audio_tracks = []
        last_pos = {}

        for i in range(3):
            audio_track = comp.create_sound_slot(fps)
            audio_track.segment = f.create.Sequence(media_kind="sound")
            audio_track.name = LAYER_NAMES[i]
            audio_track["PhysicalTrackNumber"].value = i + 1
            audio_tracks.append(audio_track)
            last_pos[i + 1] = 0.0
        linked_count = 0

        for scene in scenes:
            timeline_start_str = scene.get("timeline_start", "00:00:00.000")

            # 🆕 FRAME-ACCURATE POSITIONING
            if scene.get('start_frame') is not None:
                target_start_frame = int(scene.get('start_frame'))
                scene_length_frames = int(scene.get('end_frame', 0)) - target_start_frame
                if scene_length_frames <= 0:
                    scene_length_frames = int(float(scene.get("duration", 0.0)) * fps)
                scene_duration = float(scene_length_frames) / fps
            else:
                h, m, s = timeline_start_str.split(":")
                timeline_start_sec = int(h) * 3600 + int(m) * 60 + float(s)
                target_start_frame = int(round(timeline_start_sec * fps))
                scene_duration = float(scene.get("duration", 0.0))
                scene_length_frames = int(round(scene_duration * fps))

            # 🆕 3-LAYER STRUCTURE: Ambience / Support / Spot FX
            layers = scene.get("layers", [])

            for track_id in range(1, num_alternatives + 1):
                audio_track = audio_tracks[track_id - 1]
                layer_name = LAYER_NAMES[track_id - 1]
                # gap varsa filler
                if target_start_frame > last_pos[track_id]:
                    gap_frames = target_start_frame - last_pos[track_id]
                    if gap_frames > 0:
                        audio_track.segment.components.append(
                            f.create.Filler("sound", int(gap_frames))
                        )

                # bu track için layer bul
                layer = next(
                    (l for l in layers if int(l.get("track", 0)) == track_id),
                    None,
                )

                if not layer:
                    audio_track.segment.components.append(
                        f.create.Filler("sound", int(scene_length_frames))
                    )
                    last_pos[track_id] = target_start_frame + scene_length_frames
                    continue

                # media path'i doğrula ve absolute path'e çevir
                try:
                    media_path = Path(layer.get("source_file", "")).resolve(strict=True)
                except Exception:
                    media_path = Path(layer.get("source_file", ""))

                if not media_path.exists():
                    if verbose:
                        print(f"   ⚠️ Dosya bulunamadı: {media_path}")
                    audio_track.segment.components.append(
                        f.create.Filler("sound", int(scene_length_frames))
                    )
                    last_pos[track_id] = target_start_frame + scene_length_frames
                    continue

                source_start_frames = int(round(
                    float(layer.get("source_start_offset", 0.0)) * fps
                ))
                clip_length_frames = int(scene_length_frames)

                # önce create_ama_link deneriz (AMA metadata ile)
                try:
                    fmt_name = media_path.suffix.lower().lstrip('.') or 'wav'
                    sr = 48000
                    ch = 2
                    bd = 16
                    duration_sec = scene_duration
                    frames_count = int(duration_sec * sr) if sr > 0 else 0
                    try:
                        with wave.open(str(media_path), 'rb') as wf:
                            sr = int(wf.getframerate()) or sr
                            ch = int(wf.getnchannels()) or ch
                            try:
                                bd = int(wf.getsampwidth() * 8) or bd
                            except Exception:
                                pass
                            frames = int(wf.getnframes())
                            if sr > 0 and frames > 0:
                                duration_sec = frames / float(sr)
                                frames_count = frames
                    except Exception:
                        pass

                    ama_metadata = {
                        'format': {
                            'format_name': fmt_name,
                            'format_long_name': fmt_name.upper() + " Audio",
                        },
                        'streams': [
                            {
                                'codec_type': 'audio',
                                'codec_name': fmt_name,
                                'codec_long_name': fmt_name.upper() + " Codec",
                                'sample_rate': sr,
                                'channels': ch,
                                'bit_depth': bd,
                                'duration_ts': frames_count,
                            }
                        ],
                        'duration': duration_sec,
                        'duration_ts': frames_count,
                    }

                    # ✅ create_ama_link sonucunu güvenli al
                    ama_result = f.content.create_ama_link(
                        str(media_path.resolve()), ama_metadata
                    )
                    
                    if ama_result is None:
                        raise ValueError("AMA link created None")
                        
                    master_mob, file_source_mob, tape_mob = ama_result

                    slot_id = None
                    for s in master_mob.slots:
                        mk = getattr(s.segment, "media_kind", None) or "sound"
                        if mk == "sound":
                            slot_id = s.slot_id
                            break
                    if slot_id is None:
                        slot_id = master_mob.slots[0].slot_id

                    final_clip = master_mob.create_source_clip(
                        slot_id=slot_id,
                        start=source_start_frames,
                        length=clip_length_frames,
                    )
                    audio_track.segment.components.append(final_clip)

                except (AttributeError, TypeError, ValueError, Exception) as ama_err:
                    # AMA başarısızsa manuel SourceMob oluştur
                    src_mob, slot_id = _make_external_source_mob(
                        f, str(media_path), edit_rate=fps
                    )

                    final_clip = src_mob.create_source_clip(
                        slot_id=slot_id,
                        start=source_start_frames,
                        length=clip_length_frames,
                    )
                    audio_track.segment.components.append(final_clip)

                last_pos[track_id] = target_start_frame + clip_length_frames
                linked_count += 1

                if verbose:
                    print(
                        f"   ✓ {scene.get('scene_id')} [{layer_name}]: {media_path.name} ({scene_length_frames} frames) external"
                    )

    if verbose:
        print(f"\n✅ External-Linked AAF: {aaf_path.name}")
        print(f"   {linked_count} ses dosyası referanslandı (embed edilmedi)")
        print("   ⚠️ DAW aynı path'te dosyaları bulmalı")

    return str(aaf_path)


# ───────────────────────────────────────────────
# EMBEDDED AAF + 48kHz PARALLEL CONVERSION
# ───────────────────────────────────────────────

def _convert_single_file(args):
    """Pickle-friendly ffmpeg worker."""
    src_path, dst_path, sample_rate = args
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src_path),
            "-ar", str(sample_rate),
            "-acodec", "pcm_s24le",
            str(dst_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return (str(src_path), str(dst_path), None)
    except Exception as e:
        return (str(src_path), None, str(e))


def _parallel_convert(unique_files, temp_dir, max_workers=8, sample_rate=48000, verbose=False):
    """Benzersiz ses dosyalarını paralel olarak 48kHz/24bit PCM WAV'e çevirir."""
    tasks = []
    src_to_dst = {}
    for src in unique_files:
        src_p = Path(src)
        dst = temp_dir / f"{src_p.stem}_48k.wav"
        src_to_dst[str(src)] = str(dst)
        tasks.append((str(src), str(dst), sample_rate))

    if verbose:
        print(f"\n🔄 Ses dosyaları {sample_rate}Hz'e çevriliyor ({len(tasks)} dosya, {max_workers} çekirdek)...")

    completed = 0
    failed = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_convert_single_file, t): t for t in tasks}
        for future in as_completed(futures):
            src, dst, err = future.result()
            if err:
                failed.append((src, err))
                if verbose:
                    print(f"   ❌ Dönüştürme hatası: {Path(src).name} — {err}")
            else:
                completed += 1
                if verbose:
                    print(f"   ✓ {Path(src).name} → 96kHz")

    if verbose:
        print(f"   {completed}/{len(tasks)} dosya başarıyla dönüştürüldü.")
    return src_to_dst, failed


def create_embedded_aaf(manifest_path, output_aaf_path=None, verbose=False, max_workers=8, sample_rate=48000):
    """
    Manifest'teki tüm ses dosyalarını 96kHz'e çevirip AAF içine gömer.
    Timeline'da source_start_offset ile alt-klip referansı korunur.
    """
    with open(manifest_path) as fjson:
        data = json.load(fjson)

    output_dir = Path(manifest_path).parent
    if output_aaf_path:
        aaf_path = Path(output_aaf_path)
    else:
        aaf_path = output_dir / "sound_design.aaf"

    project_name = data.get("project_name", "sound_design")
    scenes = data.get("scenes", [])
    fps, _, _ = _resolve_rate(data.get("video_fps", 30.0))
    LAYER_NAMES = ["Ch1", "Ch2", "Ch3"]
    num_alternatives = 3

    # Toplam süreyi hesapla (timecode track uzunluğu için)
    total_duration = 0.0
    for scene in scenes:
        try:
            t_start = scene.get("timeline_start", "00:00:00.000")
            h, m, s = t_start.split(":")
            start_sec = int(h) * 3600 + int(m) * 60 + float(s)
            dur = float(scene.get("duration", 0))
            if start_sec + dur > total_duration:
                total_duration = start_sec + dur
        except Exception:
            pass
    total_duration = max(total_duration + 300, 3600.0)
    tc_length = int(total_duration * fps)

    # ── 1. Tüm benzersiz ses dosyalarını topla ──
    unique_sources = set()
    for scene in scenes:
        for layer in scene.get("layers", []):
            sf = layer.get("source_file", "")
            if sf:
                try:
                    unique_sources.add(str(Path(sf).resolve(strict=True)))
                except Exception:
                    unique_sources.add(str(Path(sf)))

    # ── 2. Paralel 96kHz dönüşüm ──
    temp_dir = output_dir / "_aaf_embed_96k"
    temp_dir.mkdir(exist_ok=True)
    src_to_dst, failed_conversions = _parallel_convert(
        list(unique_sources), temp_dir, max_workers=max_workers, sample_rate=sample_rate, verbose=verbose
    )
    failed_set = {src for src, _ in failed_conversions}

    if verbose:
        print(f"\n🎯 Embedded AAF Writer")
        print(f"   Proje: {project_name}")
        print(f"   FPS (edit rate): {fps:.3f}")
        print(f"   Katman sayısı: 3 (Ambience | Support | Spot FX)")
        print(f"   Timecode length: {tc_length} frames ({total_duration/3600:.1f}h)")

    # ── 3. AAF oluştur ve embed et ──
    with aaf2.open(str(aaf_path), "w") as f:
        comp = f.create.CompositionMob(project_name)
        comp.usage = "Usage_TopLevel"
        f.content.mobs.append(comp)

        # timecode
        tc_slot = comp.create_empty_sequence_slot(fps, media_kind="timecode")
        fps_int = int(round(fps))
        tc = f.create.Timecode(fps_int)
        tc.length = tc_length
        tc.start = 0
        tc_slot.segment.components.append(tc)

        # 3-layer audio tracks
        audio_tracks = []
        last_pos = {}
        for i in range(3):
            audio_track = comp.create_sound_slot(fps)
            audio_track.segment = f.create.Sequence(media_kind="sound")
            audio_track.name = LAYER_NAMES[i]
            audio_track["PhysicalTrackNumber"].value = i + 1
            audio_tracks.append(audio_track)
            last_pos[i + 1] = 0.0

        # Cache: converted_path -> (master_mob, slot_id)
        embed_cache = {}
        embedded_count = 0

        for scene in scenes:
            timeline_start_str = scene.get("timeline_start", "00:00:00.000")

            # Frame-accurate positioning
            if scene.get("start_frame") is not None:
                target_start_frame = int(scene.get("start_frame"))
                scene_length_frames = int(scene.get("end_frame", 0)) - target_start_frame
                if scene_length_frames <= 0:
                    scene_length_frames = int(float(scene.get("duration", 0.0)) * fps)
                scene_duration = float(scene_length_frames) / fps
            else:
                h, m, s = timeline_start_str.split(":")
                timeline_start_sec = int(h) * 3600 + int(m) * 60 + float(s)
                target_start_frame = int(round(timeline_start_sec * fps))
                scene_duration = float(scene.get("duration", 0.0))
                scene_length_frames = int(round(scene_duration * fps))

            layers = scene.get("layers", [])

            for track_id in range(1, num_alternatives + 1):
                audio_track = audio_tracks[track_id - 1]
                layer_name = LAYER_NAMES[track_id - 1]

                # gap
                if target_start_frame > last_pos[track_id]:
                    gap_frames = target_start_frame - last_pos[track_id]
                    if gap_frames > 0:
                        audio_track.segment.components.append(
                            f.create.Filler("sound", int(gap_frames))
                        )

                layer = next(
                    (l for l in layers if int(l.get("track", 0)) == track_id),
                    None,
                )

                if not layer:
                    audio_track.segment.components.append(
                        f.create.Filler("sound", int(scene_length_frames))
                    )
                    last_pos[track_id] = target_start_frame + scene_length_frames
                    continue

                # media path doğrula
                try:
                    media_path = Path(layer.get("source_file", "")).resolve(strict=True)
                except Exception:
                    media_path = Path(layer.get("source_file", ""))

                converted_path_str = src_to_dst.get(str(media_path))
                if not converted_path_str or str(media_path) in failed_set:
                    if verbose:
                        print(f"   ⚠️ Dönüştürülemeyen dosya: {media_path.name}")
                    audio_track.segment.components.append(
                        f.create.Filler("sound", int(scene_length_frames))
                    )
                    last_pos[track_id] = target_start_frame + scene_length_frames
                    continue

                converted_path = Path(converted_path_str)
                if not converted_path.exists():
                    if verbose:
                        print(f"   ⚠️ Dönüştürülmüş dosya bulunamadı: {converted_path}")
                    audio_track.segment.components.append(
                        f.create.Filler("sound", int(scene_length_frames))
                    )
                    last_pos[track_id] = target_start_frame + scene_length_frames
                    continue

                source_start_frames = int(round(
                    float(layer.get("source_start_offset", 0.0)) * fps
                ))
                clip_length_frames = int(scene_length_frames)

                # ── Embed (cache'le) ──
                cache_key = str(converted_path)
                if cache_key not in embed_cache:
                    try:
                        master_mob = f.create.MasterMob(media_path.stem)
                        f.content.mobs.append(master_mob)

                        audio_slot = master_mob.import_audio_essence(
                            str(converted_path), edit_rate=fps
                        )
                        # slot_id bul
                        slot_id = None
                        for s in master_mob.slots:
                            mk = getattr(s.segment, "media_kind", None) or "sound"
                            if mk == "sound":
                                slot_id = s.slot_id
                                break
                        if slot_id is None and master_mob.slots:
                            slot_id = master_mob.slots[0].slot_id

                        embed_cache[cache_key] = (master_mob, slot_id)
                    except Exception as embed_err:
                        if verbose:
                            print(f"   ❌ Embed hatası: {media_path.name} — {embed_err}")
                        audio_track.segment.components.append(
                            f.create.Filler("sound", int(scene_length_frames))
                        )
                        last_pos[track_id] = target_start_frame + scene_length_frames
                        continue

                master_mob, slot_id = embed_cache[cache_key]

                final_clip = master_mob.create_source_clip(
                    slot_id=slot_id,
                    start=source_start_frames,
                    length=clip_length_frames,
                )
                audio_track.segment.components.append(final_clip)

                last_pos[track_id] = target_start_frame + clip_length_frames
                embedded_count += 1

                if verbose:
                    print(
                        f"   ✓ {scene.get('scene_id')} [{layer_name}]: {media_path.name} ({scene_length_frames} frames) embedded"
                    )

    # ── 4. Temizlik ──
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    if verbose:
        print(f"\n✅ Embedded AAF: {aaf_path.name}")
        print(f"   {embedded_count} ses dosyası AAF içine gömüldü (96kHz)")
        print("   ⚠️ AAF boyutu büyük olabilir, tüm kaynak dosyalar içeride")

    return str(aaf_path)
