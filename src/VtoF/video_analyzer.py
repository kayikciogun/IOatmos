"""
Modül 2: Video Analyzer
PySceneDetect ile video sahne geçişlerini tespit eder.
AdaptiveDetector kullanılarak optimize edilmiş (F1: %91.59 - En iyi performans).
Benchmark: https://github.com/Breakthrough/PySceneDetect/tree/main/benchmark
"""

import json
import logging
import cv2
import numpy as np
from pathlib import Path
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ThresholdDetector, ContentDetector

# FFmpeg/H.264 codec uyarılarını sustur (mmco: unref short failure vs.)
import os
import sys
import warnings

# ✅ CRITICAL: Environment variables - TÜM IMPORT'LARDAN ÖNCE!
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# OpenCV log seviyesi
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

# Python warnings sustur
warnings.filterwarnings('ignore')

# Logger setup
logger = logging.getLogger(__name__)


# Context manager: FFmpeg stderr susturma (mmco uyarıları için)
class SuppressFFmpegOutput:
    """FFmpeg/OpenCV stderr çıktılarını geçici susturur"""
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.old_stderr


def timecode_to_seconds(timecode_str):
    """HH:MM:SS.mmm formatını saniyeye çevir"""
    parts = timecode_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def analyze_video_content_advanced(video_path, sample_frames=30):
    """
    Video içeriğini analiz edip optimal threshold öner (BANTLI VERSİYON).
    Film/vlog/screen-record farkına duyarlı, std'ye de bakar.
    
    NOT: Şu an kullanılmıyor (AdaptiveDetector otomatik ayarlıyor).
    Gelecekte manual threshold için kullanılabilir.
    
    Args:
        video_path: Video dosya yolu
        sample_frames: Analiz için kaç frame örneklenecek
        
    Returns:
        Önerilen threshold değeri
    """
    cap = None
    try:
        # FFmpeg stderr sustur
        with SuppressFFmpegOutput():
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Eşit aralıklarla frame'ler al
            frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
            
            prev_frame = None
            frame_diffs = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and prev_frame is not None:
                    # Frame'ler arası fark hesapla
                    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Piksel farkının ortalaması
                    diff = cv2.absdiff(gray1, gray2)
                    mean_diff = np.mean(diff)
                    frame_diffs.append(mean_diff)
                
                prev_frame = frame.copy()
    finally:
        if cap is not None:
            cap.release()
    
    if not frame_diffs:
        return 12.0  # Default
    
    # İstatistiksel analiz (BANTLI - std'ye de bakar)
    mean_change = np.mean(frame_diffs)
    std_change = np.std(frame_diffs)
    
    # Video tipine göre threshold (akıllı bantlar)
    if mean_change < 5 and std_change < 3:
        # Çok statik: tripod + konuşma / sunum / screen record
        recommended_threshold = 16.0
        video_type = "Statik (tripod/sunum)"
    elif mean_change > 25 or std_change > 10:
        # Çok dinamik: klip, elde çekim, aksiyon
        recommended_threshold = 10.0
        video_type = "Dinamik (klip/aksiyon)"
    else:
        # Normal: film, vlog, genel içerik
        recommended_threshold = 12.0
        video_type = "Normal (film/vlog)"
    
    logger.info(f"Video analizi: {video_type} (mean: {mean_change:.1f}, std: {std_change:.1f})")
    logger.info(f"Önerilen threshold: {recommended_threshold}")
    
    return recommended_threshold


def detect_blank_scenes(video_path, scenes, black_threshold=25, white_threshold=225, sample_points=5):
    """
    Boş (tamamen siyah veya beyaz) sahneleri tespit eder ve işaretler.
    Hız için kareler 256px'e küçültülerek analiz edilir.
    """
    cap = None
    try:
        with SuppressFFmpegOutput():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            blank_info = {}
            
            for scene in scenes:
                scene_id = scene['scene_id']
                start_time = timecode_to_seconds(scene['start'])
                end_time = timecode_to_seconds(scene['end'])
                duration = end_time - start_time
                
                # Örnekleme noktalarını hesapla
                sample_times = []
                if duration < 0.5:
                    sample_times = [(start_time + end_time) / 2]
                else:
                    for i in range(sample_points):
                        ratio = i / (sample_points - 1) if sample_points > 1 else 0.5
                        sample_times.append(start_time + (duration * ratio))
                
                blank_frames = 0
                black_frames = 0
                white_frames = 0
                brightness_values = []
                
                # Performans için son hesaplanan değerleri sakla (tekrar kullanma için)
                edge_density = 0.0
                lap_var = 0.0
                color_std = 0.0
                p_white = 0.0
                p_black = 0.0
                uniform_ratio = 0.0

                for sample_time in sample_times:
                    sample_frame_idx = int(sample_time * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # 🚀 OPTİMİZASYON: Analiz için kareyi küçült (256px max)
                        # Bu, median, mean ve edge detection işlemlerini inanılmaz hızlandırır.
                        h, w = frame.shape[:2]
                        target_size = 256
                        scale = target_size / max(h, w)
                        if scale < 1.0:
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        mean_brightness = np.mean(gray)
                        std_brightness = np.std(gray)
                        brightness_values.append(mean_brightness)

                        # Renk kanalları üzerinden uniformluk
                        channel_std = np.std(frame, axis=(0, 1))
                        color_std = float(np.mean(channel_std))

                        # Kenar yoğunluğu ve Laplacian
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
                        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                        # Histogram ve uniformluk
                        p_black = float(np.mean(gray < black_threshold))
                        p_white = float(np.mean(gray > white_threshold))
                        
                        # np.median ağır bir işlemdir, küçük karede bile dikkatli kullanılmalı
                        median_val = float(np.median(gray))
                        uniform_ratio = float(np.mean(np.abs(gray.astype(np.float32) - median_val) < 4.0))
                        
                        is_black = (p_black > 0.98) or (
                            (mean_brightness < black_threshold) and (std_brightness < 10 and edge_density < 0.005)
                        )
                        is_white = (p_white > 0.98) or (
                            (mean_brightness > white_threshold) and (std_brightness < 10 and edge_density < 0.005)
                        )
                        is_color_blank = (not (is_black or is_white)) and (
                            (uniform_ratio > 0.97) and 
                            (color_std < 10.0) and 
                            (edge_density < 0.004) and 
                            (lap_var < 18.0)
                        )
                        
                        if is_black:
                            black_frames += 1
                            blank_frames += 1
                        elif is_white:
                            white_frames += 1
                            blank_frames += 1
                        elif is_color_blank:
                            blank_frames += 1
            
                # Sahne blank mı? (frame'lerin %50+ blank ise)
                total_samples = len(sample_times)
                blank_ratio = blank_frames / total_samples if total_samples > 0 else 0

                # Daha konservatif karar: en az %60 blank
                is_blank = blank_ratio >= 0.6
                is_black = black_frames > white_frames and is_blank  # Çoğunluk siyah
                is_white = white_frames > black_frames and is_blank  # Çoğunluk beyaz
                # Renkli boş sahne: siyah/beyaz değilse ve sahne boşsa
                is_color = (not is_black and not is_white) and is_blank

                avg_brightness = np.mean(brightness_values) if brightness_values else 0

                blank_info[scene_id] = {
                    'is_blank': bool(is_blank),
                    'is_black': bool(is_black),
                    'is_white': bool(is_white),
                    'is_color': bool(is_color),
                    'brightness': float(avg_brightness),
                    'blank_ratio': float(blank_ratio),
                    'samples_checked': total_samples,
                    'metrics': {
                        'mean_brightness': float(np.mean(brightness_values)) if brightness_values else 0.0,
                        'edge_density_last': float(edge_density) if 'edge_density' in locals() else 0.0,
                        'lap_var_last': float(lap_var) if 'lap_var' in locals() else 0.0,
                        'color_std_last': float(color_std) if 'color_std' in locals() else 0.0,
                        'p_white_last': float(p_white) if 'p_white' in locals() else 0.0,
                        'p_black_last': float(p_black) if 'p_black' in locals() else 0.0,
                        'uniform_ratio_last': float(uniform_ratio) if 'uniform_ratio' in locals() else 0.0
                    }
                }
        
        return blank_info
    finally:
        if cap is not None:
            cap.release()


def merge_short_scenes(scenes, min_duration=0.5, preserve_blank=True):
    """
    Çok kısa sahneleri birleştirir (yanlış tespitleri temizler).
    Blank sahneleri koruyabilir.
    """
    if not scenes:
        return scenes
    
    merged = []
    current_scene = scenes[0].copy()
    
    for next_scene in scenes[1:]:
        # Blank sahne kontrolü - eğer blank ise birleştirme!
        is_current_blank = current_scene.get('blank_info', {}).get('is_blank', False)
        
        should_merge = (
            current_scene['duration'] < min_duration and
            not (preserve_blank and is_current_blank)  # Blank sahneleri koru
        )
        
        if should_merge:
            # Çok kısa sahneyi bir sonrakiyle birleştir
            current_scene['end'] = next_scene['end']
            if 'end_frame' in next_scene:
                current_scene['end_frame'] = next_scene['end_frame']
            
            current_scene['duration'] = float(
                timecode_to_seconds(next_scene['end']) - 
                timecode_to_seconds(current_scene['start'])
            )
        else:
            merged.append(current_scene)
            current_scene = next_scene.copy()
    
    # Son sahneyi ekle
    merged.append(current_scene)
    
    # Scene ID'leri yeniden düzenle
    for i, scene in enumerate(merged, 1):
        scene['scene_id'] = i
    
    return merged


def compare_scene_similarity(video_path, scene1, scene2, sample_frames=3):
    """
    İki sahneyi karşılaştırıp benzerlik skoru döndürür.
    
    Args:
        video_path: Video dosya yolu
        scene1, scene2: Sahne dict'leri
        sample_frames: Her sahneden kaç frame örneklenecek
        
    Returns:
        Benzerlik skoru (0.0-1.0)
    """
    try:
        # FFmpeg stderr sustur
        with SuppressFFmpegOutput():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Her sahneden sample frame'ler al
            def get_scene_frames(scene):
                start_sec = timecode_to_seconds(scene['start'])
                end_sec = timecode_to_seconds(scene['end'])
                duration = end_sec - start_sec
                
                # Eşit aralıklarla frame'ler
                times = np.linspace(start_sec, end_sec, min(sample_frames, int(duration * fps)))
                frames = []
                
                for t in times:
                    frame_idx = int(t * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # Küçült (hız için)
                        frame = cv2.resize(frame, (64, 64))
                        frames.append(frame)
                
                return frames
            
            frames1 = get_scene_frames(scene1)
            frames2 = get_scene_frames(scene2)
            
            cap.release()
        
        if not frames1 or not frames2:
            return 0.0
        
        # Frame'leri karşılaştır (histogram similarity)
        similarities = []
        for f1 in frames1:
            for f2 in frames2:
                # RGB histogram
                hist1 = cv2.calcHist([f1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([f2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                
                # Normalize
                cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                # Correlation
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                similarities.append(similarity)
        
        # Ortalama benzerlik
        return float(np.mean(similarities))
        
    except Exception as e:
        logger.error(f"Sahne benzerlik hatası: {e}")
        return 0.0


def merge_similar_consecutive_scenes(video_path, scenes, similarity_threshold=0.85):
    """
    Ardışık benzer sahneleri birleştirir (yanlış bölünmeleri düzeltir).
    
    Args:
        video_path: Video dosya yolu
        scenes: Sahne listesi
        similarity_threshold: Benzerlik eşiği (0.85 = %85 benzer)
        
    Returns:
        Birleştirilmiş sahne listesi
    """
    if len(scenes) <= 1:
        return scenes
    
    print(f"   🔗 Benzer sahneler birleştiriliyor (threshold: {similarity_threshold})...")
    
    merged = []
    current_scene = scenes[0].copy()
    merge_count = 0
    
    for i, next_scene in enumerate(scenes[1:], 1):
        # Blank sahneleri birleştirme!
        is_current_blank = current_scene.get('blank_info', {}).get('is_blank', False)
        is_next_blank = next_scene.get('blank_info', {}).get('is_blank', False)
        
        if is_current_blank or is_next_blank:
            # Blank sahneleri koru
            merged.append(current_scene)
            current_scene = next_scene.copy()
            continue
        
        # Benzerlik hesapla
        similarity = compare_scene_similarity(video_path, current_scene, next_scene)
        
        if similarity >= similarity_threshold:
            # Benzer sahneleri birleştir
            print(f"      🔗 Sahne {current_scene['scene_id']} + {next_scene['scene_id']} birleştirildi (benzerlik: {similarity:.2f})")
            current_scene['end'] = next_scene['end']
            if 'end_frame' in next_scene:
                current_scene['end_frame'] = next_scene['end_frame']
            
            current_scene['duration'] = float(
                timecode_to_seconds(next_scene['end']) - 
                timecode_to_seconds(current_scene['start'])
            )
            merge_count += 1
        else:
            # Farklı sahneler - ayır
            merged.append(current_scene)
            current_scene = next_scene.copy()
    
    # Son sahneyi ekle
    merged.append(current_scene)
    
    # Scene ID'leri yeniden düzenle
    for i, scene in enumerate(merged, 1):
        scene['scene_id'] = i
    
    if merge_count > 0:
        print(f"   ✅ {merge_count} sahne birleştirmesi yapıldı")
    
    return merged


def detect_scenes(video_path, 
                 mode="adaptive",
                 adaptive_threshold=3.0, 
                 min_content_val=15.0, 
                 output_path="data/scenes.json", 
                 min_scene_length=0.3, 
                 merge_similar=False, 
                 similarity_threshold=0.85,
                 threshold=None,
                 auto_optimize=None):
    """
    PySceneDetect ile sahne geçişlerini tespit eder.
    
    Args:
        video_path: Video dosya yolu
        mode: Tespit modu ("adaptive", "static", "dynamic")
            - "adaptive": Film/Müzik Klibi (Standart) - AdaptiveDetector (F1: %91.59)
            - "static": Konuşma/Sunum (Sabit Kamera) - ContentDetector + yüksek threshold
            - "dynamic": El Kamerası/Aksiyon - Adaptive + Threshold Hybrid + merge
        adaptive_threshold: Adaptive mod için minimum threshold (default: 3.0)
        min_content_val: Adaptive mod için maksimum threshold (default: 15.0)
        output_path: Çıktı JSON dosyası
        min_scene_length: Minimum sahne süresi (saniye)
        merge_similar: Benzer ardışık sahneleri birleştir
        similarity_threshold: Benzerlik eşiği (0.85 = %85 benzer)
        
    Returns:
        Tespit edilen sahne sayısı
    """
    # Mode'a göre preset ayarları
    mode_presets = {
        "adaptive": {
            "name": "🧱 Film / Müzik Klibi (Standart)",
            "description": "AdaptiveDetector (F1: %91.59)",
            "use_adaptive": True,
            "use_content": False,
            "threshold": 12.0,
            "merge_similar": False
        },
        "static": {
            "name": "📹 Konuşma / Sunum (Sabit Kamera)",
            "description": "ContentDetector + Threshold (yüksek eşik)",
            "use_adaptive": False,
            "use_content": True,
            "threshold": 16.0,
            "merge_similar": False
        },
        "dynamic": {
            "name": "🎥 El Kamerası / Aksiyon",
            "description": "Adaptive + Threshold Hybrid + merge",
            "use_adaptive": True,
            "use_content": False,
            "threshold": 12.0,
            "merge_similar": True
        }
    }
    
    # Mode'u uygula
    preset = mode_presets.get(mode, mode_presets["adaptive"])
    
    # Dışarıdan gelen threshold override (varsa)
    if threshold is not None:
        preset["threshold"] = float(threshold)
    
    # Dışarıdan gelen auto_optimize bayrağına göre adaptif/detector seçimi
    if auto_optimize is not None:
        if bool(auto_optimize):
            preset["use_adaptive"] = True
            # Adaptive seçiliyken content'i devre dışı bırak
            preset["use_content"] = False
        else:
            # Otomatik optimizasyon kapalı ise content + threshold yaklaşımı
            preset["use_adaptive"] = False
            preset["use_content"] = True
    if preset["merge_similar"]:
        merge_similar = True
    
    print(f"🔎 Video analiz ediliyor: {video_path}")
    print(f"   🎬 Mod: {preset['name']}")
    print(f"   📊 {preset['description']}")
    
    # Output dizinini oluştur
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # PySceneDetect v0.7+: VideoManager kaldırıldı; open_video + SceneManager kullanılır.
    video = open_video(video_path)
    detected_fps = float(video.frame_rate)

    video_metadata = {}
    downscale = 2

    try:
        video_width, video_height = video.frame_size
        video_fps = float(video.frame_rate)
        total_frames = (
            int(video.duration.frame_num) if video.duration is not None else 0
        )

        video_metadata = {
            "video_path": str(video_path),
            "fps": float(video_fps),
            "width": video_width,
            "height": video_height,
            "total_frames": total_frames,
            "scenedetect_fps": float(detected_fps),
        }

        # Çözünürlüğe göre downscale
        if video_width >= 3840:  # 4K
            downscale = 3
            print(f"   📐 4K video ({video_width}p) → downscale: 3x")
        elif video_width >= 1920:  # 1080p
            downscale = 2
            print(f"   📐 Full HD video ({video_width}p) → downscale: 2x")
        else:  # 720p ve altı
            downscale = 1
            print(f"   📐 HD/SD video ({video_width}p) → downscale: 1x (native)")

        # FPS'e göre min_scene_length ayarla (HER ZAMAN)
        target_min_frames = 12  # ~0.5 saniye (60fps'te 0.2s, 24fps'te 0.5s)
        min_scene_length_auto = target_min_frames / video_fps

        # Config'den gelen değeri kullan ama FPS'e göre ayarla
        min_scene_length = min_scene_length_auto
        print(
            f"   ⏱️ FPS: {video_fps:.2f} → min_scene_length: {min_scene_length:.2f}s "
            f"(auto: {target_min_frames} frames)"
        )

    except Exception as e:
        logger.warning(f"Video info alınamadı, default downscale: {e}")

    # Default min_scene_length (FPS bilgisi yoksa)
    if min_scene_length is None or min_scene_length == 0.3:
        min_scene_length = 0.5  # Default fallback

    scene_manager = SceneManager()
    scene_manager.auto_downscale = False
    scene_manager.downscale = int(downscale)
    
    # 🎯 Mode'a göre detector seçimi
    if preset["use_adaptive"]:
        # Adaptive mod: Film/Müzik Klibi veya Dinamik
        scene_manager.add_detector(AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_content_val=min_content_val,
            window_width=2,
            min_scene_len=15
        ))
        print(f"   🔍 AdaptiveDetector aktif (adaptive_threshold={adaptive_threshold})")
        
        # Dynamic modda Threshold da ekle (hibrit)
        if mode == "dynamic":
            scene_manager.add_detector(ThresholdDetector(
                threshold=preset["threshold"],
                fade_bias=0.0
            ))
            print(f"   🔍 + ThresholdDetector (fade fix)")
    
    elif preset["use_content"]:
        # Static mod: Konuşma/Sunum - ContentDetector + Threshold
        scene_manager.add_detector(ContentDetector(threshold=preset["threshold"]))
        scene_manager.add_detector(ThresholdDetector(
            threshold=preset["threshold"],
            fade_bias=0.0
        ))
        print(f"   🔍 ContentDetector + ThresholdDetector (threshold={preset['threshold']})")
    
    scene_manager.detect_scenes(video=video)
    
    # Sahne listesini al
    scene_list = scene_manager.get_scene_list()
    
    scenes = []
    for i, (start_time, end_time) in enumerate(scene_list, start=1):
        scene_data = {
            "scene_id": i,
            "start": format_timecode(start_time),
            "end": format_timecode(end_time),
            "start_frame": start_time.get_frames(),
            "end_frame": end_time.get_frames(),
            "duration": float((end_time - start_time).get_seconds())
        }
        scenes.append(scene_data)
    
    initial_count = len(scenes)
    print(f"   📍 İlk tespit: {initial_count} sahne")
    
    # ÖNEMLİ: Blank sahneleri ÖNCE tespit et, SONRA merge yap!
    print(f"   🎨 Boş sahneler tespit ediliyor (fade to black/white)...")
    blank_info = detect_blank_scenes(video_path, scenes)
    
    # Blank bilgisini sahnelere ekle (merge'den ÖNCE)
    blank_count_before_merge = 0
    for scene in scenes:
        scene_id = scene['scene_id']
        if scene_id in blank_info:
            scene['blank_info'] = blank_info[scene_id]
            if blank_info[scene_id]['is_blank']:
                blank_count_before_merge += 1
    
    if blank_count_before_merge > 0:
        print(f"   ⚫⚪ {blank_count_before_merge} boş sahne bulundu (merge öncesi)")
    
    # Kısa sahneleri birleştir (ama blank sahneleri koru!)
    if min_scene_length > 0 and len(scenes) > 1:
        scenes = merge_short_scenes(scenes, min_scene_length, preserve_blank=True)
        print(f"   🔀 Kısa sahneler birleştirildi: {initial_count} → {len(scenes)} sahne")
        print(f"   💡 Blank sahneler korundu (birleştirilmedi)")
    
    # Merge sonrası blank count'u tekrar say
    blank_count_final = sum(1 for s in scenes if s.get('blank_info', {}).get('is_blank', False))
    
    if blank_count_final > 0:
        print(f"   ⚫⚪ Son durum: {blank_count_final} boş sahne (merge sonrası)")
    
    # Benzer sahneleri birleştir (yanlış bölünmeleri düzelt)
    if merge_similar and len(scenes) > 1:
        before_merge_count = len(scenes)
        scenes = merge_similar_consecutive_scenes(video_path, scenes, similarity_threshold)
        if len(scenes) < before_merge_count:
            print(f"   🎯 Benzer sahne birleştirmesi: {before_merge_count} → {len(scenes)} sahne")
    
    # Sonuçları göster
    for scene in scenes:
        blank_marker = ""
        if scene.get('blank_info', {}).get('is_black'):
            blank_marker = " ⚫ [BLANK-BLACK]"
        elif scene.get('blank_info', {}).get('is_white'):
            blank_marker = " ⚪ [BLANK-WHITE]"
        elif scene.get('blank_info', {}).get('is_color'):
            blank_marker = " 🎨 [BLANK-COLOR]"

        print(f"   Sahne {scene['scene_id']}: {scene['start']} → {scene['end']} ({scene['duration']:.1f}s){blank_marker}")
        bi = scene.get('blank_info', {})
        # Debug: blank_ratio ve parlaklık ortalaması
        if bi:
            print(f"      ◽ Debug: blank_ratio={bi.get('blank_ratio', 0):.2f}, brightness={bi.get('brightness', 0):.1f}")
    
    # JSON'a kaydet (NumPy tipleri için güvenli)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False, default=str)
        
    # 🆕 Metadata kaydet (FPS doğruluğu için)
    if video_metadata:
        meta_path = Path(output_path).parent / "video_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(video_metadata, f, indent=2)
        # Log info
        fps_val = video_metadata.get('fps', 0)
        print(f"ℹ️ Video metadata kaydedildi: {meta_path} (FPS: {fps_val:.3f})")
    
    print(f"✅ {len(scenes)} sahne tespit edildi → {output_path}")
    return len(scenes)


def save_scene_frames(video_path, scenes, output_dir="data/temp/frames", max_dim=720, samples=("start","mid","end")):
    """
    Her sahne için örnek kareleri (start/mid/end) `output_dir` içine kaydeder.

    Args:
        video_path: Video dosya yolu
        scenes: detect_scenes tarafından üretilen sahne listesi
        output_dir: Çıktı klasörü (varsayılan: data/temp/frames)
        max_dim: Kaydedilen görseller için maksimum genişlik/yükseklik (px)
        samples: Kaydedilecek örnek noktalar ("start", "mid", "end")

    Returns:
        Kaydedilen dosya yollarının listesi
    """
    saved_paths = []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = None
    try:
        with SuppressFFmpegOutput():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for scene in scenes:
                scene_id = scene.get("scene_id")
                start_sec = timecode_to_seconds(scene.get("start"))
                end_sec = timecode_to_seconds(scene.get("end"))
                duration = max(0.0, end_sec - start_sec)

                # Örnek zaman noktaları
                sample_times = []
                epsilon = 0.01
                if "start" in samples:
                    sample_times.append(min(max(start_sec + epsilon, 0.0), end_sec))
                if "mid" in samples:
                    mid_t = start_sec + duration / 2.0
                    sample_times.append(min(max(mid_t, start_sec), end_sec))
                if "end" in samples:
                    sample_times.append(max(min(end_sec - epsilon, end_sec), start_sec))

                names = []
                if "start" in samples: names.append("start")
                if "mid" in samples: names.append("mid")
                if "end" in samples: names.append("end")

                for idx, sample_time in enumerate(sample_times):
                    frame_index = int(round(sample_time * fps))
                    frame_index = max(0, min(frame_index, max(0, total_frames - 1)))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    # Boyutlandırma
                    h, w = frame.shape[:2]
                    scale = 1.0
                    if max(h, w) > max_dim:
                        scale = max_dim / float(max(h, w))
                    if scale != 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Dosya ismi: Tek örnekse scene_01.jpg, çokluysa scene_01_mid.jpg
                    suffix = f"_{names[idx]}" if len(samples) > 1 else ""
                    filename = f"scene_{scene_id:02d}{suffix}.jpg"
                    out_file = out_path / filename
                    
                    cv2.imwrite(str(out_file), frame)
                    saved_paths.append(str(out_file))

    finally:
        if cap is not None:
            cap.release()

    return saved_paths


def format_timecode(frametime):
    """
    FrameTimecode'u HH:MM:SS.mmm formatına çevir.
    
    Args:
        frametime: PySceneDetect FrameTimecode objesi
        
    Returns:
        Formatlanmış timecode string
    """
    total_seconds = frametime.get_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


if __name__ == "__main__":
    # Test için
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        detect_scenes(video_path)
    else:
        print("Kullanım: python video_analyzer.py <video_dosya_yolu>")
