import os
import re
import json
import time
import subprocess
import urllib.request
import base64
import asyncio
import aiohttp
from pathlib import Path

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
#   Online (OpenRouter) Ayarları
# ============================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "x-ai/grok-4.1-fast"

# ============================================================
#   Local (llama.cpp Server) Ayarları
# ============================================================

LLAMA_SERVER_PORT = 8787
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"

# ============================================================
#   Prompt
# ============================================================

SIMPLE_DESIGNER_PROMPT = """You are a professional sound designer analyzing a frame.
Output ONLY a JSON object, no explanation, no markdown:

{
  "description": "describe environment you see in the frame, 10-15 words"
}

"""

# ============================================================
#   Yanıt Parser
# ============================================================

def _empty_response():
    return {"sound_description": ""}

def _parse_json_response(raw: str) -> dict:
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "sound_description": str(data.get("description", "")).strip(),
            }
    except: pass
    return _empty_response()

def parse_vlm_response(raw_text: str) -> dict:
    """VLM yanıtını parse eder. CATEGORY alanı artık kullanılmıyor."""
    if not raw_text or not raw_text.strip():
        return _empty_response()

    # Temizlik (Resim tagleri, Düşünce tagleri ve Markdown temizliği)
    raw_text = re.sub(r'\[img-\d+\]', '', raw_text)
    raw_text = re.sub(r'<思考>.*?</思考>', '', raw_text, flags=re.DOTALL)
    raw_text = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL)
    raw_text = raw_text.replace('**', '').replace('*', '')

    # Önce JSON formatı dene
    json_result = _parse_json_response(raw_text)
    if json_result.get("sound_description"):
        return json_result

    # Fallback: Serbest metin, description aramaya çalış
    description = ""
    desc_match = re.search(r'DESCRIPTION:\s*(.*)', raw_text, re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()

    if not description:
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        for line in lines:
            if "CATEGORY:" in line.upper(): continue
            if "DESCRIPTION:" in line.upper(): continue
            if not any(skip in line.lower() for skip in ["thinking", "analyze", "{", "}"]):
                description = line.strip()
                break

    return {"sound_description": description}

# ============================================================
#   Online (OpenRouter) Batch
# ============================================================

def _image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def _call_openrouter_async(session, image_path: str, retries: int = 3) -> dict:
    if not OPENROUTER_API_KEY:
        print("   ❌ HATA: OPENROUTER_API_KEY bulunamadı!")
        return _empty_response()

    img_b64 = _image_to_b64(image_path)
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"

    payload = {
        "model": MODEL,
        "reasoning": {"effort": "medium"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": SIMPLE_DESIGNER_PROMPT}
            ]
        }],
        "temperature": 0.9,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "IOatmos",
    }

    for attempt in range(retries):
        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60) as resp:
                if resp.status == 401:
                    print(f"   ❌ YETKİ HATASI (401): API anahtarı geçersiz")
                    return _empty_response()
                if resp.status != 200:
                    err_text = await resp.text()
                    print(f"   ⚠️ API Hatası ({resp.status}): {err_text[:200]}")
                    continue

                data = await resp.json()
                if "choices" not in data:
                    print(f"   ⚠️ Yanıt formatı hatalı")
                    continue

                raw = data["choices"][0]["message"]["content"].strip()
                return _parse_json_response(raw)
        except Exception as e:
            if attempt == retries - 1:
                print(f"   ❌ Sahne {image_path} hatası: {str(e)}")
            await asyncio.sleep(2)

    return _empty_response()

async def _analyze_online_batch(scene_frame_paths: list[dict], max_workers: int = 16) -> list[dict]:
    if OPENROUTER_API_KEY:
        masked = OPENROUTER_API_KEY[:10] + "..." + OPENROUTER_API_KEY[-4:]
        print(f"   🔑 API Anahtarı aktif: {masked}")

    connector = aiohttp.TCPConnector(limit=max_workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_call_openrouter_async(session, item["frame_path"]) for item in scene_frame_paths]
        print(f"   ⏳ {len(tasks)} istek gönderildi...")
        responses = await asyncio.gather(*tasks)

        results = []
        for item, parsed in zip(scene_frame_paths, responses):
            results.append({**item, **parsed})
            desc = parsed.get("sound_description", "")
            if desc:
                print(f"   ✅ Scene {item['scene_id']:02d} {desc[:50]}...")
            else:
                print(f"   ⚠️ Scene {item['scene_id']:02d} BOŞ YANIT")

        return results

# ============================================================
#   Local (llama.cpp Server)
# ============================================================

def start_llama_server(base_model, mmproj):
    """llama-server'ı arka planda başlatır."""
    server_path = "llama.cpp/build/bin/llama-server"
    if not os.path.exists(server_path):
        print("   ⚠️ llama-server bulunamadı!")
        return None

    # Port temizliği
    try:
        subprocess.run(f"lsof -ti:{LLAMA_SERVER_PORT} | xargs kill -9", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(1)
    except: pass

    cmd = [
        f"./{server_path}", "-m", base_model, "--mmproj", mmproj,
        "--port", str(LLAMA_SERVER_PORT), "-c", "4096", "-ngl", "99",
        "-fa", "on", "-t", "8", "--temp", "0.2", "--log-disable"
    ]

    print(f"   ⏳ VLM Sunucusu başlatılıyor (port {LLAMA_SERVER_PORT})...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    max_wait = 120
    for i in range(max_wait):
        try:
            req = urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if json.loads(req.read().decode()).get("status") == "ok":
                print(f"   ✅ VLM Sunucusu hazır ({i+1}sn)")
                return proc
        except: pass
        if proc.poll() is not None: break
        time.sleep(1)

    print("   ❌ VLM Sunucusu başlatılamadı!")
    return None

def query_llama_server(image_path, retries=3):
    """Görüntüyü sunucuya gönderir."""
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": SIMPLE_DESIGNER_PROMPT}
            ]}],
            "temperature": 0.3, "max_tokens": 1000,
        }).encode("utf-8")

        req = urllib.request.Request(f"{LLAMA_SERVER_URL}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})

        for _ in range(retries):
            try:
                resp = urllib.request.urlopen(req, timeout=90)
                msg = json.loads(resp.read().decode())["choices"][0]["message"]
                raw = msg.get("content", "").strip() or msg.get("reasoning_content", "").strip()
                return parse_vlm_response(raw)
            except: time.sleep(2)
    except Exception as e:
        print(f"   ⚠️ VLM Sorgu hatası: {e}")
    return parse_vlm_response("")

def stop_llama_server(proc):
    """Sunucuyu durdurur."""
    if not proc: return
    try:
        os.killpg(os.getpgid(proc.pid), 15)
        proc.wait(timeout=5)
    except:
        try: proc.kill()
        except: pass
    print("   🛑 VLM Sunucusu durduruldu")

def analyze_with_llamacpp_cli(image_path, base_model, mmproj):
    """CLI Fallback (Yavaş mod)."""
    cli_path = "llama.cpp/build/bin/llama-cli"
    cmd = [f"./{cli_path}", "-m", base_model, "--mmproj", mmproj, "--image", image_path, "-p", SIMPLE_DESIGNER_PROMPT, "-n", "150", "--temp", "0.2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_vlm_response(result.stdout)

def _analyze_local_batch(scene_frame_paths: list[dict], base_model: str, mmproj: str, max_workers: int = 8) -> list[dict]:
    """Local VLM ile batch sahne analizi (llama-server)."""
    proc = start_llama_server(base_model, mmproj)
    if not proc:
        print("   ⚠️ Local sunucu başlatılamadı, CLI fallback kullanılıyor...")
        results = []
        for item in scene_frame_paths:
            res = analyze_with_llamacpp_cli(item["frame_path"], base_model, mmproj)
            results.append({**item, **res})
            desc = res.get("sound_description", "")
            if desc:
                print(f"   ✅ Scene {item['scene_id']:02d} {desc[:500]}...")
            else:
                print(f"   ⚠️ Scene {item['scene_id']:02d} BOŞ YANIT")
        return results

    try:
        results = []
        for item in scene_frame_paths:
            res = query_llama_server(item["frame_path"])
            results.append({**item, **res})
            desc = res.get("sound_description", "")
            if desc:
                print(f"   ✅ Scene {item['scene_id']:02d} {desc[:50]}...")
            else:
                print(f"   ⚠️ Scene {item['scene_id']:02d} BOŞ YANIT")
        return results
    finally:
        stop_llama_server(proc)

# ============================================================
#   Ana API
# ============================================================

def analyze_scenes_batch(scene_frame_paths: list[dict], mode: str = "online", **kwargs) -> list[dict]:
    if mode == "online":
        return asyncio.run(_analyze_online_batch(scene_frame_paths))
    elif mode == "local":
        base_model = kwargs.get("base_model")
        mmproj = kwargs.get("mmproj")
        if not base_model or not mmproj:
            print("   ❌ Local mod için base_model ve mmproj gerekli!")
            results = []
            for item in scene_frame_paths:
                results.append({**item, **_empty_response()})
            return results
        return _analyze_local_batch(scene_frame_paths, base_model, mmproj)
    else:
        results = []
        for item in scene_frame_paths:
            results.append({**item, **_empty_response()})
        return results
