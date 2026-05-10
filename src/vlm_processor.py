import os
import re
import json
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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "x-ai/grok-4.1-fast"

SIMPLE_DESIGNER_PROMPT = """You are a professional sound designer analyzing a frame.
Output ONLY a JSON object, no explanation, no markdown:

{
  "description": "describe the specific audible environment layers of atmosphere design, 10-15 words"
}

"""

def _image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
        "temperature": 0.5,
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

def analyze_scenes_batch(scene_frame_paths: list[dict], mode: str = "online", **kwargs) -> list[dict]:
    if mode == "online":
        return asyncio.run(_analyze_online_batch(scene_frame_paths))
    else:
        results = []
        for item in scene_frame_paths:
            results.append({**item, **_empty_response()})
        return results

# Yerel llama.cpp placeholder
def start_llama_server(*args, **kwargs): return None
def stop_llama_server(*args, **kwargs): pass
def query_llama_server(*args, **kwargs): return _empty_response()
def analyze_with_llamacpp_cli(*args, **kwargs): return _empty_response()
