import os
import glob
import json
import time
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import google.generativeai as genai
import ast
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set your Google AI Studio API key here or via environment variable GEMINI_API_KEY
# Get a free key at: https://aistudio.google.com/app/apikey
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

BASE_DIR = os.environ.get("LPCVC_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "train2014", "train2014")

FINAL_OUTPUT = os.path.join(BASE_DIR, "data", "captions_aistudio.json")

# ==============================================================================
# PROMPT LOGIC
# ==============================================================================
prompt = """Analyze this image in detail.
Provide EXACTLY 4 distinct descriptive phrases.

1. Two short phrases (1-3 words) for primary objects/colors (e.g., 'red car', 'wooden table').
2. Two highly detailed relational phrases (5-10 words) describing specific attributes, patterns, actions, or part-to-whole relationships (e.g., 'rabbit with white and brown fur', 'skateboard with a yellow lightning bolt in the center', 'person feeding a fish to a penguin').

Format: strictly a Python list of strings."""

# 讀取已經存在的兩個檔案，避免重複標註
existing_data = {}
EXISTING_LABELS = [
    os.path.join(BASE_DIR, "vllm_captions", "captions_qwen25_7b.json"),
    os.path.join(BASE_DIR, "vllm_captions", "captions_full_gemini.json"),
    FINAL_OUTPUT
]

for path in EXISTING_LABELS:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            existing_data.update(json.load(f))

# ==============================================================================
# WORKER
# ==============================================================================
def label_image(img_path):
    while True:
        try:
            pil_img = Image.open(img_path)
            
            # AI Studio SDK 支援設定 JSON 格式
            response = model.generate_content(
                [prompt, pil_img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                )
            )
            
            # 處理被 AI Studio 安全過濾器擋掉的情況
            if response.candidates and "SAFETY" in str(response.candidates[0].finish_reason):
                return img_path, ["object not clear", "safety blocked"]
            
            if response.text is None:
                raise Exception("DEBUG CRASH: response.text is None!")
                
            text = response.text
            match = re.search(r'\[(.*?)\]', text, re.DOTALL)
            if match:
                list_str = "[" + match.group(1) + "]"
                try:
                    phrases = ast.literal_eval(list_str)
                    if isinstance(phrases, list) and len(phrases) > 0:
                        clean = [str(p).strip() for p in phrases if str(p).strip()]
                        return img_path, clean
                except Exception as parse_e:
                    raise Exception(f"DEBUG CRASH: ast failed. Error: {parse_e}")
                    
            clean = [p.strip().strip('"\'') for p in text.split(',') if p.strip()]
            if len(clean) >= 2:
                return img_path, clean
                
            raise Exception("DEBUG CRASH: failed JSON parse!")

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                print(f"[Rate Limit] -> {os.path.basename(img_path)}. 休眠 30 秒後重試...")
                time.sleep(30)
            elif "500" in err_str or "503" in err_str:
                print(f"[Server Error] -> {os.path.basename(img_path)}. 休眠 10 秒後重試...")
                time.sleep(10)
            else:
                print(f"[系統錯誤] -> {os.path.basename(img_path)}: {type(e).__name__} - {e}")
                time.sleep(5)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    all_jpgs = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    existing_basenames = {os.path.basename(k) for k in existing_data.keys()}
    
    to_process = [p for p in all_jpgs if os.path.basename(p) not in existing_basenames]
    
    # 🌟 為了跟 Vertex AI 腳本完美分工，我們讓這個腳本「從最後面開始往前跑」！
    # 這樣兩支腳本就像工人從隧道兩端同時開挖，絕對不會撞在一起重複算！
    to_process.reverse()

    print(f"Total processed currently: {len(existing_data)}")
    print(f"Remaining to process (AI Studio API): {len(to_process)}")
    
    WORKERS = 1
    print(f"Launching generation with {WORKERS} concurrent workers (AI Studio)...")

    # 這個腳本專門寫入 captions_aistudio.json
    output_data = {}
    if os.path.exists(FINAL_OUTPUT):
        with open(FINAL_OUTPUT, 'r', encoding='utf-8') as f:
            output_data.update(json.load(f))

    with tqdm(total=len(to_process)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(label_image, p): p for p in to_process}
            
            count = 0
            for future in concurrent.futures.as_completed(futures):
                fname, labels = future.result()
                output_data[fname] = labels
                
                count += 1
                pbar.update(1)
                
                if count % 50 == 0:
                    temp_path = FINAL_OUTPUT + ".tmp"
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                    os.replace(temp_path, FINAL_OUTPUT)

    temp_path = FINAL_OUTPUT + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, FINAL_OUTPUT)
    print("\n🎉 AI Studio 高速隧道開挖完畢！")
