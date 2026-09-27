"""Test Gemini models via the google-generativeai SDK."""
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(Path(__file__).parent / ".env")
key = os.environ.get("GEMINI_API_KEY", "")
if not key:
    print("GEMINI_API_KEY not set in .env — aborting")
    raise SystemExit(1)

genai.configure(api_key=key)
print(f"Testing with key prefix: {key[:12]}...\n")

# List available models
try:
    models_list = [m.name for m in genai.list_models()
                   if "generateContent" in m.supported_generation_methods]
    print(f"Available text models: {models_list}\n")
except Exception as e:
    print(f"Could not list models: {e}")
    models_list = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

# Test each
for m in models_list:
    try:
        model = genai.GenerativeModel(m)
        resp = model.generate_content(
            "Say OK only.",
            generation_config=genai.GenerationConfig(max_output_tokens=10, temperature=0),
        )
        reply = resp.text
        print(f"  [OK]   {m} -> {reply!r}")
    except Exception as e:
        print(f"  [FAIL] {m} -> {str(e)[:100]}")
