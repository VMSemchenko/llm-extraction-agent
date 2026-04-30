#!/usr/bin/env python3
"""
LLM Extraction Agent — Lesson 06 Homework

Витягує структуровані дані (JSON) з неструктурованих транскриптів зустрічей.
Підтримує провайдери: Ollama (self-hosted) та Google Gemini (cloud).
"""

import json
import os
import re
import sys
import time
import csv
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

load_dotenv()

# ── CONFIG ──
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── OLLAMA (self-hosted) ──
def call_ollama(prompt: str) -> str:
    """Викликає локальну модель через Ollama API"""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


# ── GOOGLE GEMINI (cloud) ──
def call_gemini(prompt: str) -> str:
    """Викликає Gemini через Google AI API"""
    if not HAS_GEMINI:
        raise RuntimeError("google-genai package is not installed. Run: pip install google-genai")

    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return response.text


# ── EXTRACTION LOGIC ──
def build_prompt(text: str) -> str:
    """Створює промпт для екстракції даних із зустрічі."""
    return f"""Прочитай цей текст зустрічі та витягни:
1. summary (одне речення українською — про що була зустріч)
2. tasks (список завдань з полями: owner, task, deadline)
3. decisions (рішення, які було прийнято)

Текст:
{text}

Поверни ТІЛЬКИ JSON, нічого більше:
{{
  "summary": "...",
  "tasks": [
    {{"owner": "...", "task": "...", "deadline": "..."}}
  ],
  "decisions": ["..."]
}}"""


def estimate_tokens(text: str) -> int:
    """Приблизна оцінка кількості токенів (слово ≈ 1.3 токена)."""
    return int(len(text.split()) * 1.3)


def extract_meeting_data(text: str, provider: str = "ollama") -> dict:
    """
    Витягує структурований JSON з тексту зустрічі.

    Args:
        text: Транскрипт зустрічі
        provider: "ollama" або "gemini"

    Returns:
        dict з ключами: result, latency, tokens_in, tokens_out, tokens_total, cost, json_valid
    """
    prompt = build_prompt(text)
    tokens_in = estimate_tokens(prompt)

    start = time.time()
    try:
        if provider == "ollama":
            response_text = call_ollama(prompt)
        elif provider == "gemini":
            response_text = call_gemini(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        return {
            "result": None,
            "raw_response": str(e),
            "latency": round(time.time() - start, 2),
            "tokens_in": tokens_in,
            "tokens_out": 0,
            "tokens_total": tokens_in,
            "cost": 0,
            "json_valid": False,
            "error": str(e),
        }
    latency = time.time() - start

    tokens_out = estimate_tokens(response_text)
    tokens_total = tokens_in + tokens_out

    # Cost calculation (Gemini 2.0 Flash: ~$0.10 per 1M tokens)
    if provider == "gemini":
        cost = (tokens_total / 1_000_000) * 0.10
    else:
        cost = 0.0

    # Parse JSON
    json_valid = False
    result = None
    try:
        result = json.loads(response_text)
        json_valid = True
    except json.JSONDecodeError:
        # Try to extract JSON from response (models sometimes add markdown)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                json_valid = True
            except json.JSONDecodeError:
                pass

    return {
        "result": result,
        "raw_response": response_text,
        "latency": round(latency, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_total,
        "cost": round(cost, 6),
        "json_valid": json_valid,
    }


def run_single(filepath: str, provider: str) -> dict:
    """Запускає агента на одному файлі з одним провайдером."""
    text = Path(filepath).read_text(encoding="utf-8")
    dataset_name = Path(filepath).stem

    icon = "\U0001f916" if provider == "ollama" else "\u2601\ufe0f "
    print(f"\n{icon}  {provider.upper()} \u2014 {dataset_name}")
    print(f"   \u0417\u0430\u043f\u0438\u0442 \u0434\u043e {provider}...")

    data = extract_meeting_data(text, provider)

    if data["json_valid"]:
        print(f"   \u2705 JSON \u0432\u0430\u043b\u0456\u0434\u043d\u0438\u0439 | Latency: {data['latency']}s | Tokens: {data['tokens_total']} | Cost: ${data['cost']}")
        tasks_count = len(data["result"].get("tasks", [])) if data["result"] else 0
        decisions_count = len(data["result"].get("decisions", [])) if data["result"] else 0
        print(f"   \U0001f4cb Tasks: {tasks_count} | Decisions: {decisions_count}")
        print(f"   \U0001f4dd Summary: {data['result'].get('summary', 'N/A')[:100]}")
    else:
        print(f"   \u274c JSON \u043d\u0435\u0432\u0430\u043b\u0456\u0434\u043d\u0438\u0439 | Latency: {data['latency']}s")
        print(f"   Raw: {data.get('raw_response', '')[:200]}")

    # Save result
    result_file = RESULTS_DIR / f"{dataset_name}_{provider}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "provider": provider,
                "json_valid": data["json_valid"],
                "latency": data["latency"],
                "tokens_in": data["tokens_in"],
                "tokens_out": data["tokens_out"],
                "tokens_total": data["tokens_total"],
                "cost": data["cost"],
                "result": data["result"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"   \U0001f4be Saved: {result_file}")

    return {
        "dataset": dataset_name,
        "provider": provider,
        **{k: data[k] for k in ["json_valid", "latency", "tokens_in", "tokens_out", "tokens_total", "cost"]},
        "tasks_found": len(data["result"].get("tasks", [])) if data.get("result") else 0,
    }


def run_all(providers: list = None):
    """Запускає агента на всіх sample-файлах."""
    if providers is None:
        providers = ["ollama"]

    samples = sorted(Path("samples").glob("*.txt"))
    if not samples:
        print("\u274c No sample files found in samples/")
        sys.exit(1)

    all_metrics = []

    for sample in samples:
        for provider in providers:
            try:
                metrics = run_single(str(sample), provider)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"   \u274c Error: {e}")
                all_metrics.append({
                    "dataset": sample.stem,
                    "provider": provider,
                    "json_valid": False,
                    "latency": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "tokens_total": 0,
                    "cost": 0,
                    "tasks_found": 0,
                })

    # Save eval_results.csv
    if all_metrics:
        csv_path = Path("eval_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n\U0001f4ca Evaluation saved: {csv_path}")


# ── CLI ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Extraction Agent")
    parser.add_argument("file", nargs="?", help="Path to meeting transcript file")
    parser.add_argument("--all", action="store_true", help="Run on all sample files")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "both"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    args = parser.parse_args()

    providers = ["ollama", "gemini"] if args.provider == "both" else [args.provider]

    if args.all:
        run_all(providers)
    elif args.file:
        for p in providers:
            run_single(args.file, p)
    else:
        print("\u2139\ufe0f  No file specified. Running on all samples with Ollama...")
        print("   Use --provider both to also test Gemini")
        print("   Use --provider gemini to test only Gemini")
        run_all(providers)
