#!/usr/bin/env python3
"""
LLM Extraction Agent — Lesson 06 Homework

Витягує структуровані дані (JSON) з неструктурованих транскриптів зустрічей.
Підтримує два провайдери: Ollama (self-hosted) та OpenAI (cloud).
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
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

load_dotenv()

# ── CONFIG ──
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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


# ── OPENAI (cloud) ──
def call_openai(prompt: str) -> str:
    """Викликає GPT-4o-mini через OpenAI API"""
    if not HAS_OPENAI:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        raise RuntimeError("Set a valid OPENAI_API_KEY in .env")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a meeting transcription parser. "
                    "Extract tasks, decisions, and a summary from meeting text. "
                    "Return ONLY valid JSON, no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


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
        provider: "ollama" або "openai"

    Returns:
        dict з ключами: result, latency, tokens_in, tokens_out, tokens_total, cost, json_valid
    """
    prompt = build_prompt(text)
    tokens_in = estimate_tokens(prompt)

    start = time.time()
    try:
        if provider == "ollama":
            response_text = call_ollama(prompt)
        elif provider == "openai":
            response_text = call_openai(prompt)
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

    # Cost calculation
    if provider == "openai":
        cost = (tokens_total / 1_000_000) * 0.30  # $0.30 per 1M tokens (4o-mini avg)
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
    print(f"\n{icon}  {provider.upper()} — {dataset_name}")
    print(f"   Запит до {provider}...")

    data = extract_meeting_data(text, provider)

    if data["json_valid"]:
        print(f"   \u2705 JSON валідний | Latency: {data['latency']}s | Tokens: {data['tokens_total']} | Cost: ${data['cost']}")
        tasks_count = len(data["result"].get("tasks", [])) if data["result"] else 0
        decisions_count = len(data["result"].get("decisions", [])) if data["result"] else 0
        print(f"   \U0001f4cb Tasks: {tasks_count} | Decisions: {decisions_count}")
        print(f"   \U0001f4dd Summary: {data['result'].get('summary', 'N/A')[:100]}")
    else:
        print(f"   \u274c JSON невалідний | Latency: {data['latency']}s")
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
        choices=["ollama", "openai", "both"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    args = parser.parse_args()

    providers = ["ollama", "openai"] if args.provider == "both" else [args.provider]

    if args.all:
        run_all(providers)
    elif args.file:
        for p in providers:
            run_single(args.file, p)
    else:
        # Default: run all with ollama only
        print("\u2139\ufe0f  No file specified. Running on all samples with Ollama...")
        print("   Use --provider both to also test OpenAI")
        print("   Use --provider openai to test only OpenAI")
        run_all(providers)
