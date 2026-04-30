# 🤖 LLM Extraction Agent

Homework for Lesson 06: LLM Engineering (API + Self-hosted)

Агент для витягування структурованої інформації (JSON) з неструктурованих транскриптів робочих зустрічей.
Порівняння **Ollama** (self-hosted, Mistral 7B) та **OpenAI** (cloud, GPT-4o-mini).

## 📋 Структура

```
├── extraction_agent.py       # Головний агент (Ollama + OpenAI)
├── requirements.txt          # Python залежності
├── .env.example              # Шаблон для API ключа
├── samples/                  # Тестові транскрипти зустрічей
│   ├── simple_meeting.txt    # Простий — чітка структура
│   ├── chaotic_standup.txt   # Хаотичний — переривання, шум
│   └── technical_sync.txt    # Технічний — жаргон, складні рішення
├── results/                  # JSON результати від моделей
│   ├── simple_ollama.json
│   ├── simple_openai.json
│   └── ...
├── eval_results.csv          # Таблиця порівняння метрик
└── ANALYSIS.md               # Висновки та аналіз
```

## 🚀 Запуск

### Підготовка

```bash
# 1. Клонувати репозиторій
git clone https://github.com/VMSemchenko/llm-extraction-agent.git
cd llm-extraction-agent

# 2. Створити віртуальне середовище
python3 -m venv .venv
source .venv/bin/activate

# 3. Встановити залежності
pip install -r requirements.txt

# 4. Налаштувати .env (для OpenAI)
cp .env.example .env
# Вписати свій OPENAI_API_KEY

# 5. Переконатись що Ollama запущена
ollama serve  # в окремому терміналі
ollama pull mistral
```

### Запуск агента

```bash
# На конкретному файлі
python extraction_agent.py samples/simple_meeting.txt

# На всіх файлах (автоматично)
python extraction_agent.py --all

# Тільки Ollama
python extraction_agent.py samples/simple_meeting.txt --provider ollama

# Тільки OpenAI
python extraction_agent.py samples/simple_meeting.txt --provider openai

# Обидва провайдери
python extraction_agent.py --all --provider both
```

## 📊 Метрики

| Метрика | Опис |
|---------|------|
| JSON валідність | Чи вдається `json.loads()` спарсити? |
| Знайдено завдань | Скільки tasks знайдено vs реальних |
| Галюцинації | Вигадані імена/дати |
| Токени (total) | Input + Output |
| Вартість | OpenAI pricing / Ollama = $0 |
| Latency | Час відповіді в секундах |

## 📚 Технології

- **Ollama** (Mistral 7B) — self-hosted LLM
- **OpenAI** (GPT-4o-mini) — cloud API
- **Python 3.10+**

## 📝 Автор

Vladyslav Semchenko — AI Engineering Course, Lesson 06
