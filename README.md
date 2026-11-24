# Build AI Agents in Python (Free, No GPU)

Code for the **Astrada Codes** YouTube series. Every episode runs on a normal laptop with the **free Gemini API**: no GPU, no big downloads.

| # | Episode | Video | Code |
|---|---|---|---|
| 0 | How to Get a Free Gemini API Key (2026) + First Python Test | [Watch](https://youtu.be/Nt7wZXDt25U) | see setup below |
| 1 | Build a Free AI Agent in Python (Gemini API, No GPU) | [Watch](https://youtu.be/rhu-PyfFgHM) | [episode-01-first-agent](episode-01-first-agent/agent.py) |
| 2 | Give Your AI Agent Memory in Python | [Watch](https://youtu.be/eMnkdTagJTk) | [episode-02-memory](episode-02-memory/agent.py) |

## Setup (once)

1. Get a free API key at [aistudio.google.com](https://aistudio.google.com) (Get API key, then Create API key).
2. Save it as an environment variable (never put it in your code):
   - Windows: `setx GEMINI_API_KEY "paste-your-key"`, then open a **new** terminal
   - macOS / Linux: `export GEMINI_API_KEY="paste-your-key"`
3. Install the library: `pip install google-genai`

## Run

```bash
python episode-01-first-agent/agent.py
python episode-02-memory/agent.py --demo   # or without --demo to chat
```

Model used: `gemini-3.5-flash-lite` (free tier). If you get `404 NOT_FOUND`, list the models your key can use:

```python
from google import genai
for m in genai.Client().models.list():
    print(m.name)
```

## Notes

- Free tier: rate limits apply, and Google can use free-tier prompts to improve its products. Keep private data out.
- Never commit your API key. `.gitignore` already skips `memory.json` and `output.txt`.

Subscribe: [youtube.com/@AstradaCodes](https://www.youtube.com/@AstradaCodes)
