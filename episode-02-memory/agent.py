"""Episode 2 - Give your AI agent memory (free Gemini API, no GPU).
Short-term memory = the conversation list. Long-term memory = a JSON file.
Run:  python agent.py           chat (type 'exit' to quit)
      python agent.py --demo    scripted demo, writes output.txt
      python agent.py --forget  delete the saved memory
"""
import ast, json, operator, sys, time
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"
HERE = Path(__file__).parent
MEMORY = HERE / "memory.json"
OUT = HERE / "output.txt"

def log(msg=""):
    print(msg)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ---------- TOOLS (same as episode 1) ----------
def get_current_time() -> str:
    """Returns the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def calculate(expression: str) -> str:
    """Safely evaluates a math expression like '120 * 1.5 * 7'."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return str(round(ev(ast.parse(expression, mode="eval").body), 4))

TOOLS = {f.__name__: f for f in (get_current_time, calculate)}

# ---------- MEMORY ----------
def load_memory():
    """Long-term memory: read past messages from memory.json."""
    if not MEMORY.exists():
        return []
    turns = json.loads(MEMORY.read_text(encoding="utf-8"))
    return [types.Content(role=t["role"], parts=[types.Part(text=t["text"])]) for t in turns]

def save_memory(contents, keep=40):
    """Save only the text messages (not tool calls), newest `keep` messages."""
    turns = [{"role": c.role, "text": p.text} for c in contents for p in (c.parts or []) if p.text]
    turns = turns[-keep:]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    MEMORY.write_text(json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------- MODEL + LOOP ----------
client = genai.Client()
config = types.GenerateContentConfig(
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    system_instruction="You are a helpful assistant. Use the tools when they help. "
                       "Remember what the user tells you. Answer briefly.",
)

def ask(contents, text):
    """One user message -> agent loop -> answer. `contents` IS the short-term memory."""
    contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
    for step in range(1, 6):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        calls = resp.function_calls or []
        if not calls:
            answer = (resp.text or "").strip()
            contents.append(types.Content(role="model", parts=[types.Part(text=answer)]))
            return answer
        contents.append(resp.candidates[0].content)
        for fc in calls:
            args = dict(fc.args or {})
            result = TOOLS[fc.name](**args)
            log(f"   [tool] {fc.name}({args}) -> {result}")
            contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(name=fc.name, response={"result": result})]))
    return "(stopped: too many steps)"

def demo():
    OUT.write_text("", encoding="utf-8")
    MEMORY.unlink(missing_ok=True)
    start = time.time()
    log("=== SESSION 1 ===")
    contents = load_memory()
    for text in ["Hi! My name is Rafi and I sell homemade cookies.",
                 "I sell 120 cookies a day at $1.50 each. What's my weekly revenue?",
                 "Thanks! What's my name again?"]:
        log(f"You:   {text}")
        log(f"Agent: {ask(contents, text)}")
        save_memory(contents)
    log("\n=== SESSION 2 (program restarted, memory loaded from memory.json) ===")
    contents = load_memory()
    log(f"   [memory] {len(contents)} messages loaded")
    text = "I'm back. Do you remember my name and what I sell?"
    log(f"You:   {text}")
    log(f"Agent: {ask(contents, text)}")
    save_memory(contents)
    log(f"\nTotal time: {time.time() - start:.1f} s  |  model: {MODEL}")

def chat():
    contents = load_memory()
    print(f"Memory: {len(contents)} messages loaded. Type 'exit' to quit.")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if text:
            print("Agent:", ask(contents, text))
            save_memory(contents)

if __name__ == "__main__":
    if "--forget" in sys.argv:
        MEMORY.unlink(missing_ok=True); print("Memory deleted.")
    elif "--demo" in sys.argv:
        demo()
    else:
        chat()
