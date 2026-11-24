"""Episode 1 - Your first AI agent in Python (free, no GPU).
An agent = a model + tools + a loop.
Run:  python agent.py
"""
import ast, operator, os, time
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"
OUT = Path(__file__).with_name("output.txt")
OUT.write_text("", encoding="utf-8")

def log(msg=""):
    print(msg)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ---------- 1. TOOLS: normal Python functions ----------
def get_current_time() -> str:
    """Returns the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def calculate(expression: str) -> str:
    """Safely evaluates a math expression like '47.5 * 0.18'."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return str(round(ev(ast.parse(expression, mode="eval").body), 4))

def list_files(folder: str = ".") -> str:
    """Lists the files in a folder next to this script."""
    p = (Path(__file__).parent / folder).resolve()
    return ", ".join(sorted(x.name for x in p.iterdir())) or "(empty)"

TOOLS = {f.__name__: f for f in (get_current_time, calculate, list_files)}

# ---------- 2. MODEL + 3. LOOP ----------
client = genai.Client()
config = types.GenerateContentConfig(
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    system_instruction="You are a helpful assistant. Use the tools when they help. Answer briefly.",
)

task = "What time is it right now? Also, what is an 18% tip on a $47.50 bill? And which files are in this folder?"
log(f"TASK: {task}\n")
contents = [types.Content(role="user", parts=[types.Part(text=task)])]
start = time.time()

for step in range(1, 6):
    resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
    calls = resp.function_calls or []
    if not calls:
        log(f"FINAL ANSWER:\n{resp.text.strip()}")
        break
    contents.append(resp.candidates[0].content)
    for fc in calls:
        args = dict(fc.args or {})
        result = TOOLS[fc.name](**args)
        log(f"STEP {step}: agent calls {fc.name}({args}) -> {result}")
        contents.append(types.Content(role="user", parts=[types.Part.from_function_response(name=fc.name, response={"result": result})]))

log(f"\nTotal time: {time.time() - start:.1f} s  |  model: {MODEL}")
