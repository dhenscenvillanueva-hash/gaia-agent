import json
import os
import re
import subprocess
import sys
import tempfile

import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

API_URL = "https://agents-course-unit4-scoring.hf.space"

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
MODEL = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
THINK = os.getenv("OLLAMA_THINK", "1") == "1"
CACHE_FILE = "answers_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (alfred-gala-agent; GAIA course)"}


def build_llm():
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=16384, num_predict=4096,
                          reasoning=THINK, client_kwargs={"timeout": 900, "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY", "")}})
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODEL, temperature=0, timeout=300, max_retries=8)
    if PROVIDER == "hf":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODEL, temperature=0, timeout=300,
                          base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])
    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=MODEL, temperature=0, max_tokens=4096, timeout=300)
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


SYSTEM_PROMPT = """You are a general AI assistant that answers questions using tools.
Never guess facts: use web_search, then read_webpage (Wikipedia pages work well) to verify.
Use run_python for any calculation, counting, table or data processing (e.g. put the matching rows in a list and print its length).
When a page is long, call read_webpage with `find` set to a keyword to jump to the relevant part.

Finish your reply with: FINAL ANSWER: [YOUR FINAL ANSWER]
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
- Number: no thousands separators, no units ($, %) unless asked.
- String: no articles, no abbreviations, digits in plain text unless asked.
- Copy names/items exactly as worded in the source, keeping descriptors (e.g. "freshly squeezed lemon juice", not "lemon juice").
- Follow any format the question requests (alphabetical order, IOC code, first name only, etc.)."""


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "sup", "noscript"]):
        t.decompose()
    for t in soup.select(".mw-editsection"):
        t.decompose()
    for table in soup.find_all("table"):
        rows = [" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])) for tr in table.find_all("tr")]
        table.replace_with("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n")
    for t in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "br", "dd", "dt", "div"]):
        t.append("\n")
    text = re.sub(r"[ \t]+", " ", soup.get_text(" "))
    return re.sub(r"\s*\n\s*", "\n", text).strip()


@tool
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and snippets of the top results."""
    try:
        from ddgs import DDGS
        results = DDGS(timeout=20).text(query, max_results=6)
        return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results) or "No results."
    except Exception as e:
        return f"Search error: {e}"


@tool
def read_webpage(url: str, find: str = "") -> str:
    """Read a web page as text (tables become ' | ' rows). Optional `find`: a keyword; only the parts of the page around it are returned."""
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        text = html_to_text(r.text)
    except Exception as e:
        return f"Error reading {url}: {e}"
    if not find:
        return text[:8000]
    hits = [m.start() for m in re.finditer(re.escape(find), text, re.I)]
    if not hits:
        return f"'{find}' not found. Page start:\n{text[:3000]}"
    out, last_end = [], -1
    for h in hits:
        start, end = max(h - 300, last_end), h + 4000
        if start >= end:
            continue
        out.append(text[start:end])
        last_end = end
        if sum(map(len, out)) > 10000:
            break
    return "\n...\n".join(out)[:10000]


@tool
def wikipedia_search(query: str) -> str:
    """Find English Wikipedia article titles and URLs matching a query."""
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=15, params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5})
        hits = r.json()["query"]["search"]
        return "\n".join(f"{h['title']}: https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')}" for h in hits) or "No results."
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def run_python(code: str) -> str:
    """Execute Python code and return what it prints. Always print() the result. pandas is available."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr)[:4000] or "(no output - did you print()?)"
    except subprocess.TimeoutExpired:
        return "Timed out after 60s."
    finally:
        os.unlink(f.name)


@tool
def youtube_transcript(url: str) -> str:
    """Get the spoken transcript (captions) of a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        vid = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
        snippets = YouTubeTranscriptApi().fetch(vid.group(1) if vid else url)
        return "\n".join(s.text for s in snippets)[:8000]
    except Exception as e:
        return f"Transcript unavailable ({type(e).__name__}). Try web_search about the video instead."


TOOLS = [web_search, wikipedia_search, read_webpage, run_python, youtube_transcript]


def transcribe(path: str) -> str:
    import imageio_ffmpeg
    import numpy as np
    import whisper
    pcm = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    return whisper.load_model("small").transcribe(audio, fp16=False)["text"].strip()


def download_attachment(task_id: str, file_name: str) -> str:
    r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
    if r.ok:
        path = os.path.join(tempfile.gettempdir(), file_name)
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download("gaia-benchmark/GAIA", f"2023/validation/{file_name}", repo_type="dataset")


def load_attachment(task_id: str, file_name: str) -> str:
    """Download the task's file and turn it into text the model can use."""
    path = download_attachment(task_id, file_name)
    ext = file_name.rsplit(".", 1)[-1].lower()
    if ext == "py":
        code = open(path, encoding="utf-8", errors="replace").read()
        out = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=120)
        return f"Attached Python file ({path}):\n```python\n{code}\n```\nOutput when executed:\n{out.stdout}{out.stderr}"
    if ext in ("xlsx", "xls", "csv"):
        df = pd.read_csv(path) if ext == "csv" else pd.read_excel(path)
        return (f"Attached spreadsheet saved at {path} (load it with pandas in run_python).\n"
                f"Columns: {list(df.columns)}\nContents:\n{df.to_string()[:6000]}")
    if ext in ("mp3", "wav", "m4a", "flac"):
        return f"Attached audio transcript:\n{transcribe(path)}"
    if ext in ("txt", "md", "json"):
        return f"Attached file:\n{open(path, encoding='utf-8', errors='replace').read()[:6000]}"
    return f"An attached {ext} file is saved at {path}, but you cannot view it directly."


def extract_final(text) -> str:
    if isinstance(text, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.findall(r"FINAL ANSWER:\s*(.+)", text, re.I)
    ans = m[-1] if m else text.strip().splitlines()[-1] if text.strip() else ""
    ans = " ".join(ans.split()).strip("*`[]").strip().rstrip(".")
    return re.sub(r"\s*,\s*", ", ", ans)


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


class GaiaAgent:
    def __init__(self):
        self.llm = build_llm()
        self.graph = create_agent(self.llm, TOOLS, system_prompt=SYSTEM_PROMPT)

    def __call__(self, question: str, task_id: str = "", file_name: str = "") -> str:
        prompt = question
        if file_name:
            try:
                prompt += "\n\n" + load_attachment(task_id, file_name)
            except Exception as e:
                prompt += f"\n\n(Attachment {file_name} could not be loaded: {e})"
        for attempt in range(3):
            try:
                return self.solve(prompt)
            except Exception as e:
                print(f"   attempt {attempt + 1} failed: {type(e).__name__}: {e}")
        return ""

    def solve(self, prompt: str) -> str:
        messages = [HumanMessage(prompt)]
        try:
            for state in self.graph.stream({"messages": messages}, {"recursion_limit": 30}, stream_mode="values"):
                messages = state["messages"]
        except GraphRecursionError:
            print("   step limit reached, answering from research so far")
            if getattr(messages[-1], "tool_calls", None):
                messages = messages[:-1]
        last = _text(messages[-1].content)
        if "FINAL ANSWER" not in last.upper():
            nudge = HumanMessage("Based on everything above, reply with only: FINAL ANSWER: <answer>")
            last = _text(self.llm.invoke([*messages, nudge]).content)
        return extract_final(last)


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def answer_all(questions: list, refresh: bool = False) -> dict:
    """Answer every question, caching per task_id so an interrupted run can resume."""
    agent, cache = GaiaAgent(), {} if refresh else load_cache()
    for i, q in enumerate(questions, 1):
        if cache.get(q["task_id"]):
            continue
        print(f"\n[{i}/{len(questions)}] {q['question'][:90]!r}")
        cache[q["task_id"]] = agent(q["question"], q["task_id"], q.get("file_name", ""))
        print(f"   -> {cache[q['task_id']]!r}")
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    return cache


def submit(username: str, answers: dict) -> dict:
    space_id = os.getenv("SPACE_ID", "")
    payload = {
        "username": username,
        "agent_code": os.getenv("AGENT_CODE_URL") or (f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "local run of final_assesment.py"),
        "answers": [{"task_id": k, "submitted_answer": " ".join(v.split())} for k, v in answers.items()],
    }
    r = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_questions() -> list:
    r = requests.get(f"{API_URL}/questions", timeout=30)
    r.raise_for_status()
    return r.json()


def run_and_submit(refresh: bool, profile: gr.OAuthProfile | None):
    if not profile:
        return "Please log in to Hugging Face first.", None
    questions = fetch_questions()
    answers = answer_all(questions, refresh)
    table = pd.DataFrame([{"Question": q["question"][:90], "Answer": answers[q["task_id"]]} for q in questions])
    try:
        res = submit(profile.username, answers)
        return f"Score: {res.get('score')}% ({res.get('correct_count')}/{res.get('total_attempted')})\n{res.get('message', '')}", table
    except Exception as e:
        return f"Submission error: {e}", table


def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown(f"# GAIA Agent ({MODEL} via {PROVIDER})")
        gr.LoginButton()
        refresh = gr.Checkbox(label="Ignore cached answers and re-run every question", value=False)
        btn = gr.Button("Run & Submit")
        status = gr.Textbox(label="Status", lines=4, interactive=False)
        table = gr.DataFrame(label="Answers", wrap=True)
        btn.click(run_and_submit, inputs=[refresh], outputs=[status, table])
    demo.launch()


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        answer_all(fetch_questions(), refresh="--refresh" in sys.argv)
    elif sys.argv[1:2] == ["submit"]:
        print(submit(sys.argv[2], load_cache()))
    else:
        launch_ui()
