# QpiVOLTA Research Brain — Configuration & Tuning Guide

A local AI assistant with RAG, ReAct agent loop, web search, and agentic code execution.
Runs fully offline on Ollama. No cloud APIs needed.

---

## Quick Start

```bash
# normal mode (loads RAG vectorstore)
rag_ollama

# fast mode — no vectorstore, direct LLM + agent
rag_chat   # alias for: python main.py --no-rag

# with indexing enabled (scans SOURCE_DIRS for new files)
AUTO_INDEX=true rag_ollama
```

---

## File Map

| File | Purpose |
|------|---------|
| `config.py` | All paths, models, extensions — main config file |
| `main.py` | Agent loop, ReAct logic, tuning constants |
| `memory.py` | Conversation history, rolling summary |
| `rag.py` | RAG retrieval chain, confidence check |
| `indexer.py` | Vectorstore indexing logic |
| `loaders.py` | File loaders (PDF, DOCX, PPTX, code, etc.) |
| `editor.py` | Sandboxed workspace file operations |
| `downloader.py` | Paper download from arXiv / URLs |
| `websearch.py` | Web search integration |

---

## config.py — Tunable Values

### Models

```python
AVAILABLE_MODELS = {
    "1": ("granite4:3b",       "fast, fits in VRAM fully"),
    "2": ("phi4-mini",         "fast, good reasoning"),
    "3": ("gemma4:e4b",        "slow but smartest, partial CPU"),
    "4": ("gemma4:e2b",        "medium speed, good quality"),
    "6": ("qwen2.5-coder:7b",  "coding and learning"),
}
```

Add any model you have in Ollama:
```python
"7": ("llama3.2:3b", "general purpose"),
```

Check available models: `ollama list`

### Embedding Model

```python
EMBEDDING_MODEL = "nomic-embed-text"
```

Change to a different embedding model if needed. Must be pulled in Ollama first.
Only affects RAG mode. Changing this invalidates your existing vectorstore — delete `DB_DIR` and re-index.

### Paths

```python
SOURCE_DIRS = ["/path/to/papers", ...]   # folders scanned for indexing
DB_DIR      = "/path/to/vectorstore"     # ChromaDB storage
CONV_DIR    = "/path/to/conversations"   # saved chat JSON files
DOWNLOAD_DIR= "/path/to/downloads"       # papers downloaded via /p
WORKSPACE_DIR = "/path/to/workspace"     # agent writes files here
```

### Indexing

```python
AUTO_INDEX = os.environ.get("AUTO_INDEX", "false").lower() == "true"
```

Default: off. Enable per-run:
```bash
AUTO_INDEX=true python main.py
```

To always index on startup, change default:
```python
AUTO_INDEX = os.environ.get("AUTO_INDEX", "true").lower() == "true"
```

### Supported File Extensions for Indexing

```python
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".py", ".md", ...}
```

Add extensions to index more file types. Loader must exist in `loaders.py`.

---

## main.py — Tunable Values

### Model Temperature

```python
llm = ChatOllama(model=model_name, temperature=0.1)
```

- `0.0` — fully deterministic, best for code
- `0.1` — default, slight variation
- `0.7` — more creative, worse for code
- `1.0` — maximum randomness

### ReAct Agent — Max Steps per Query

```python
def _react_loop(query, llm, history, vectorstore=None, force_web=False, max_steps=10):
```

Change `max_steps=10` to allow more or fewer tool-use steps per query.
Higher = more capable for complex tasks, slower, more tokens used.

### ReAct Agent — Max Fix Attempts (legacy `_handle_agent`)

No longer the default path, but if called directly:
```python
def _handle_agent(task, llm, max_tries=5):
```

Change `max_tries=5`.

### Context Window — Recent Turns

```python
history_text = format_history_for_prompt(history, max_turns=8)
```

How many recent conversation turns to include verbatim in every prompt.
- Increase for better short-term memory (uses more tokens)
- `qwen2.5-coder:7b` has 32k context — safe up to ~20 turns before hitting limits
- Each turn ≈ 200–600 tokens depending on answer length

### Context Window — Rolling Summary Frequency

```python
if len(history) > 0 and len(history) % 4 == 0:
```

A compressed summary of the full conversation is generated every 4 turns.
Change `4` to compress more or less often:
- `2` — summarize every 2 turns (more overhead, tighter memory)
- `8` — summarize every 8 turns (less overhead, longer raw history before compression)

### Thinking Mode

`<|think|>` prefix triggers chain-of-thought reasoning in Qwen models before answering.
Currently **not active** in the ReAct loop. To re-enable, find all `llm.invoke(messages)` calls in `_react_loop` and change the first message to prefix the query:

```python
# in _react_loop, change:
HumanMessage(content=f"{SYSTEM_PROMPT}\n\n{context}")
# to:
HumanMessage(content=f"{SYSTEM_PROMPT}\n\n{context}\n\n<|think|>")
```

For the `_handle_edit` single-step path:
```python
# change:
result = llm.invoke([HumanMessage(content=prompt)]).content
# to:
result = llm.invoke([HumanMessage(content=f"<|think|> {prompt}")]).content
```

**Note:** Thinking adds latency (~2–5s extra). Only supported on Qwen3 and Qwen2.5 models.
Granite and Gemma will ignore the token.

### System Prompt

The full agent system prompt is in `main.py` as `SYSTEM_PROMPT`:

```python
SYSTEM_PROMPT = """You are an intelligent assistant and coding agent..."""
```

Edit this to change agent personality, add constraints, change output format, or add new actions.
To make it more research-focused:
```python
SYSTEM_PROMPT = """You are a research assistant specializing in quantum computing and physics...
```

To make it stricter about code quality:
```python
# add to Rules section:
- Always add error handling to code you write.
- Prefer typed Python over untyped.
```

---

## memory.py — Tunable Values

### Default History Turns in Prompt

```python
def format_history_for_prompt(history, max_turns=8):
```

This is the fallback used by RAG mode (`rag.py`). Change `max_turns` here to affect RAG context.

---

## rag.py — Tunable Values

### Confidence Threshold for Web Search

```python
if confidence >= 70:
    return draft_answer, []
```

If RAG answer confidence < 70%, web search triggers automatically.
- Raise to `85` — web search triggers more often
- Lower to `50` — web search rarely triggers
- Set to `100` — web search always triggers (same as always using `/w`)
- Set to `0` — web search never triggers from confidence check

### RAG Retrieval — Number of Chunks

```python
vectorstore.as_retriever(search_kwargs={"k": 5})
```

`k=5` means 5 document chunks retrieved per query.
- Increase to `10` for broader context (slower, more tokens)
- Decrease to `3` for faster, more focused retrieval

---

## indexer.py — Tunable Values

### Chunk Size and Overlap

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
```

- `chunk_size` — characters per chunk. Larger = more context per chunk, fewer chunks
- `chunk_overlap` — overlap between chunks. Helps avoid cutting mid-sentence
- For dense technical papers: `chunk_size=2000, chunk_overlap=300`
- For fast indexing: `chunk_size=1000, chunk_overlap=100`

### Indexing Batch Size

```python
def save_in_batches(vectorstore, docs, batch_size=50):
```

Reduce if indexing crashes on large files. Increase for faster indexing on good hardware.

### File Load Timeout

```python
raw_docs = load_file_with_timeout(file_path, seconds=120)
```

Files taking longer than 120s to load are skipped. Increase for very large PDFs.

---

## Adding a New Model

1. Pull it in Ollama: `ollama pull mistral:7b`
2. Add to `config.py`:
```python
"7": ("mistral:7b", "general purpose"),
```
3. Start the app, pick `7`.

---

## Aliases (bashrc)

```bash
alias rag_ollama='source /path/to/.venv/bin/activate && python /path/to/main.py && deactivate'
alias rag_chat='source /path/to/.venv/bin/activate && python /path/to/main.py --no-rag && deactivate'
```

---

## Commands Inside the App

| Command | What it does |
|---------|-------------|
| `<anything>` | Agent decides: answer / write / run / read / search |
| `/w <question>` | Force web search before answering |
| `/edit <task>` | Single-step file create/edit in workspace |
| `/p <question>` | Web search + download papers + RAG index (RAG mode only) |
| `convos` | List and load saved conversations |
| `exit` | Save conversation and quit |
