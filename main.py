import warnings
import logging
import argparse
import os
import subprocess
import random
import string
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from rich.console import Console
from rich.markdown import Markdown

console = Console()

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    AVAILABLE_MODELS,
    WORKSPACE_DIR,
)
from config import SOURCE_DIRS, AUTO_INDEX
from memory import (
    load_conversation,
    save_conversation,
    list_conversations,
    format_history_for_prompt,
)
from downloader import interactive_paper_download
from websearch import search_papers

SANDBOX = Path(WORKSPACE_DIR).resolve()


# ── file helpers ────────────────────────────────────────────────────────────


def _extract_write(text, lines=None):
    """Parse FILE + CONTENT block. Returns (filepath, content) or (None, None).
    Pass lines= to restrict FILE: lookup and CONTENT: extraction to a slice."""
    if lines is None:
        lines = text.strip().splitlines()
    fp = next(
        (l.split(":", 1)[1].strip() for l in lines if l.startswith("FILE:")), None
    )
    if not fp:
        return None, None
    if fp.startswith("workspace/"):
        fp = fp[len("workspace/") :]
    slice_text = "\n".join(lines)
    try:
        start = slice_text.index("CONTENT:\n") + len("CONTENT:\n")
        end = slice_text.index("\n---END---")
        raw = slice_text[start:end].splitlines()
        if raw and raw[0].startswith("```"):
            raw = raw[1:]
        if raw and raw[-1].strip() == "```":
            raw = raw[:-1]
        return fp, "\n".join(raw)
    except ValueError:
        return None, None


def _safe_path(rel):
    target = (SANDBOX / rel).resolve()
    if not str(target).startswith(str(SANDBOX)):
        raise PermissionError(f"Path escape: {rel}")
    return target


def _read_file(rel):
    p = _safe_path(rel)
    return p.read_text() if p.exists() else None


def _write_file(rel, content):
    p = _safe_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)


def _delete_file(rel):
    p = _safe_path(rel)
    if p.exists():
        p.unlink()
        return True
    return False


def _list_workspace():
    return [str(f.relative_to(SANDBOX)) for f in SANDBOX.rglob("*") if f.is_file()]


def _run_file(abs_path):
    """Compile+run. Returns (success, output)."""
    ext = os.path.splitext(abs_path)[1].lower()
    env = os.environ.copy()
    try:
        if ext == ".py":
            r = subprocess.run(
                ["python", abs_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        elif ext == ".rs":
            out = abs_path[:-3]
            c = subprocess.run(
                ["rustc", abs_path, "-o", out],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if c.returncode != 0:
                return False, c.stderr
            r = subprocess.run(
                [out], capture_output=True, text=True, timeout=30, env=env
            )
        elif ext == ".js":
            r = subprocess.run(
                ["node", abs_path], capture_output=True, text=True, timeout=30, env=env
            )
        elif ext == ".ts":
            r = subprocess.run(
                ["npx", "ts-node", abs_path],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        elif ext in (".c", ".h"):
            out = abs_path.replace(ext, "")
            c = subprocess.run(
                ["gcc", abs_path, "-o", out],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if c.returncode != 0:
                return False, c.stderr
            r = subprocess.run(
                [out], capture_output=True, text=True, timeout=30, env=env
            )
        elif ext == ".cpp":
            out = abs_path[:-4]
            c = subprocess.run(
                ["g++", abs_path, "-o", out],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if c.returncode != 0:
                return False, c.stderr
            r = subprocess.run(
                [out], capture_output=True, text=True, timeout=30, env=env
            )
        elif ext == ".java":
            c = subprocess.run(
                ["javac", abs_path], capture_output=True, text=True, timeout=60, env=env
            )
            if c.returncode != 0:
                return False, c.stderr
            cls = os.path.splitext(os.path.basename(abs_path))[0]
            r = subprocess.run(
                ["java", "-cp", os.path.dirname(abs_path), cls],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        elif ext in (".exe", "") and os.path.isfile(abs_path) and os.access(abs_path, os.X_OK):
            # pre-compiled binary (Linux extensionless or mistakenly named .exe)
            r = subprocess.run(
                [abs_path], capture_output=True, text=True, timeout=30, env=env
            )
        else:
            return False, f"Unsupported extension: {ext}"
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out or "(no output)"
    except subprocess.TimeoutExpired:
        return False, "Timed out"
    except FileNotFoundError as e:
        return False, f"Tool not found: {e}"


# ── ReAct agent loop ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """

You are an expert systems programmer. Languages: Python, Fortran, C/C++, Bash, Makefiles, C#, Julia, Rust.

═══════════════════════════════════════════════════════
UNIVERSAL PRINCIPLES
═══════════════════════════════════════════════════════
- Read existing code before writing. Match style, naming, indentation exactly.
- Surgical edits only. Touch nothing outside the requirement.
- If ambiguous: surface interpretations, ask. Never guess silently.
- No features beyond what was asked. No abstraction for single-use code.
- Dead code: mention, do not delete unless orphaned by your own changes.
- Verify: mentally trace execution before outputting.

═══════════════════════════════════════════════════════
DEBUGGING REASONING PATTERN
═══════════════════════════════════════════════════════
1. Locate: exact file, line, symbol from error message.
2. Classify: syntax | type | linker | runtime | logic | UB.
3. Hypothesize: one cause at a time. Most likely first.
4. Minimal reproduction: isolate. Remove noise.
5. Fix: one change. Re-reason. Do not stack speculative fixes.
6. Compiler errors: read left-to-right, top-to-bottom. First error causes cascades — fix it first.

═══════════════════════════════════════════════════════
READING COMPILER ERRORS
═══════════════════════════════════════════════════════
GCC/Clang:  file:line:col: error: message   — col is where parser gave up, not necessarily cause
MSVC:       file(line): error CXXXX: message — error codes are searchable
Rustc:      error[EXXXX] — run `rustc --explain EXXXX` for full explanation
gfortran:   line N of MODULE — always check IMPLICIT NONE violations first
Linker:     "undefined reference to X" — X declared but not defined/linked; check object list and -l flags
            "multiple definition of X" — X defined in two TUs; use extern in header
Python:     Traceback bottom line = actual error; read upward for call chain
Julia:      MethodError = dispatch failed; check argument types, not just names

═══════════════════════════════════════════════════════
C / C++
═══════════════════════════════════════════════════════
GOTCHAS:
- UB is silent: signed overflow, out-of-bounds, null deref, use-after-free, uninitialized read.
- Strict aliasing: never cast T* to U* and dereference. Use memcpy or union.
- Integer promotion: `char a,b; a+b` is int. Beware truncation on assignment.
- VLAs (C99): stack-allocated, no bounds check, banned in C11 optionally and C++.
- #include order: own header first (catches missing deps), then system headers.
- ODR (C++): one definition per translation unit. inline, template, constexpr exempt.
- virtual destructor: if base class has virtual methods, destructor must be virtual.
- Copy/move: Rule of 0/3/5. If you define destructor, define copy and move too.
- `std::vector<bool>`: not a container of bool. Use `std::vector<char>` or `std::deque<bool>`.
- Narrowing in initializer lists `{}`  is an error in C++, not in C.
- `memset` for zero-init of POD only. Never on objects with vtable.
- `printf` format: `%zu` for size_t, `%td` for ptrdiff_t, `%p` for pointer.

IDIOMS:
- RAII: acquire in constructor, release in destructor. No naked new/delete in modern C++.
- `nullptr` not NULL in C++. NULL is 0 in disguise.
- Prefer `std::array` over C arrays; `std::string_view` over const char*.
- `[[nodiscard]]` on error-returning functions.
- `constexpr` for compile-time constants. Not `#define`.
- Range-for over index loop unless index needed.
- `emplace_back` over `push_back` for non-trivial objects.
- Smart pointers: `unique_ptr` default, `shared_ptr` only if ownership shared, `weak_ptr` to break cycles.

═══════════════════════════════════════════════════════
FORTRAN
═══════════════════════════════════════════════════════
MANDATORY FIRST LINE OF EVERY PROGRAM/MODULE/SUBROUTINE/FUNCTION:
  IMPLICIT NONE
Without it, undeclared variables starting I-N are implicitly INTEGER, rest REAL. Silent bugs.

ARRAY INDEXING: 1-based by default. `A(1)` is first element.
Custom bounds: `REAL :: A(0:N-1)` or `REAL :: A(-3:3)`.
Array slices: `A(2:5)`, `A(::2)` (stride 2), `A(::-1)` (reverse).
Whole-array ops: `A = B + C` vectorizes if same shape. No loop needed.
RESHAPE, TRANSPOSE, MATMUL, DOT_PRODUCT are intrinsics.

MODULES (modern Fortran, prefer over COMMON):
  MODULE mymod
    IMPLICIT NONE
    INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(15,307)  ! double precision
    REAL(dp), ALLOCATABLE :: buffer(:)
  CONTAINS
    SUBROUTINE init(n)
      INTEGER, INTENT(IN) :: n
      ALLOCATE(buffer(n))
    END SUBROUTINE
  END MODULE

USE mymod, ONLY: dp, init   ! explicit imports only

COMMON blocks (legacy — read, don't write):
  COMMON /blockname/ var1, var2   ! order and type must match across all TUs

INTENT: always declare INTENT(IN), INTENT(OUT), INTENT(INOUT) on dummy args.
ALLOCATABLE: check ALLOCATED() before use. DEALLOCATE when done.
KIND: never hardcode `REAL*8`. Use SELECTED_REAL_KIND or ISO_FORTRAN_ENV: REAL64.
FORMAT: fixed-form (.f) is column 7-72 only. Free-form (.f90) preferred.
EQUIVALENCE: avoid. It aliases memory like union but less safe.
DO loop var must not be modified inside loop body.
STRING: CHARACTER(LEN=N). Concatenation: `//`. Trim: TRIM(s), ADJUSTL(s).

GOTCHAS:
- Array passed to subroutine loses bounds unless passed as assumed-shape `(:)` with interface block or module.
- Assumed-size `(*)` disables whole-array ops. Avoid.
- Division of two integers is integer division. `1/2 = 0`. Cast explicitly: `1.0_dp/2`.
- Logical operators: `.AND.` `.OR.` `.NOT.` `.EQV.` `.NEQV.` — not && || !
- Comparison: `.EQ.` `.NE.` `.LT.` etc. or == /= < (modern). Never mix.
- RECURSIVE keyword required for recursive procedures.

═══════════════════════════════════════════════════════
PYTHON
═══════════════════════════════════════════════════════
GOTCHAS:
- Mutable default args: `def f(x=[])` — list shared across calls. Use `None` sentinel.
- Late binding closures: `[lambda: i for i in range(3)]` all return 2. Use `lambda i=i: i`.
- `is` tests identity, `==` tests equality. Never `x is 5` for int comparison (CPython caches -5..256 only).
- `float('nan') != float('nan')`. Use `math.isnan()`.
- `//` is floor division (rounds toward -inf), not truncation. `-7//2 == -4`, not -3.
- Generator exhaustion: generators are single-pass. `list()` to reuse.
- Dict ordering: preserved insertion order Python 3.7+. Do not rely on it for logic.
- `except Exception` catches almost everything. `except BaseException` catches SystemExit too — usually wrong.
- Deepcopy vs copy: nested mutable objects need `copy.deepcopy`.
- GIL: threads don't parallelize CPU-bound code. Use multiprocessing or asyncio.
- `__slots__`: reduces per-instance dict overhead. Declare explicitly if needed.

IDIOMS:
- Comprehensions over map/filter for readability.
- `enumerate(xs)` not `range(len(xs))`.
- `zip(a, b, strict=True)` (3.10+) to catch length mismatch.
- `@dataclass` for plain data containers.
- `pathlib.Path` over `os.path` string manipulation.
- `with open(...) as f` always. Never naked `open`.
- `typing.Protocol` for structural subtyping. No need to inherit.
- `__all__` in module to control `from m import *`.

TYPE HINTS:
- `list[int]` not `List[int]` (3.9+).
- `X | None` not `Optional[X]` (3.10+).
- `from __future__ import annotations` for forward refs in 3.7-3.9.

═══════════════════════════════════════════════════════
BASH
═══════════════════════════════════════════════════════
HEADER (every script):
  #!/usr/bin/env bash
  set -euo pipefail
  IFS=$'\n\t'

`-e`: exit on error. `-u`: error on unset var. `-o pipefail`: pipe fails if any stage fails.

GOTCHAS:
- Always quote variables: `"$var"` not `$var`. Unquoted = word split + glob expansion.
- `[[ ]]` not `[ ]`. `[[` is bash builtin, handles spaces, no word split.
- Arithmetic: `$(( a + b ))` not `$a + $b`. `let` is portable but `(( ))` is cleaner.
- `$()` not backticks. Nestable, readable.
- `local` variables in functions: always declare `local x=...` to avoid polluting global scope.
- Array: `arr=(a b c)`, element: `"${arr[0]}"`, all: `"${arr[@]}"`. Never `${arr[*]}` unless you know why.
- Process substitution: `diff <(cmd1) <(cmd2)`.
- `trap 'cleanup' EXIT` for guaranteed cleanup.
- `readonly` for constants. `export` only when subprocesses need it.
- Empty string test: `[[ -z "$var" ]]` not `[[ "$var" == "" ]]`.
- `&&` and `||` for short-circuit: `cmd1 && cmd2`. In `set -e`, `cmd || true` to allow failure.

IDIOMS:
- `${var:-default}` — use default if unset/empty.
- `${var:?error msg}` — abort if unset/empty.
- `${#arr[@]}` — array length.
- `${var%%pattern}` / `${var##pattern}` — strip suffix/prefix.
- `printf '%s\n' "$@"` over `echo` for portable output.
- `mktemp` for temp files. Clean up with trap.
- `read -r line` not `read line` — `-r` disables backslash interpretation.

═══════════════════════════════════════════════════════
MAKEFILES
═══════════════════════════════════════════════════════
STRUCTURE:
  # Variables
  CC      := gcc           # := immediate expand, = lazy expand
  CFLAGS  := -Wall -O2
  SRC     := $(wildcard src/*.c)
  OBJ     := $(SRC:src/%.c=build/%.o)

  # Default target (first rule = default)
  .PHONY: all clean test
  all: build/mybin

  # Link
  build/mybin: $(OBJ)
  	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)

  # Compile pattern rule
  build/%.o: src/%.c | build
  	$(CC) $(CFLAGS) -c $< -o $@

  # Order-only prerequisite: create dir if missing
  build:
  	mkdir -p $@

  clean:
  	rm -rf build/

AUTOMATIC VARIABLES:
  $@  — target name
  $<  — first prerequisite
  $^  — all prerequisites (no duplicates)
  $*  — stem of pattern rule (% match)
  $|  — order-only prerequisites

GOTCHAS:
- Recipe lines MUST use TAB, not spaces. Editor beware.
- Each recipe line runs in separate shell. Export vars with `export` or use `;` and `\` for multi-line.
- `.PHONY` targets never treated as files. Always declare for `all`, `clean`, `test`, `install`.
- `$(shell cmd)` runs at parse time, not recipe time. Side effects may surprise.
- `?=` sets only if not already set (env override friendly).
- `-include deps.mk` — leading `-` suppresses error if file missing (dep files on first build).

DEPENDENCY TRACKING:
  CFLAGS += -MMD -MP
  -include $(OBJ:.o=.d)
  # GCC generates .d files; -MP adds phony targets for headers to prevent error on header deletion.

FORTRAN PATTERN:
  FC      := gfortran
  FFLAGS  := -O2 -Wall -Wextra -fcheck=all -fimplicit-none
  build/%.o: src/%.f90
  	$(FC) $(FFLAGS) -c $< -o $@ -J build/   # -J: module .mod output dir

MODULE ORDER: Fortran .mod files must exist before USE. Encode dependency explicitly:
  build/main.o: build/mymod.o   # ensures mymod compiled first

═══════════════════════════════════════════════════════
RUST
═══════════════════════════════════════════════════════
GOTCHAS:
- Ownership: value has one owner. Move on assignment unless Copy. Clone explicitly.
- Borrow rules: any number of `&T` OR exactly one `&mut T`. Not both simultaneously.
- Lifetime elision handles most cases. Explicit `'a` only when compiler demands.
- `unwrap()` panics on None/Err. In production: `?` operator or match/if-let.
- Integer overflow: debug panics, release wraps. Use `checked_*`, `saturating_*`, `wrapping_*` explicitly.
- `String` vs `&str`: String is owned heap buffer. &str is borrowed slice. Function params: prefer `&str`.
- `Vec<T>` vs `&[T]`: similar — own vs borrow. Function params: prefer `&[T]`.
- Trait objects `dyn Trait`: heap alloc + vtable. Use generics `<T: Trait>` for monomorphization when possible.
- `derive(Clone, Debug, PartialEq)` for data types.
- `#[allow(dead_code)]` to silence locally. Fix globally.
- Closures capture by reference by default. Use `move ||` to force ownership capture.
- `Rc<T>` for single-thread shared ownership. `Arc<T>` for multi-thread.
- `Cell<T>` / `RefCell<T>` for interior mutability. RefCell panics at runtime on borrow violation.

IDIOMS:
- `?` for error propagation. Return `Result<T, E>` from fallible functions.
- `impl Display for T` before `impl Debug`. Debug is `{:?}`, Display is `{}`.
- Iterator chains: `map`, `filter`, `flat_map`, `fold`, `collect`. Lazy; call `collect()` to evaluate.
- `match` exhaustively. Compiler enforces. Never use `_` to silence unhandled arms unless truly don't care.
- `From`/`Into` for zero-cost conversions. Impl `From<X> for Y`, get `Into<Y> for X` free.
- `thiserror` crate for library errors. `anyhow` for application errors.
- `cargo clippy` — run it. Treat lints as errors in CI: `cargo clippy -- -D warnings`.

═══════════════════════════════════════════════════════
JULIA
═══════════════════════════════════════════════════════
GOTCHAS:
- 1-based indexing. `A[1]` is first. `end` is last: `A[end]`, `A[2:end]`.
- Type instability destroys performance. Functions must return consistent types. Use `@code_warntype f(args)`.
- Global variables are type-unstable by default. Use `const` for global constants.
- Multiple dispatch: method selected on all argument types. Ambiguity is a runtime error.
- Abstract type in function signature is fine for dispatch, bad for struct fields (kills performance).
  - Struct fields: use concrete types or parametric types `struct Foo{T<:Real}`.
- `copy` vs `deepcopy`: nested mutable → need deepcopy.
- Broadcasting: `f.(A)` applies f element-wise. `.+`, `.*` etc. Fused automatically.
- `@views A[2:5]` avoids copy. Use in hot loops.
- `@inbounds` removes bounds check. Only after verifying correctness.
- `@simd` for inner loops. Requires no loop-carried dependencies.
- String interpolation: `"value = $x"`, `"expr = $(x+1)"`.
- `nothing` ≠ `missing`. `nothing` = absence. `missing` = unknown (propagates in math).

IDIOMS:
- `@benchmark` (BenchmarkTools) for timing. Never use `@time` for microbenchmarks.
- `@show x` = prints `x = <value>` to stdout. Useful for debugging.
- `typeof(x)`, `eltype(A)`, `size(A)`, `axes(A)`.
- Modules: `module M; export f; end`. `using M` imports exported names. `import M: f` explicit.
- `let` block for local scope: `let x=1; ...; end`.
- Generators: `sum(x^2 for x in 1:n if isodd(x))` — no array allocation.

═══════════════════════════════════════════════════════
C#
═══════════════════════════════════════════════════════
GOTCHAS:
- `==` on reference types: identity by default. Override `Equals`/`GetHashCode` for value equality.
- `string` is reference type but `==` is overloaded for value comparison. OK.
- `struct` is value type. Passed by copy. Mutating struct fields through interface = copy problem.
- `async void` only for event handlers. Always `async Task` or `async Task<T>`.
- `ConfigureAwait(false)` in library code to avoid deadlocks on sync context.
- `IDisposable`: implement for unmanaged resources. Always `using` or `using var` at call site.
- `null` everywhere pre-C#8. Enable `<Nullable>enable</Nullable>` in project; treat warnings as errors.
- `?.` null-conditional and `??` null-coalescing are your friends.
- Boxing: casting value type to `object` or interface allocates heap. Hot paths: avoid.
- `List<T>` not `ArrayList`. `Dictionary<K,V>` not `Hashtable`. Never use non-generic collections.
- `readonly` field: set in constructor only. `const`: compile-time literal only.
- LINQ deferred execution: `Where`/`Select` return `IEnumerable`, not evaluated yet. `.ToList()` to force.

IDIOMS:
- `record` types (C#9+) for immutable data with structural equality.
- `var` when type obvious from RHS. Explicit type when it adds clarity.
- Pattern matching: `switch (x) { case int n when n > 0: ... }` or `x is int n && n > 0`.
- `Span<T>` / `Memory<T>` for zero-copy slicing.
- `StringBuilder` for concatenation in loops. String concat in loop = O(n²).
- Expression-bodied members: `public int X => _x;` for simple properties/methods.
- `nameof(symbol)` not string literals for member names (refactor-safe).
- Init-only properties: `public int X { get; init; }` for immutable after construction.

═══════════════════════════════════════════════════════
SURGICAL EDIT PROTOCOL
═══════════════════════════════════════════════════════
1. Read the existing file fully.
2. Identify minimal diff to satisfy requirement.
3. Match: indentation, brace style, naming convention, comment style.
4. Do NOT: reformat, rename, add logging, add comments, restructure unless asked.
5. If removing a function/symbol, check all call sites. List any orphaned references.
6. Output only changed sections with enough context (3-5 lines) to locate them.
7. State what changed and why in one sentence. No essays.



You are an intelligent assistant and coding agent. You have access to tools.

For each step, respond with ONE action in EXACTLY this format (no extra text before the action line):

ACTION: <action>
...parameters...

Available actions:

ACTION: answer
CONTENT:
<your answer to the user, markdown ok>
---END---

ACTION: write
FILE: <relative path, no workspace/ prefix>
CONTENT:
<raw file content, NO markdown fences>
---END---

ACTION: read
FILE: <relative path>

ACTION: read_lines
FILE: <relative path>
START: <line number>
END: <line number>

ACTION: run
FILE: <relative path>

ACTION: delete
FILE: <relative path>

ACTION: run_cmd
CMD: <shell command, runs in workspace dir>

ACTION: list

ACTION: search
QUERY: <web search query>

ACTION: done

Rules:
- Use 'answer' for questions, explanations, chat — no code execution needed.
- Use 'write' then 'run' for coding tasks. After run succeeds, use 'done'.
- Use 'read' for files — large files are auto-truncated to 200 lines with total line count shown. Use 'read_lines' to page through the rest.
- Use 'read_lines' to read a specific line range of a file. Max 1000 lines per call — observation will tell you how many lines remain.
- Use 'run_cmd' for shell commands: make, gfortran, bash scripts, ls, cat, etc. Runs in workspace dir.
- This is a Linux environment. Never use Windows-style filenames or extensions (.exe, .bat, .cmd, .ps1). Compiled binaries have no extension (e.g. `gfortran heat.f90 -o heat`).
- To install missing Python packages, use `uv pip install <pkg>`. If the install fails or the module is still missing, retry up to 3 times total — then stop and report the error.
- Use 'list' to discover all files in the workspace when you need to find what's available.
- PDF files are supported — use 'read' on them directly, text will be extracted automatically.
- Use 'delete' to remove files.
- Use 'search' only when you need current/external information.
- Use 'done' to end the loop after completing a multi-step task.
- NEVER wrap file content in markdown fences.
- File paths relative to workspace root only.
"""


def _react_loop(query, llm, history, vectorstore=None, force_web=False, max_steps=10):
    """Main ReAct loop. Returns final answer string."""
    from websearch import web_search as do_web_search

    history_text = format_history_for_prompt(history, max_turns=8)
    summary = history[-1].get("summary", "") if history else ""
    history_block = ""
    if summary:
        history_block += f"Conversation summary so far:\n{summary}\n\n"
    if history_text:
        history_block += f"Recent turns:\n{history_text}\n\n"
    workspace_files = _list_workspace()

    context = (
        f"{history_block}Workspace files: {workspace_files}\n\nUser request: {query}"
    )

    if vectorstore is not None:
        try:
            rag_docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
            if rag_docs:
                rag_text = "\n\n".join(
                    f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                    for d in rag_docs
                )
                context += f"\n\nRetrieved context from research library (use only if relevant):\n{rag_text}"
                print("  📚 RAG context injected.", flush=True)
        except Exception as e:
            print(f"  ⚠ RAG retrieval failed: {e}", flush=True)

    if force_web:
        print("  🌐 Searching web...", flush=True)
        web_text, _ = do_web_search(query)
        if web_text:
            context += f"\n\nWeb search results:\n{web_text}"

    messages = [HumanMessage(content=f"{SYSTEM_PROMPT}\n\n{context}")]

    final_answer = None
    _react_loop._did_execute = False
    pending_actions = []  # queue of (action, result, lines) tuples
    _failed_cmd = None        # last run_cmd that failed, eligible for auto-retry
    _failed_cmd_retries = 0   # how many times it has been retried

    for step in range(1, max_steps + 1):
        if pending_actions:
            action, result, lines = pending_actions.pop(0)
            print(f"  🤖 Step {step} (queued: {action})...", flush=True)
            # obs will be set by the action handler below; capture it for failure check
            _queued_step = True
        else:
            _queued_step = False
            print(f"  🤖 Step {step}...", flush=True)
            if step == 1:
                invoke_messages = messages[:-1] + [
                    HumanMessage(content=f"<|think|>\n{messages[-1].content}")
                ]
            else:
                invoke_messages = messages
            result = llm.invoke(invoke_messages).content
            lines = result.strip().splitlines()
            action_line = next((l for l in lines if l.startswith("ACTION:")), None)
            action = action_line.split(":", 1)[1].strip().lower() if action_line else None
            if action != "answer":
                console.print(Markdown(result))
            messages.append(HumanMessage(content=f"ASSISTANT: {result}"))
            if not action_line:
                final_answer = result
                break

            # queue any additional actions found in same response
            all_action_indices = [i for i, l in enumerate(lines) if l.startswith("ACTION:")]
            for idx in all_action_indices[1:]:
                extra_action = lines[idx].split(":", 1)[1].strip().lower()
                # grab lines from this ACTION: to next ACTION: or end
                next_idx = all_action_indices[all_action_indices.index(idx) + 1] if all_action_indices.index(idx) + 1 < len(all_action_indices) else len(lines)
                extra_lines = lines[idx:next_idx]
                pending_actions.append((extra_action, result, extra_lines))

        # track whether any files were written or commands run this session
        obs = ""
        if action == "answer":
            try:
                start = result.index("CONTENT:\n") + len("CONTENT:\n")
                end = result.index("\n---END---")
                final_answer = result[start:end].strip()
            except ValueError:
                final_answer = result

            # pushback: if nothing was executed yet, don't accept answer
            if not _react_loop._did_execute:
                print(
                    "  ⚠ Model answered without executing — pushing back.", flush=True
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "OBSERVATION: You described what to do but did not do it. "
                            "No files were written and no commands were run. "
                            "You MUST use ACTION: write to create files and ACTION: run_cmd to execute them. "
                            "Stop explaining. Act now."
                        )
                    )
                )
                final_answer = None
                continue
            break

        elif action == "done":
            if final_answer is None:
                final_answer = "Done."
            break

        elif action == "write":
            fp, content = _extract_write(result, lines)
            if fp and content is not None:
                abs_path = _write_file(fp, content)
                print(f"  ✓ Written: {abs_path}", flush=True)
                obs = f"File written: {fp}"
                _react_loop._did_execute = True
            else:
                obs = "ERROR: could not parse write action"
                print(f"  ✗ {obs}", flush=True)
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "read":
            fp = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("FILE:")),
                None,
            )
            if fp:
                if fp.startswith("workspace/"):
                    fp = fp[len("workspace/") :]
                if fp.lower().endswith(".pdf"):
                    try:
                        from langchain_community.document_loaders import PyPDFLoader

                        abs_pdf = str(_safe_path(fp))
                        docs = PyPDFLoader(abs_pdf).load()
                        content = "\n\n".join(d.page_content for d in docs)
                        obs = (
                            f"PDF content of {fp}:\n{content}"
                            if content
                            else f"PDF empty or unreadable: {fp}"
                        )
                    except Exception as e:
                        obs = f"ERROR reading PDF {fp}: {e}"
                else:
                    content = _read_file(fp)
                    if content is None:
                        obs = f"File not found: {fp}"
                    else:
                        file_lines = content.splitlines()
                        total = len(file_lines)
                        TRUNC_LINES = 200
                        if total > TRUNC_LINES:
                            shown = "\n".join(file_lines[:TRUNC_LINES])
                            obs = (
                                f"File content of {fp} (lines 1-{TRUNC_LINES} of {total}):\n{shown}\n"
                                f"[truncated — use read_lines to read further: ACTION: read_lines / FILE: {fp} / START: N / END: M]"
                            )
                        else:
                            obs = f"File content of {fp} ({total} lines):\n{content}"
            else:
                obs = "ERROR: no FILE specified"
            print(f"  📖 Read: {fp}", flush=True)
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "read_lines":
            fp = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("FILE:")),
                None,
            )
            start = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("START:")),
                None,
            )
            end = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("END:")),
                None,
            )
            if fp:
                if fp.startswith("workspace/"):
                    fp = fp[len("workspace/") :]
                content = _read_file(fp)
                if content is None:
                    obs = f"File not found: {fp}"
                else:
                    file_lines = content.splitlines()
                    total = len(file_lines)
                    try:
                        s = max(0, int(start) - 1) if start else 0
                        e = int(end) if end else s + 200
                    except ValueError:
                        s, e = 0, 200
                    # enforce 1000 line cap
                    if e - s > 1000:
                        e = s + 1000
                    e = min(e, total)
                    chunk = "\n".join(file_lines[s:e])
                    remaining = total - e
                    note = (
                        f"\n[{remaining} more lines remain — call read_lines again if needed]"
                        if remaining > 0
                        else "\n[end of file]"
                    )
                    obs = f"File {fp} lines {s + 1}-{e} of {total}:\n{chunk}{note}"
                    print(f"  📖 Read lines {s + 1}-{e} of {fp}", flush=True)
            else:
                obs = "ERROR: no FILE specified"
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "run":
            fp = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("FILE:")),
                None,
            )
            if fp:
                if fp.startswith("workspace/"):
                    fp = fp[len("workspace/") :]
                abs_path = str(_safe_path(fp))
                print(f"  ▶ Running: {fp}", flush=True)
                success, output = _run_file(abs_path)
                status = "SUCCESS" if success else "FAILED"
                obs = f"Run {status}:\n{output}"
                print(f"  {'✅' if success else '❌'} {status}: {output}", flush=True)
            else:
                obs = "ERROR: no FILE specified"
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "delete":
            fp = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("FILE:")),
                None,
            )
            if fp:
                if fp.startswith("workspace/"):
                    fp = fp[len("workspace/") :]
                deleted = _delete_file(fp)
                obs = f"Deleted: {fp}" if deleted else f"File not found: {fp}"
                print(f"  🗑 {obs}", flush=True)
            else:
                obs = "ERROR: no FILE specified"
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "list":
            files = _list_workspace()
            obs = (
                "Workspace files:\n" + "\n".join(files)
                if files
                else "Workspace is empty."
            )
            print(f"  📂 Listed {len(files)} files", flush=True)
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "run_cmd":
            cmd = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("CMD:")),
                None,
            )
            if cmd:
                print(f"  ⚙ Running: {cmd}", flush=True)
                try:
                    r = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=str(SANDBOX),
                    )
                    full_out = (r.stdout + r.stderr).strip() or "(no output)"
                    status = (
                        "SUCCESS"
                        if r.returncode == 0
                        else f"FAILED (exit {r.returncode})"
                    )

                    # write full output to log file
                    rand_id = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=6)
                    )
                    log_dir = SANDBOX / ".logs"
                    log_dir.mkdir(exist_ok=True)
                    log_name = f"{rand_id}.log"
                    log_path = log_dir / log_name
                    log_path.write_text(
                        f"CMD: {cmd}\n"
                        f"TIME: {datetime.datetime.now().isoformat()}\n"
                        f"STATUS: {status}\n"
                        f"{'=' * 60}\n"
                        f"{full_out}"
                    )

                    # truncate for context: last 3000 chars
                    TRUNC = 3000
                    if len(full_out) > TRUNC:
                        tail = full_out[-TRUNC:]
                        obs = (
                            f"run_cmd {status}:\n"
                            f"[output truncated — full log: .logs/{log_name}]\n"
                            f"...{tail}"
                        )
                    else:
                        obs = f"run_cmd {status}:\n{full_out}\n[log: .logs/{log_name}]"

                    print(
                        f"  {'✅' if r.returncode == 0 else '❌'} {status} — log: .logs/{log_name}",
                        flush=True,
                    )
                    _react_loop._did_execute = True

                    _INSTALL_PREFIXES = (
                        "uv pip install", "pip install", "pip3 install",
                        "npm install", "yarn add",
                        "cargo install",
                        "apt install", "apt-get install",
                        "brew install",
                    )
                    if r.returncode != 0:
                        # store failed command for potential auto-retry after install
                        _failed_cmd = cmd
                    elif any(cmd.strip().startswith(p) for p in _INSTALL_PREFIXES) and _failed_cmd and _failed_cmd_retries < 3:
                        # install succeeded — re-queue the command that originally failed
                        _failed_cmd_retries += 1
                        retry_lines = [f"CMD: {_failed_cmd}"]
                        pending_actions.insert(0, ("run_cmd", f"ACTION: run_cmd\nCMD: {_failed_cmd}", retry_lines))
                        print(f"  🔁 Install succeeded — retrying: {_failed_cmd} (attempt {_failed_cmd_retries}/3)", flush=True)
                    elif r.returncode == 0 and cmd == _failed_cmd:
                        # the retried command succeeded — clear tracking
                        _failed_cmd = None
                        _failed_cmd_retries = 0
                except subprocess.TimeoutExpired:
                    obs = "run_cmd FAILED: timed out after 120s"
                except Exception as e:
                    obs = f"run_cmd ERROR: {e}"
            else:
                obs = "ERROR: no CMD specified"
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        elif action == "search":
            sq = next(
                (l.split(":", 1)[1].strip() for l in lines if l.startswith("QUERY:")),
                query,
            )
            print(f"  🌐 Searching: {sq}", flush=True)
            web_text, _ = do_web_search(sq)
            obs = f"Search results:\n{web_text}" if web_text else "No results found."
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        else:
            obs = f"Unknown action: {action}"
            messages.append(HumanMessage(content=f"OBSERVATION: {obs}"))

        # if this was a queued step and it failed, flush remaining queue so LLM recovers
        if _queued_step and pending_actions and ("FAILED" in obs or "ERROR" in obs):
            if _failed_cmd and _failed_cmd_retries >= 3:
                print(f"  ✗ Retry limit reached for: {_failed_cmd} — giving up.", flush=True)
                messages.append(HumanMessage(content=f"OBSERVATION: Retry limit (3) reached for `{_failed_cmd}`. Stopping retries."))
                _failed_cmd = None
                _failed_cmd_retries = 0
            print(f"  ⚠ Queued action failed — flushing {len(pending_actions)} pending action(s), handing to LLM.", flush=True)
            pending_actions.clear()

    if final_answer is None:
        final_answer = f"Agent reached max steps ({max_steps}) without finishing."

    return final_answer


# ── explicit override handlers ───────────────────────────────────────────────


def _handle_edit(task, llm):
    """Single-step file edit — kept as explicit override."""
    files = _list_workspace()
    workspace_context = f"Workspace files: {files}\n\n"
    for f in files:
        if f in task:
            content = _read_file(f)
            workspace_context += f"--- {f} ---\n{content}\n\n"
    prompt = (
        f"{workspace_context}"
        f"Task: {task}\n\n"
        "File paths must be relative to workspace root. No 'workspace/' prefix.\n"
        "No markdown fences in content.\n"
        "Respond with ONE action:\n"
        "ACTION: write\nFILE: <path>\nCONTENT:\n<content>\n---END---\n\n"
        "or\n\nACTION: read\nFILE: <path>"
    )
    result = llm.invoke([HumanMessage(content=prompt)]).content
    print(f"\n{result}")
    lines = result.strip().splitlines()
    action = next(
        (l.split(":", 1)[1].strip() for l in lines if l.startswith("ACTION:")), None
    )
    fp = next(
        (l.split(":", 1)[1].strip() for l in lines if l.startswith("FILE:")), None
    )
    if fp and fp.startswith("workspace/"):
        fp = fp[len("workspace/") :]
    if action == "write" and fp:
        efp, content = _extract_write(result)
        if efp and content is not None:
            saved = _write_file(efp, content)
            print(f"\n  ✓ Written: {saved}")
        else:
            print("  ✗ Parse error")
    elif action == "read" and fp:
        print(_read_file(fp) or "  File not found.")


# ── bootstrap ────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--no-rag", action="store_true", help="Skip vectorstore, direct LLM only"
    )
    return p.parse_args()


def choose_model():
    print("\nAvailable models:")
    for key, (name, desc) in AVAILABLE_MODELS.items():
        print(f"  [{key}] {name} — {desc}")
    while True:
        choice = input("\nChoose model (default 6 = qwen2.5-coder:7b): ").strip()
        if not choice:
            return "qwen2.5-coder:7b"
        if choice in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[choice][0]
        print("  Invalid choice, try again.")


if __name__ == "__main__":
    args = parse_args()

    print("--- QpiVOLTA Research Brain Initializing ---")

    if not args.no_rag:
        print(f"Watching {len(SOURCE_DIRS)} source folder(s):")
        for d in SOURCE_DIRS:
            print(f"  • {d}")

    model_name = choose_model()
    print(f"\n✓ Using model: {model_name}")

    llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=32768)
    vectorstore = None

    if not args.no_rag:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        from indexer import update_research_library
        from rag import get_answer

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        print()
        if AUTO_INDEX:
            update_research_library(vectorstore)
        else:
            print("✓ Skipping indexing (AUTO_INDEX not set)")
    else:
        print("✓ RAG disabled — direct LLM mode")

    history, convo_file = load_conversation()

    print(f"\n--- Ready. Model: {model_name} ---")
    print("  <anything>         → agent decides: answer / write / run / read / search")
    print("  /w <question>      → force web search first")
    print("  /edit <task>       → single-step file edit")
    if not args.no_rag:
        print("  /p <question>      → web search + download papers + RAG")
    print("  convos             → list & load saved conversations")
    print("  exit               → save and quit")
    print()
    print(f"  Workspace: {WORKSPACE_DIR}")

    while True:
        query = input("\n> ").strip()
        if not query:
            continue

        if query.lower() == "exit":
            convo_file = save_conversation(history, convo_file, model_name)
            if convo_file:
                print(f"💾 Conversation saved to {convo_file}")
            break

        if query.lower() == "convos":
            list_conversations()
            continue

        if query.lower().startswith("/edit"):
            _handle_edit(query[5:].strip(), llm)
            continue

        # explicit overrides
        force_web = query.startswith("/w")
        download_mode = query.startswith("/p") and not args.no_rag
        if force_web:
            query = query[2:].strip()
        elif download_mode:
            query = query[2:].strip()

        mode = "📄" if download_mode else ("🌐" if force_web else "🤖")
        print(f"\n[{model_name} {mode}]...", flush=True)

        try:
            if download_mode and vectorstore:
                # /p: web search + download + RAG (keep original flow)
                answer, web_results = get_answer(
                    query, vectorstore, llm, history, use_web=True
                )
                console.print(Markdown(answer))
                if web_results:
                    paper_results = search_papers(query)
                    all_results = web_results + [
                        r for r in paper_results if r not in web_results
                    ]
                    downloaded = interactive_paper_download(all_results)
                    if downloaded:
                        print(f"\n  📚 Indexing {len(downloaded)} paper(s)...")
                        update_research_library(vectorstore, extra_files=downloaded)
            else:
                # agent path for all other queries (RAG + no-RAG)
                answer = _react_loop(
                    query, llm, history, vectorstore=vectorstore, force_web=force_web
                )
                console.print(Markdown(answer))
                web_results = []

            # compress history into rolling summary every 4 turns
            if len(history) > 0 and len(history) % 4 == 0:
                try:
                    summary_prompt = (
                        "Summarize this conversation concisely, preserving all key facts, "
                        "decisions, code written, and context needed to continue it.\n\n"
                        + "\n".join(
                            f"User: {t['question']}\nAssistant: {t['answer']}"
                            for t in history
                        )
                    )
                    summary_text = llm.invoke(
                        [HumanMessage(content=summary_prompt)]
                    ).content
                    history.append(
                        {"question": query, "answer": answer, "summary": summary_text}
                    )
                except Exception:
                    history.append({"question": query, "answer": answer})
            else:
                history.append({"question": query, "answer": answer})
            convo_file = save_conversation(history, convo_file, model_name)

        except Exception as e:
            print(f"An error occurred: {e}")
