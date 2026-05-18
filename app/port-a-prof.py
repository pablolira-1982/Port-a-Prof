from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import httpx
import json
import re
import base64
import tempfile
import os
import traceback
import asyncio
import sys
from pathlib import Path
import textwrap

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── CONFIG ───────────────────────────────────────────────────────────
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_COMPLETION_URL = f"{LLAMA_BASE_URL}/completion"           # raw — used in teaching session (with thinking tokens)
LLAMA_CHAT_URL       = f"{LLAMA_BASE_URL}/v1/chat/completions"  # OpenAI-compatible — used for vision, audio, title generation
MODEL_NAME = "Port-a-Prof"
APP_TITLE  = "Port-a-Prof"

_http_client = None

# ── DEBUG LOGGING ────────────────────────────────────────────────────
DEBUG = True   # set to False to silence all debug outputs!

class C:
    """ANSI color codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    # labels
    CYAN    = "\033[96m"   # section headers
    YELLOW  = "\033[93m"   # prompts
    GREEN   = "\033[92m"   # raw model output
    MAGENTA = "\033[95m"   # parsed / structured data
    BLUE    = "\033[94m"   # inputs from client
    RED     = "\033[91m"   # errors / warnings
    DIM     = "\033[2m"    # dividers

_DIVIDER = f"{C.DIM}{'─' * 36}{C.RESET}"

def dbg(label: str, content: str, color: str = C.CYAN) -> None:
    """Prints a labelled debug block to stderr."""
    if not DEBUG:
        return
    header = f"{color}{C.BOLD}▶  {label}{C.RESET}"
    wrapped = "\n".join(
        textwrap.fill(l, width=60, initial_indent="  ", subsequent_indent="  ")
        if l.strip() else ""
        for l in content.splitlines()
    )
    print(f"\n{_DIVIDER}", file=sys.stderr)
    print(header, file=sys.stderr)
    print(_DIVIDER, file=sys.stderr)
    print(wrapped, file=sys.stderr)
    print(file=sys.stderr)


# ── Teacher prompts ─────────────────────────────────────────────────────
#
# Call order on each turn:
#   Phase 0  (first turn only) — build_solution_steps_prompt
#   Phase 1  (every turn)      — build_status_and_role_prompt  
#   Phase 2  (every turn)      — build_teacher_prompt


# ── Phase 0: internal solution steps (upon session entry only) ───────────

def build_solution_steps_prompt(problem: str) -> str:
    inner = f"""
You are an expert problem solver. Produce a concise, internal solution for the problem below.

### PROBLEM
{problem}

Return JSON only with this schema:
{{"solution_steps": ["1. step one", "2. step two", "..."]}}

Rules:
- Number each step "1.", "2.", "3.", etc.
- Each step is one logical operation or calculation, shown explicitly with working.
- The final step must include the final answer naturally as part of the calculation or conclusion. Do not add a separate step restating it.
- Write expressions in their simplest equivalent form. Omit zero terms, drop trivial operations (e.g. +0, *1), and reduce before presenting.
- Do not explain or tutor — this is a reference only.
- Output valid JSON only. No markdown."""
    return f"""<bos><|turn>user
{inner}<turn|>
<|turn>model
"""


# ── Phase 1: zero-shot student status diagnosis + teacher role selection ─────

def build_status_and_role_prompt(
    problem: str,
    solution_steps: str,
    student_input: str,
    prior_status: str = "",
    prior_teacher_reply: str = "",
    first_turn: bool = True,
) -> str:
    context_block = "" if first_turn else f"""
## CONTEXT (reference only)
PRIOR_STATUS: {prior_status}
PRIOR_TEACHER_REPLY: "{prior_teacher_reply}"
"""
    label = "STUDENT_COMMENT" if first_turn else "STUDENT_REPLY"
    status_rule = "" if first_turn else "- Base student_status on STUDENT_REPLY only. You must not link to the PRIOR_STATUS, or praise or affirm the student for any information provided in PRIOR_TEACHER_REPLY."

    inner = f"""You are a diagnostic controller for an AI teacher. Diagnose the student and select a response role.

## PROBLEM
{problem}

## SOLUTION STEPS (ground truth — reference only)
{solution_steps}
{context_block}
## {label}
"{student_input}"

## OUTPUT 
Return JSON only with this schema:
{{"student_status": "", "role": ""}}

IMPORTANT:
Before writing `student_status`, perform a step-by-step audit of the student's expression against `## SOLUTION STEPS`.

For EVERY term or value in the student's dialouge:
1. Identify which reference term or value in `## SOLUTION STEPS` it corresponds to.
2. Verify that the value, sign, coefficient, operator, and exponent all match exactly.
3. Treat any misplaced value, missing coefficient, incorrect sign, incorrect exponent, or incorrect operation as an error, even if the number itself appears elsewhere in the correct solution.

Do not judge correctness based on numerical similarity alone. A correct number used in the wrong structure is still incorrect.

student_status: 1-2 sentences, third person. Describe what the student demonstrated, attempted, or expressed in ## {label}. Compare all values term-by-term to the relevant step in ## SOLUTION STEPS to confirm correctness or identify specific errors, including correct adherence to formulas. Do not infer full understanding from partial or vague statements. Only mark a concept as understood if the student explicitly demonstrates it. Mention what remains unresolved or ambiguous. If input is blank, the student does not know where to begin. 
role — choose exactly one:
- session_close: student's answer matches the final answer as per ## SOLUTION STEPS
- inject_info: student has a knowledge gap that must be filled before proceeding
- confirm_and_advance: student's latest step is correct but the problem is not yet complete
- partial_worked_step: student asks for help setting up or carrying out the next step
- redirect: student made an error that must be corrected before proceeding. 

Rules:
{status_rule}
- If any incorrect calculation or reasoning is present, redirect takes precedence over all other roles including session_close and confirm_and_advance.
- If the student's reply contains the final solution value (equivalent to the result of the last entry in ## SOLUTION STEPS), even embedded in a larger expression, the role must be session_close.
- A student expression is correct if it is mathematically equivalent to the expected step, even if simplified. Omitting zero terms, skipping trivial operations, or reordering commutative operations are all valid. Do not flag missing terms whose value is zero or whose omission does not change the result. 
- Compare the signs of all values used by the student to the signs of values in ## SOLUTION STEPS. If there is any sign mismatch, the role must be redirect.
- Return ONLY the JSON object. Output nothing after the closing brace."""

    return f"""<bos><|turn>user
{inner}<turn|>
<|turn>model
"""


# ── Phase 2: teacher reply, input is same format as QLoRA fine-tuning  ────────

def build_teacher_prompt(solution_steps, problem, student_attempt, current_status, role, support):

    support_block = ""
    if role == "partial_worked_step" and support:
        support_block = f"""

## SUPPORT
{support}"""

    return f"""<bos><|turn>user
    
## PROBLEM
{problem}

## STUDENT_ATTEMPT
{student_attempt}

## STATUS
{current_status}

## TEACHER_ROLE
{role}{support_block}<turn|>
<|turn>model
"""


def extract_json(content):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        fallback = re.search(r"\{.*?\}", content, re.DOTALL)
        if fallback:
            cleaned = fallback.group(0)
        else:
            raise ValueError("No JSON object found")
    return json.loads(cleaned)


# ── Pre-session checks ────────────────────────────────────────────────
#
#  1. build_problem_validation_prompt  — is this a valid academic problem?
#  2. call_llama_vision                — image → extracted problem text
#  Endpoints: POST /validate-problem  and  POST /extract-problem

def build_problem_validation_prompt(problem: str) -> str:
    inner = f"""You are a validation assistant for an AI teachering system.
Classify the input below into exactly one of three categories.

INPUT:
{problem}

Categories:
- "valid": a deterministic problem with a clear correct answer — mathematics, science, logic, coding, grammar rules, factual history, etc.
- "non_deterministic": a genuine academic question but with no single correct answer — essay writing, literary analysis, creative writing, opinion-based humanities, ethical debates, etc.
- "invalid": gibberish, random characters, greetings, off-topic chat, offensive content, or not a question at all.

Return JSON only: {{"result": "valid"}} or {{"result": "non_deterministic"}} or {{"result": "invalid"}}
No markdown, no extra text."""
    return f"""<bos><|turn>user
{inner}<turn|>
<|turn>model
"""


async def call_llama_vision(image_b64: str, mime_type: str, text_prompt: str, max_tokens: int = 512) -> str:
    """Send an image + text prompt to Gemma via /v1/chat/completions (vision path)."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": False
    }
    resp = await _http_client.post(LLAMA_CHAT_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


async def call_llama_completion(
    prompt: str,
    n_predict: int = 512,
    temperature: float = 0.0,
    thinking: bool = False,
    label: str = "PROMPT",
    color: str = C.YELLOW,
) -> str:
    THINK_OPEN  = "<|channel>thought"
    THINK_CLOSE = "<channel|>"

    if thinking:
        prompt    = prompt.rstrip() + THINK_OPEN + "\n"
        n_predict = max(n_predict, 1024)

    thinking_tag = f"  {C.GREEN}(thinking enabled){C.RESET}" if thinking else f"  {C.DIM}(thinking disabled){C.RESET}"
    dbg(f"{label}{thinking_tag}", prompt, color)

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "n_predict": n_predict,
        "seed": 42,
        "stop": ["<turn|>", "<|turn>user"],
        "stream": True,   
    }

    full_text = ""
    print(f"\n{C.GREEN}", end="", file=sys.stderr, flush=True)   # raw model output is green

    async with _http_client.stream("POST", LLAMA_COMPLETION_URL, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            token = chunk.get("content", "")
            full_text += token
            print(token, end="", file=sys.stderr, flush=True)   # real-time printing
            if chunk.get("stop"):
                break

    print(C.RESET, file=sys.stderr, flush=True)   # reset color after done

    raw = full_text

    if thinking and THINK_CLOSE in raw:
        _, _, raw = raw.partition(THINK_CLOSE)
        raw = raw.strip()

    return raw


# ── Lifespan ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=180.0)


@app.on_event("shutdown")
async def shutdown_event():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()


# ── Static / index ───────────────────────────────────────────────────

@app.get("/logo.png")
async def serve_logo():
    return FileResponse("logo.png", media_type="image/png")

@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path("index.html").read_text(encoding="utf-8")
    return html.replace("__MODEL_NAME__", MODEL_NAME).replace("__APP_TITLE__", APP_TITLE)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("logo.png", media_type="image/x-icon")

@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    return FileResponse("logo.png", media_type="image/png")

@app.get("/apple-touch-icon-precomposed.png")
async def apple_touch_icon_precomposed():
    return FileResponse("logo.png", media_type="image/png")


# ── Transcription (Gemma 4 audio via llama-server) ───────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    tmp_in  = None
    tmp_wav = None
    try:
        # Save incoming audio (webm on Chrome, mp4 on Safari — ffmpeg handles both)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
            while chunk := await audio.read(1024 * 1024):
                f.write(chunk)
            tmp_in = f.name

        # Convert to 16kHz mono WAV 
        tmp_wav = tmp_in + ".wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_wav,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg audio→wav conversion failed")

        # Base64-encode the WAV
        with open(tmp_wav, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        # Send to Gemma via llama-server
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": "Transcribe this audio exactly. Return only the spoken words, nothing else."},
                ],
            }],
            "max_tokens": 512,
            "temperature": 0,
            "stream": False,
        }

        resp = await _http_client.post(LLAMA_CHAT_URL, json=payload)
        resp.raise_for_status()
        transcript = resp.json()["choices"][0]["message"]["content"].strip()
        dbg("TRANSCRIBE", transcript, C.GREEN)
        return {"transcript": transcript}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for p in (tmp_in, tmp_wav):
            if p and os.path.exists(p):
                os.unlink(p)


# ── Pre-session: problem validation ──────────────────────────────────

@app.post("/validate-problem")
async def validate_problem(problem_text: str = Form(...)):
    """Check whether the submitted text is a genuine academic problem."""
    try:
        problem = problem_text.strip()
        if not problem:
            return {"result": "invalid"}

        prompt = build_problem_validation_prompt(problem)
        raw = await call_llama_completion(prompt, n_predict=128, temperature=0, thinking=False, label="VALIDATE PROBLEM PROMPT")

        parsed = extract_json(raw)
        result = parsed.get("result", "invalid")
        dbg("VALIDATE PROBLEM RESULT", f"result={result}", C.MAGENTA)
        return {"result": result}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Pre-session: image problem extraction ────────────────────────────
#
# increase max_px or quality if problem text is not extracting accurately
def compress_image(data: bytes, max_px: int = 500, quality: int = 60) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(data))
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


@app.post("/extract-problem")
async def extract_problem(image: UploadFile = File(...)):
    """Read an uploaded image and extract any problem/question visible in it."""
    try:
        contents          = await image.read()
        contents, mime_type = compress_image(contents)
        image_b64         = base64.b64encode(contents).decode("utf-8")

        dbg("EXTRACT PROBLEM", f"mime={mime_type}  bytes={len(contents)}", C.BLUE)

        vision_prompt = (
            "Look at this image and extract the exact problem or question shown.\n"
            "Return JSON only:\n"
            '{"found": true, "problem": "the full extracted problem text"}\n'
            "If no clear problem or question is visible, return:\n"
            '{"found": false, "problem": null}\n'
            "No markdown, no extra text."
        )

        raw    = await call_llama_vision(image_b64, mime_type, vision_prompt, max_tokens=1024)

        parsed  = extract_json(raw)
        found   = bool(parsed.get("found", False))
        problem = parsed.get("problem") if found else None
        dbg("EXTRACT PROBLEM RESULT", f"found={found}  problem={problem}", C.MAGENTA)
        return {"found": found, "problem": problem}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Teacher endpoint ────────────────────────────────────────────────────
# Call chain per turn:
#   Phase 0  (first turn only) — generate solution steps; thinking always on (ground truth reference for teacher; accuracy is crucial)
#   Phase 1  (every turn)      — diagnose student status + select role;
#                                thinking controlled by user via home-screen toggle
#   Phase 2  (every turn)      — generate teacher reply
#
# teacher roles: inject_info · confirm_and_advance · partial_worked_step · redirect · session_close
# partial_worked_step tracks a consecutive count of how many turns the student has
# requested help on the same step — support escalates low → medium → high each consecutive turn
#
# session_state is a dict holding the problem, solution steps, prior student status,
# last teacher reply, and partial_worked_step count — passed back by the frontend each turn.
# On the first turn it is null; the client sends problem_text and student_comment instead.

@app.post("/teacher")
async def teacher(
    session_state: str     = Form(None), # JSON or null on first turn
    problem_text: str    = Form(None),   # first turn only
    student_comment: str = Form(""),     # first turn only — may be ""
    student_response: str = Form(None),  # subsequent turns
    use_thinking: bool   = Form(True),   # home-screen toggle — enable/disable thinking
):
    try:
        state      = json.loads(session_state) if session_state else None
        first_turn = state is None

        # On subsequent turns the client doesn't re-send the toggle, so read
        # it back from state. On the first turn the client sends it explicitly.
        if not first_turn:
            use_thinking = state.get("use_thinking", True)

        # ── Resolve inputs ────────────────────────────────────────────
        if first_turn:
            problem        = (problem_text or "").strip()
            student_input  = (student_comment or "").strip()
            prior_status   = ""
            prior_reply    = ""
            solution_steps = ""   # filled by Phase 0 below
            
        else:
            problem        = state.get("formal_problem", "")
            prior_status   = state.get("prior_status", "")
            prior_reply    = state.get("last_teacher_reply", "")
            solution_steps = state.get("solution_steps", "")
            student_input  = (student_response or "").strip()
            

        # ── Phase 0: generate solution steps (first turn only) ────────
        if first_turn:
            steps_prompt = build_solution_steps_prompt(problem)

            steps_raw = await call_llama_completion(steps_prompt, n_predict=256, temperature=0, thinking=True, label="PHASE 0", color=C.YELLOW)

            try:
                steps_parsed   = extract_json(steps_raw)
                steps_list     = steps_parsed.get("solution_steps", [])
                solution_steps = "\n".join(steps_list)
            except Exception as e:
                dbg("PHASE 0 PARSE ERROR", f"{e}\n\nRaw output was:\n{steps_raw}", C.RED)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Phase 0 parse failed: {e}", "raw": steps_raw},
                )

            
        # ── Phase 1: diagnose student status + select role (merged) ───
        # Bypass: if the student says nothing in the comment, can skip
        # the LLM call entirely to reduce latency. Default role is inject_info.
        if first_turn and not student_input:
            student_status = "The student has not yet commented and does not know where to begin."
            role    = "inject_info"
            dbg("PHASE 1 SKIPPED  (empty comment)", student_status, C.MAGENTA)
        else:
            phase1_prompt = build_status_and_role_prompt(
                problem=problem,
                solution_steps=solution_steps,
                student_input=student_input,
                prior_status=prior_status,
                prior_teacher_reply=prior_reply,
                first_turn=first_turn,
            )
            phase1_raw = await call_llama_completion(phase1_prompt, n_predict=192, temperature=0, thinking=use_thinking, label="PHASE 1", color=C.CYAN)

            try:
                phase1_parsed  = extract_json(phase1_raw)
                student_status = phase1_parsed.get("student_status", prior_status)
                role           = phase1_parsed.get("role", "hint")
            except Exception as e:
                dbg("PHASE 1 PARSE ERROR", f"{e}\n\nRaw output was:\n{phase1_raw}", C.RED)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Phase 1 parse failed: {e}", "raw": phase1_raw},
                )


        # ── Adaptive support escalation (partial_worked_step only) ────
        prior_count = (state or {}).get("partial_worked_count", 0)
        if role == "partial_worked_step":
            count  = prior_count + 1
            support = {1: "low", 2: "medium"}.get(count, "high")
        else:
            count  = 0
            support = None

        dbg("PHASE 1 OUTPUT", f"student_status: {student_status}\nrole: {role}\ncount: {count} | support: {support}", C.CYAN)

        # Shared next state (written before Phase 2 so session_close can reuse it)
        next_state = {
            "formal_problem":        problem,
            "prior_status":          student_status,
            "last_teacher_reply":      "",        # filled after Phase 2
            "solution_steps":        solution_steps,
            "partial_worked_count": count,
            "use_thinking":          use_thinking,
        }

        # ── Session close — bypass LLM call to reduce latency ───────
        if role == "session_close":
            close_msg = "Correct. That completes the problem. Well done! 😊"
            dbg("SESSION CLOSE", "", C.RED)
            next_state["last_teacher_reply"] = close_msg
            return {
                "response": close_msg,
                "session_state": next_state,
                "controller": {
                    "role":           role,
                    "support":        support,
                    "student_status": student_status,
                },
                "session_complete": True,
            }

        # ── Phase 2: generate teacher reply ─────────────────────────────
        teacher_prompt = build_teacher_prompt(
            solution_steps=solution_steps,
            problem=problem,
            student_attempt=student_input,
            current_status=student_status,
            role=role,
            support=support,
        )

        teacher_output = (await call_llama_completion(
            teacher_prompt,
            n_predict=512,
            temperature=0,
            label="PHASE 2",
            color=C.MAGENTA,
        )).strip()

        next_state["last_teacher_reply"] = teacher_output

        return {
            "response": teacher_output,
            "session_state": next_state,
            "controller": {
                "role":           role,
                "support":        support,
                "student_status": student_status,
            },
            "session_complete": False,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Title generation (kept on /v1/chat/completions) ──────────────────

@app.post("/generate-title")
async def generate_title(messages: str = Form(...)):
    try:
        history = json.loads(messages)

        # Use the first user + assistant pair for context
        context = [m for m in history if isinstance(m.get("content"), str)][:2]
        prompt = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:300]}" for m in context
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate ultra-short chat titles. "
                        "Reply with ONLY 2-5 words that summarise the conversation topic. "
                        "No punctuation, no quotes, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Summarise this conversation in 2-5 words:\n\n{prompt}",
                },
            ],
            "stream": False,
            "max_tokens": 16,
            "temperature": 0,
        }

        resp = await _http_client.post(LLAMA_CHAT_URL, json=payload)
        resp.raise_for_status()
        title = resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
        return {"title": title}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})