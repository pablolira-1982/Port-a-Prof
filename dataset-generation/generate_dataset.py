"""
Port-a-Prof student-teacher synthetic dataset generator.

Reads the question files (e.g. questions/algebra.json) output by
generate_questions.py and generates synthetic multi-turn 
student-teacher learning trajectories
for supervised fine-tuning.

Each generated trajectory simulates a student-teacher interaction
where a student progresses through a problem with varying levels of
understanding, mistakes, and teacher support.

── Input format ── 
Each category JSON file must have the structure:

{
  "category": "algebra",
  "questions": [
    {
      "question": "...",
      "subtopic": "...",
      "solution": ["step 1", "step 2", ...]
    }
  ]
}

The solution steps act as internal ground-truth reasoning references
used to guide dialogue generation and validate student behaviour.

── Dataset generation pipeline ── 
For every problem, the generator instantiates five fixed trajectory
templates defined in TRAJECTORY_TEMPLATES.

Each trajectory contains 2–3 conversational turns.

For every turn, the system performs two sequential LLM operations:

1. Dialogue generation
   generate_pair()

   Generates:
   - one student message
   - one teacher response

   Generation is conditioned on:
   - the problem
   - internal solution steps
   - conversation history
   - the trajectory template
   - teacher role + support level constraints

2. Internal state labelling
   label_state()

   After each student turn, the system generates a structured
   supervision label describing the student's latest learning state.

   Labels include:
   - current_status
       Third-person description of the student's current understanding,
       misconception, procedural mistake, or progress.

   - teacher_role
       Strategy used by the teacher
       (redirect, partial_worked_step, inject_info, confirm_and_advance, session_close).

   - support (for partial worked step only)
       low / medium / high.

These internal state labels are used as the supervision signal during Port-a-Prof's
fine-tuning so the model learns the behaviour/style of each teaching role.

Please note that thinking mode is enabled for both dialogue generation and state
labelling to encourage more coherent multi-turn interactions, more
accurate identification of student misconceptions, and stronger
consistency with the ground-truth solution steps.

── Output format ── 
One output file is generated per problem:

output/<category>_q<n>_trajectories.json

Structure:

{
  "problem": "...",
  "topic": "...",
  "subtopic": "...",
  "solution_steps": [...],
  "trajectories": [
    {
      "trajectory_id": "traj_1",
      "entries": [
        {
          "dialogue_history": [...],
          "internal_state": {...},
          "target_teacher_response": "..."
        }
      ]
    }
  ]
}

Each entry represents one supervised training example.

── Robustness and validation ── 
All LLM calls pass through call_with_retry() with automatic retries.

extract_json() automatically handles common malformed outputs:
- markdown-wrapped JSON
- invalid LaTeX escape sequences
- partially corrupted JSON formatting

Any failures are logged to:
failures_<timestamp>.json

── CLI ── 
python generate_dataset.py                         # generate trajectories for all questions
python generate_dataset.py --limit 5               # process only the first 5 questions
python generate_dataset.py --skip-existing         # skip questions with existing output files
python generate_dataset.py --workers 3             # build trajectories in parallel using 3 workers
python generate_dataset.py --verbose               # print prompts, raw model outputs, and parsed JSON
python generate_dataset.py --model gemma4:e4b      # override the default model (gemma4:e4b) with another ollama model
"""

import argparse
import json
import re
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Global configuration for Ollama inference, generation settings, paths, and runtime behaviour
OLLAMA_BASE     = "http://localhost:11434"
MODEL           = "gemma4:e4b"
INPUT_DIR       = Path(r"questions")
OUTPUT_DIR      = Path(r"dataset")
TEMPERATURE     = 0.3
MAX_RETRIES     = 3
REQUEST_TIMEOUT = 360
NUM_PREDICT     = 4096
VERBOSE         = False
PRINT_LOCK   = threading.Lock()
FAILURE_LOG  = [] 
FAILURE_LOCK = threading.Lock()

# Records any generation, parsing, and validation failures for later inspection
def log_failure(q_label: str, subtopic: str, failure_type: str, detail: str):
    with FAILURE_LOCK:
        FAILURE_LOG.append({
            "q_label":      q_label,
            "subtopic":     subtopic,
            "failure_type": failure_type,
            "detail":       detail,
        })

# Behaviour definitions for fixed teacher roles used during dialogue generation
ROLE_DESCRIPTIONS = {
    "session_close": (
        "Confirm the student's final answer is correct in a short, friendly sentence. "
        "Then ask if they have any other problems they'd like to work on. "
        "Do NOT introduce new content, re-explain steps, or ask further questions about the problem."
    ),
    "inject_info": (
        "Explain the relevant concept, rule, or theory required to solve the problem clearly and simply. "
        "Use LaTeX formatting for equations. Use short paragraphs, line breaks, and **bold text** where helpful. "
        "Do NOT solve the problem or carry out any solution steps — only provide the underlying concepts and theory required to solve it. "
        "At the end of the explanation, link back to the current problem, then provide one guiding question that helps the "
        "student apply the theory to progress through the problem."
    ),
    "redirect": (
        "The student has made a specific arithmetic or procedural error. "
        "Draw their attention to the exact step or value that is wrong — "
        "name the operation or expression directly, but do not state the correct value or redo the calculation. "
        "Ask the student to check or redo that specific step themselves. "
        "Maximum two sentences."
    ),
    "confirm_and_advance": (
        "Confirm what the student did right in a friendly but not effusive tone, focusing on the method or reasoning. "
        "Do NOT reteach, explain, or introduce new concepts. "
        "Do NOT affirm or praise the student for recalling or repeating information provided by the teacher in ## CONVERSATION HISTORY. "
        "Do NOT provide any worked steps, calculations, or partial results. "
        "Ask one clear next-step question that is direct and actionable but does not include the answer or perform the step. "
        "Maximum two sentences."
    ),
}

# Specifications to control support intensity for partial worked step
SUPPORT_DESCRIPTIONS = {
    "partial_worked_step": {
        "low": (
            "Do NOT praise, affirm or tell the student they are correct under any circumstance. Focus only on what is needed to progress the student through the problem. "
            "Write the equation, expression, or relationship for the specific milestone the student is currently working on, using only symbolic or variable form — "
            "do NOT substitute any specific values from the problem. "
            "The milestone should represent a meaningful piece of logic — not a single trivial step that would be obvious without guidance."
            "Use $$ ... $$ for standalone equations and $ ... $ for inline math references within a sentence. "
            "Use **bold text** to highlight key terms or values and separate distinct ideas with \\n for clarity. "
            "Include clear, concise explanations where appropriate. "
            "End with one clear next action for the student."
        ),
        "medium": (
            "Do NOT praise, affirm or tell the student they are correct under any circumstance. Focus only on what is needed to progress the student through the problem. "
            "Do NOT repeat or restate anything from the teacher dialouge in ## CONVERSATION HISTORY unless necessary. "
            "Link directly to the symbolic setup already provided and extend it — "
            "identify and define the relevant variables from the problem, include any necessary reasoning "
            "for how they are obtained, but do NOT substitute, simplify, or compute. "
            "Use $$ ... $$ for standalone equations and $ ... $ for inline math references within a sentence. "
            "Use **bold text** to highlight key terms or values and separate distinct ideas with \\n for clarity. "
            "End by asking the student to substitute and continue from the setup."
        ),
        "high": (
            "Do not praise, affirm or tell the student they are correct under any circumstance. Focus only on what is needed to progress the student through the problem. "
            "Do not repeat or restate anything from the teacher dialouge in ## CONVERSATION HISTORY unless necessary. "
            "Using the variables and setup already defined, work through the evaluation. "
            "Include a brief explanation for any non-obvious operations or transformations. "
            "Use $$ ... $$ for standalone equations and $ ... $ for inline math references within a sentence. "
            "Use **bold text** to highlight key terms or values and separate distinct lines of working with \\n for clarity. "
            "Stop before one final small action (e.g. the last arithmetic operation, final rearrangement, or final value). "
            "End with one clear instruction asking the student to complete that remaining action."
        ),
    },
}

# Coding-specific support behaviour rules using inline code formatting instead of LaTeX
SUPPORT_DESCRIPTIONS_CODING = {
    "partial_worked_step": {
        "low": (
            "Do NOT praise, affirm or tell the student they are correct under any circumstance. Focus only on what is needed to progress the student through the problem. "
            "Write the relevant construct, expression, or condition for the specific milestone the student is currently working on, using only variable names — "
            "do not substitute any specific values from the problem. "
            "The milestone should represent a meaningful piece of logic — not a single trivial step that would be obvious without guidance."
            "Use inline backticks (`...`) for all code, variable names, and operators. "
            "Do NOT use LaTeX or code blocks under any circumstances. "
            "Use **bold text** to highlight key terms and separate distinct ideas with \\n for clarity. "
            "Include clear, concise explanations where appropriate. "
            "End with one clear next action for the student."
        ),
        "medium": (
            "Do NOT praise, affirm or tell the student they are correct under any circumstance. Focus only on what is needed to progress the student through the problem. "
            "Do NOT repeat or restate anything from the teacher dialouge in ## CONVERSATION HISTORY unless necessary. "
            "Link directly to the construct already provided and extend it — "
            "identify and define the relevant variables from the problem, include any necessary reasoning "
            "for how they are obtained, but do not evaluate, resolve, or compute the result. "
            "Use inline backticks (`...`) for all code, variable names, and operators. "
            "Do NOT use LaTeX or code blocks under any circumstances. "
            "Use **bold text** to highlight key terms and separate distinct ideas with \\n for clarity. "
            "End by asking the student to evaluate or continue from the construct."
        ),
        "high": (
            "Do NOT praise, affirm or tell the student they are correct under any circumstance. Focus only on what is needed to progress the student through the problem. "
            "Do NOT repeat or restate anything from the teacher dialouge in ## CONVERSATION HISTORY unless necessary. "
            "Using the variables and construct already defined, work through the evaluation. "
            "Include a brief explanation for any non-obvious operations or transformations. "
            "Use inline backticks (`...`) for all code, variable names, and operators. "
            "Do NOT use LaTeX or code blocks under any circumstances. "
            "Use **bold text** to highlight key terms and separate distinct lines of working with \\n for clarity. "
            "Stop before one final small action (e.g. the last condition check, assignment, or return value). "
            "End with one clear instruction asking the student to complete that remaining action."
        ),
    },
}

# Fixed student-learning trajectory templates used to synthesise multi-turn conversations
TRAJECTORY_TEMPLATES = [
    {
        "trajectory_id": "traj_1",
        "student_path": [
            "The student states that they are unsure how to begin the problem.",
            "The student acknowledges the teacher's explanation but is unsure how to apply it to this specific problem. "
            "They ask the teacher to help them with the step. "
            "The final answer does not appear."
        ],
        "teacher_turns": [
            {"role": "inject_info",         "support": None},
            {"role": "partial_worked_step", "support": "low"},
        ],
    },
    {
        "trajectory_id": "traj_2",
        "student_path": [
            "The student attempts an early meaningful milestone but makes a procedural error — "
            "a wrong sign, incorrect arithmetic, or a silly calculation mistake. "
            "The final answer does not appear.",
            "The student corrects their procedural error from the \"student\" turn of ## CONVERSATION HISTORY, "
            "arriving at the correct value for that step. "
            "They then express uncertainty about what to do next and ask for help setting up the next step. "
            "The final answer does not appear."
        ],
        "teacher_turns": [
            {"role": "redirect", "support": None},
            {"role": "partial_worked_step", "support": "low"},
        ],
    },
    {
        "trajectory_id": "traj_3",
        "student_path": [
            "The student is near the end of the solution and correctly completes the second-to-last meaningful milestone. "
            "This may involve finding the last needed intermediate value, substituting known values into the final expression, or setting up the final calculation. "
            "They stop before computing or stating the final answer. "
            "The final answer does not appear.",
            "The student completes the final calculation or reasoning step and states the correct final answer. "
            "The final answer must exactly match the final result implied by the internal solution steps."
        ],
        "teacher_turns": [
            {"role": "confirm_and_advance", "support": None},
            {"role": "session_close",       "support": None},
        ],
    },
    {
        "trajectory_id": "traj_4",
        "student_path": [
            "The student correctly completes the first meaningful milestone and then asks for help setting up the next step.",
            "The student utilises the setup given in the \"teacher\" turn of ## CONVERSATION HISTORY, "
            "but makes an arithmetic or logic mistake during their evaluation of the step — "
            "for example, combining the wrong terms, applying the wrong operation, "
            "or miscalculating — resulting in an incorrect value for that step. "
            "The final answer does not appear."
        ],
        "teacher_turns": [
            {"role": "partial_worked_step", "support": "low"},
            {"role": "redirect",            "support": None},
        ],
    },
    {   "trajectory_id": "traj_5",
        "student_path": [
            "The student does not know how to begin the problem. "
            "They ask for some help with the set-up.",

            "The student has made ZERO progress from the teacher's explanation in ## CONVERSATION HISTORY — "
            "they show no understanding of what was provided. "
            "They do NOT know how to obtain the required variables and ask where the values come from. "
            "The student has NOT performed any calculations. "
            "The final answer does NOT appear.",

            "The student has made ZERO progress from the teacher's explanation in ## CONVERSATION HISTORY — "
            "they show no understanding of what was provided. "
            "They do NOT know how to carry out the evaluation and ask to be shown how to work through it. "
            "The student has NOT performed any calculations. "
            "The final answer does NOT appear.",
        ],
        "teacher_turns": [
            {"role": "partial_worked_step", "support": "low"},
            {"role": "partial_worked_step", "support": "medium"},
            {"role": "partial_worked_step", "support": "high"},
        ]
    }
]

# Prints structured verbose logs for prompts, raw outputs, and parsed responses
def _vprint(header: str, body: str, colour: str = "", tag: str = "") -> None:
    RESET  = "\033[0m"
    PROMPT = "\033[0;33m"
    RAW    = "\033[0;32m"
    PARSED = "\033[0;35m"
    colours = {"prompt": PROMPT, "raw": RAW, "parsed": PARSED}
    c      = colours.get(colour, "\033[1;36m")
    prefix = f"[{tag}] " if tag else ""
    with PRINT_LOCK:
        print(f"\n{c}{'─' * 70}")
        print(f"  {prefix}{header}")
        print(f"{'─' * 70}{RESET}")
        print(body)
        print()

# Sends a generation request to Ollama and returns the cleaned raw model output
def call_ollama(prompt: str, tag: str = "", thinking: bool = False) -> str:
    if VERBOSE:
        _vprint("PROMPT →", prompt, "prompt", tag)

    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
    }
    if thinking:
        payload["think"] = True

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    raw  = data["response"]

    # Log thinking if present
    if thinking and VERBOSE:
        scratchpad = data.get("thinking", "").strip()
        if scratchpad:
            _vprint("THINKING SCRATCHPAD ←", scratchpad, "", tag)

    # Strip special tokens
    SPECIAL_TOKENS = [
        "<end_of_turn>",
        "</end_of_turn>", 
        "<start_of_turn>",
        "<bos>",
        "<eos>",
        "<|endoftext|>",
    ]
    for tok in SPECIAL_TOKENS:
        raw = raw.replace(tok, "")

    raw = raw.strip()
    if VERBOSE:
        _vprint("RAW MODEL OUTPUT ←", raw, "raw", tag)
    return raw

# Repairs invalid single-backslash LaTeX escapes before JSON parsing
def _fix_latex_escapes(s: str) -> str:
    _KEEP = frozenset({'"', '\\', '/', 'n', 'u'})
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == '\\':
                # Already double-backslash — keep and skip both chars
                result.append('\\\\')
                i += 2
            elif next_char in _KEEP:
                # Intentional JSON escape — keep as-is
                result.append('\\')
                result.append(next_char)
                i += 2
            else:
                # Single-backslash LaTeX — double the backslash only,
                # let the next char be processed normally on the next iteration
                result.append('\\\\')
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)

# Recursively extracts all string values from a parsed JSON object
def _scan_string_values(obj) -> list:
    strings = []
    if isinstance(obj, dict):
        for v in obj.values():
            strings.extend(_scan_string_values(v))
    elif isinstance(obj, list):
        for v in obj:
            strings.extend(_scan_string_values(v))
    elif isinstance(obj, str):
        strings.append(obj)
    return strings

# Detects silent LaTeX corruption caused by invalid JSON escape parsing
def _check_post_parse_corruption(obj) -> list:
    _CORRUPT = {'\x08': r'\b', '\x0c': r'\f', '\r': r'\r', '\t': r'\t'}
    issues = []
    for s in _scan_string_values(obj):
        for char, name in _CORRUPT.items():
            if char in s:
                issues.append(
                    f"string contains {name} control char "
                    f"(likely unescaped LaTeX e.g. \\{name[1:]}ext / \\frac / \\rho / \\beta)"
                )
    return issues

# Extracts and validates one JSON object from raw model output
def extract_json(text: str):
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    candidate = fenced.group(1).strip() if fenced else text.strip()

    def _try_parse(s):
        try:
            obj = json.loads(s)
            if _check_post_parse_corruption(obj):
                return None
            return obj
        except json.JSONDecodeError:
            return None

    # Attempt 1: parse as-is
    result = _try_parse(candidate)
    if result is not None:
        return result

    # Attempt 2: fix single-backslash LaTeX (e.g. \ge → \\ge) and retry
    result = _try_parse(_fix_latex_escapes(candidate))
    if result is not None:
        return result

    # Fallback: find first JSON-looking object and repeat both attempts
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = candidate[start:end + 1]
        result = _try_parse(sub)
        if result is not None:
            return result
        result = _try_parse(_fix_latex_escapes(sub))
        if result is not None:
            return result
    return None

# Executes an Ollama generation call with retry and validation handling
def call_with_retry(prompt: str, validator=None, label: str = "", tag: str = "", q_label: str = "", subtopic: str = "", thinking: bool = False):
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw    = call_ollama(prompt, tag=tag, thinking=thinking)
            result = extract_json(raw)
            if result is None:
                raise ValueError("No valid JSON found in response")
            if validator and not validator(result):
                raise ValueError(f"Validation failed ({type(result).__name__})")
            if VERBOSE:
                _vprint(f"PARSED [{label}] ✓", json.dumps(result, indent=2), "parsed", tag)
            return result
        except Exception as e:
            last_error = str(e)
            if VERBOSE:
                with PRINT_LOCK:
                    print(f"        [{tag or label}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    # All retries exhausted — log the failure
    if q_label:
        log_failure(q_label, subtopic, label, f"{tag}: all {MAX_RETRIES} attempts failed: {last_error}")
    return None

# Builds the behavioural specification for one student-teacher interaction pair
def _pair_spec(template: dict, pair_index: int, is_coding: bool = False) -> str:
    student_spec = template["student_path"][pair_index]
    turn         = template["teacher_turns"][pair_index]
    role         = turn["role"]
    support  = turn["support"]
    descriptions = SUPPORT_DESCRIPTIONS_CODING if is_coding else SUPPORT_DESCRIPTIONS
    lines        = [f'  "student": {student_spec}']
    if support and role in descriptions:
        lines.append(f'  "teacher": {descriptions[role][support]}')
    elif role in ROLE_DESCRIPTIONS:
        lines.append(f'  "teacher": {ROLE_DESCRIPTIONS[role]}')
    else:
        lines.append(f'  "teacher": [{role}]')
    return "\n".join(lines)

# Generates one student message and one teacher response for a trajectory step
def generate_pair(
    problem: str,
    steps: list,
    template: dict,
    pair_index: int,
    previous_dialogue: list,
    is_coding: bool = False,
    is_chemistry: bool = False,
    q_label: str = "",
    subtopic: str = "",
) -> Optional[dict]:
    """Returns {"student": "...", "teacher": "..."}"""
    steps_str    = "\n".join(steps)
    pair_spec    = _pair_spec(template, pair_index, is_coding=is_coding)
    current_role = template["teacher_turns"][pair_index]["role"]

    context = ""
    if previous_dialogue:
        context = (
            "\n## CONVERSATION HISTORY\n"
            + "\n".join(
                f'  "{m["role"]}": {m["content"]}'
                for m in previous_dialogue
            )
            + "\n"
        )

    answer_rule = (
        "The student MUST state the correct final answer in their turn."
        if current_role == "session_close" else
        "The student must NOT state the correct final answer in their turn. "

    )

    if is_chemistry:
        teacher_rules = """
## TEACHER RULES
- Do not open with praise or encouragement. Never start with "Great", "Good job", "Well done", "Nice work", "Excellent", or any generic affirmation.
- Respond to the student's exact wording; do not give a generic or templated reply.
- Be clear, engaged, and conversational while remaining precise.
- Never give the final answer directly.
- Keep responses concise and targeted. Do not restate known information.
- Ground all explanations in the current problem using actual values or expressions.
- Write all LaTeX with single backslashes: $$\\ge$$, $$\\le$$, $$\\frac{{a}}{{b}}$$
- Use LaTeX for equations: $$ ... $$ for standalone equations and $ ... $ for inline math references within a sentence.
- Use \\n for line breaks. Do not include literal newlines inside strings.
- Escape double quotes inside strings as \\"
- All chemical formulas must use Unicode subscripts only: H₂O, CuSO₄, Cu(OH)₂
- NEVER praise or tell the student they are correct for identifying trivial quanitites that are blatantly provided in a problem"""
    elif is_coding:
        teacher_rules = """\
## TEACHER RULES
- Do not open with praise or encouragement. Never start with "Great", "Good job", "Well done", "Nice work", "Excellent", or any generic affirmation.
- Respond to the student's exact wording; do not give a generic or templated reply.
- Be clear, engaged, and conversational while remaining precise.
- Never give the final answer directly.
- Keep responses concise and targeted. Do not restate known information.
- Ground all explanations in the current problem using actual values or expressions.
- Use inline backticks for ALL code, variable names, keywords, and operators: `age`, `if`, `else`, `>=`.
- Do NOT use LaTeX.
- Use ONLY inline backticks (`...`) for code. Do NOT use code blocks (``` ... ```) under any circumstances.
- Every piece of code must fit on a single line inside backticks. If it doesn't fit, break it into separate inline references.
- Use \\n for line breaks. Do not include literal newlines inside strings.
- Escape double quotes inside strings as \\"
- NEVER praise or tell the student they are correct for identifying trivial quanitites that are blatantly provided in a problem"""
    else:
        teacher_rules = """\
## TEACHER RULES
- Do not open with praise or encouragement. Never start with "Great", "Good job", "Well done", "Nice work", "Excellent", or any generic affirmation.
- Respond to the student's exact wording; do not give a generic or templated reply.
- Be clear, engaged, and conversational while remaining precise.
- Never give the final answer directly.
- Keep responses concise and targeted. Do not restate known information.
- Ground all explanations in the current problem using actual values or expressions.
- Write all LaTeX with single backslashes: $$\\ge$$, $$\\le$$, $$\\frac{{a}}{{b}}$$
- Use LaTeX for equations: $$ ... $$ for standalone equations and $ ... $ for inline math references within a sentence.
- Use \\n for line breaks. Do not include literal newlines inside strings.
- Escape double quotes inside strings as \\" 
- NEVER praise or tell the student they are correct for identifying trivial quanitites that are blatantly provided in a problem"""

    prompt = f"""You are generating a conversational student-teacher dataset.

## CONTEXT
Problem: {problem}

Solution steps (internal reference only — do not quote verbatim):
{steps_str}
{context}
## OUTPUT
Write exactly one student message and one teacher response as a JSON object with the following format:
{{
  "student": "...",
  "teacher": "..."
}}

The "student" and "teacher" values MUST follow this narrative:
{pair_spec}

## STUDENT RULES
- Write in natural, conversational language.
- Vary how the response begins, minimise filler words like "Okay", "So", "Alright", "Sure"
- PLAIN TEXT ONLY — NO exceptions, even if the teacher used backticks or LaTeX:
  - NO backticks. Write: the age variable, the if statement — NOT `age`, `if`
  - NO LaTeX or math symbols. Write: 17 >= 18 — NOT $$17 \\ge 18$$
  - NO code blocks, no line breaks.
- Do not mirror the teacher's formatting. Convert any backticks, LaTeX, or code blocks into plain spoken words.
- Show hesitation, partial understanding, or uncertainty where appropriate.
- Do not use structured solution format (no numbered steps or formal derivations).
- Vary how the response begins. Avoid repetitive phrasing.
- {answer_rule}
- Do not refer to these instructions.

{teacher_rules}

## FORMAT RULES
- Output exactly one valid JSON object. No text before or after.
- All string values must be valid JSON strings.
- The JSON must be parseable with json.loads without modification.
"""

    return call_with_retry(
        prompt,
        validator=lambda x: (
            isinstance(x, dict)
            and "student" in x
            and "teacher" in x
            and isinstance(x["student"], str)
            and isinstance(x["teacher"], str)
        ),
        label="pair",
        tag=f"{template['trajectory_id']}-p{pair_index + 1}",
        q_label=q_label,
        subtopic=subtopic,
        thinking=True,
    )

# Generates supervision labels describing the student's current learning state
def label_state(
    problem: str,
    steps: list,
    snapshot: list,
    expected_role: str,
    expected_support: Optional[str],
    q_label: str = "",
    subtopic: str = "",
) -> Optional[dict]:
    steps_str = "\n".join(steps)
    snap_str  = json.dumps(snapshot, indent=2)

    prompt = f"""You are labelling a conversational student-teacher dataset entry.

## Context
Problem: {problem}

Solution steps:
{steps_str}

Dialogue (ends on the student's latest turn):
{snap_str}

## Task
Generate the JSON object below.

## Rules
- Verify every numerical value or variable the student states against the solution steps before writing current_status.
- If any value does not match the solution steps, current_status must identify it as an error.
- Write current_status in third person. Describe only what the student most recently did, understood, or struggled with.

## Formatting — STRICT PLAIN TEXT ONLY
Do NOT mirror the teacher's formatting. Convert any backticks, LaTeX, or code blocks into natural language.
The current_status field must contain plain prose. No exceptions:
- No LaTeX. No $$ ... $$ No \\ge, \\frac, \\text, or any backslash commands.
- Write math as plain text - Example: 17 >= 18, not $$17 \\ge 18$$
- No backticks. No inline code. No code blocks.
- Write code/variable names as plain words: the age variable, the if statement — not `age`, `if`.

## Output format
Output only the JSON object. No text before or after.
{{
  "current_status": "<1–2 sentences. Third person. Identify what the student most recently did, understood, or struggled with — including any specific misconceptions, conceptual gaps or calculation errors.>"
}}"""

    result = call_with_retry(
        prompt,
        validator=lambda x: isinstance(x, dict) and "current_status" in x,
        label="state",
        tag=expected_role,
        q_label=q_label,
        subtopic=subtopic,
        thinking=True,
    )
    if result:
        result["teacher_role"] = expected_role
        result["support"]    = expected_support if expected_support else "none"
    return result

# Build one trajectory (called in parallel)
def build_trajectory(problem: str, steps: list, template: dict, is_coding: bool = False, is_chemistry: bool = False, q_label: str = "", subtopic: str = "") -> Optional[dict]:
    tid            = template["trajectory_id"]
    teacher_turns  = template["teacher_turns"]
    entries            = []
    previous_dialogue  = []

    for pair_index, turn in enumerate(teacher_turns):
        pair = generate_pair(problem, steps, template, pair_index, previous_dialogue, is_coding=is_coding, is_chemistry=is_chemistry, q_label=q_label, subtopic=subtopic)
        if not pair:
            break

        snapshot = previous_dialogue + [{"role": "student", "content": pair["student"]}]

        state = label_state(
            problem, steps, snapshot, turn["role"], turn["support"], q_label=q_label, subtopic=subtopic
        )
        if not state:
            break

        entries.append({
            "dialogue_history":        snapshot,
            "internal_state":          state,
            "target_teacher_response": pair["teacher"],
        })

        previous_dialogue = snapshot + [{"role": "teacher", "content": pair["teacher"]}]

    if not entries:
        return None
    return {
        "trajectory_id": tid,
        "entries":       entries,
    }

# Generates and saves all trajectories for a single problem
def process_question(
    problem: str,
    steps: list,
    topic: str,
    subtopic: str,
    q_label: str,
    output_dir: Path,
    skip_existing: bool,
    workers: int,
    is_coding: bool = False,
    is_chemistry: bool = False
) -> bool:
    out_path = output_dir / f"{q_label}_trajectories.json"
    if skip_existing and out_path.exists():
        return True

    trajectories = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_trajectory, problem, steps, tmpl, is_coding, is_chemistry, q_label, subtopic): tmpl
            for tmpl in TRAJECTORY_TEMPLATES
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                trajectories.append(result)

    order = {t["trajectory_id"]: i for i, t in enumerate(TRAJECTORY_TEMPLATES)}
    trajectories.sort(key=lambda t: order.get(t["trajectory_id"], 99))

    if not trajectories:
        log_failure(q_label, subtopic, "question", "No trajectories built")
        return False

    if len(trajectories) < len(TRAJECTORY_TEMPLATES):
        log_failure(q_label, subtopic, "partial", f"Only {len(trajectories)}/{len(TRAJECTORY_TEMPLATES)} trajectories built")

    output = {
        "problem":        problem,
        "topic":          topic,
        "subtopic":       subtopic,
        "solution_steps": steps,
        "trajectories":   trajectories,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return True

# Processes all questions within one category dataset file
def process_category_file(source_path: Path, output_dir: Path, skip_existing: bool, workers: int, limit: Optional[int] = None) -> tuple[int, int]:
    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    category  = data.get("category", source_path.stem)
    questions = data.get("questions", [])
    is_coding = source_path.stem.lower() == "coding"
    is_chemistry = source_path.stem.lower() == "chemistry"

    if limit is not None:
        questions = questions[:limit]

    success = failed = 0
    with tqdm(total=len(questions), desc=f"{source_path.name}", unit="q", leave=True, ncols=80) as pbar:
        for i, q in enumerate(questions, 1):
            problem  = q["question"]
            subtopic = q.get("subtopic", f"q{i}")
            raw_steps = q.get("solution", [])
            steps = [
                re.sub(r"^\d+[\.\)]\s*", "", s.strip())
                for s in raw_steps
            ]
            q_label = f"{category}_q{i}"
            pbar.set_postfix_str(subtopic[:35])

            try:
                ok = process_question(
                    problem=problem,
                    steps=steps,
                    topic=category,
                    subtopic=subtopic,
                    q_label=q_label,
                    output_dir=output_dir,
                    skip_existing=skip_existing,
                    workers=workers,
                    is_coding=is_coding,
                    is_chemistry=is_chemistry
                )
                if ok:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                log_failure(q_label, subtopic, "exception", str(e))
                failed += 1

            pbar.update(1)

    return success, failed

# Command line entry point for full trajectory dataset generation
def main():
    global MODEL, VERBOSE
    parser = argparse.ArgumentParser(description="Generate teaching trajectory datasets")
    parser.add_argument("--limit",         type=int,  default=None,
                        help="Max number of questions to process across all category files")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--input-dir",     type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir",    type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model",         type=str,  default=MODEL)
    parser.add_argument("--workers",       type=int,  default=3,
                        help="Parallel workers for trajectory generation (default: 3)")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()
    MODEL   = args.model
    VERBOSE = args.verbose
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        r         = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        base      = MODEL.split(":")[0]
        if not any(base in m for m in available):
            print(f"WARNING: '{MODEL}' not found. Available: {available}")
            print(f"  Pull with: ollama pull {MODEL}")
    except requests.ConnectionError:
        print("ERROR: Ollama is not running. Start with: ollama serve")
        return

    category_files = sorted(args.input_dir.glob("*.json"))
    total_questions = sum(
        len(json.loads(f.read_text(encoding="utf-8")).get("questions", []))
        for f in category_files
    )

    print(f"Model:      {MODEL}")
    print(f"Input:      {args.input_dir}/  ({len(category_files)} category files, {total_questions} questions)")
    print(f"Output:     {args.output_dir}/")
    print(f"Workers:    {args.workers} parallel trajectories per question")
    print(f"Templates:  {len(TRAJECTORY_TEMPLATES)} fixed")
    if args.limit:
        print(f"Limit:      {args.limit} questions")
    if VERBOSE:
        print("Verbose:    ON")

    total_success = total_failed = 0
    remaining = args.limit  # None means unlimited

    for fp in category_files:
        if remaining is not None and remaining <= 0:
            break
        s, f = process_category_file(
            source_path=fp,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            workers=args.workers,
            limit=remaining,
        )
        total_success += s
        total_failed  += f
        if remaining is not None:
            remaining -= (s + f)

    print(f"\n{'═' * 60}")
    print(f"Done.  {total_success} succeeded, {total_failed} failed.")
    print(f"Outputs in '{args.output_dir}/'")

    # Write failure log
    log_path = args.output_dir / f"failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if FAILURE_LOG:
        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(FAILURE_LOG, lf, indent=2, ensure_ascii=False)
        print(f"Failure log  → {log_path}  ({len(FAILURE_LOG)} entries)")
    else:
        print("No failures logged.")

if __name__ == '__main__':
    main()