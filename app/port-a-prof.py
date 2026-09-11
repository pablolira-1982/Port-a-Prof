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
LLAMA_COMPLETION_URL = f"{LLAMA_BASE_URL}/completion"           # bruto — usado na sessão de ensino (com tokens de raciocínio)
LLAMA_CHAT_URL       = f"{LLAMA_BASE_URL}/v1/chat/completions"  # compatível com OpenAI — usado para visão, áudio e geração de título
MODEL_NAME = "Port-a-Prof"
APP_TITLE  = "Port-a-Prof"

_http_client = None

# ── LOGS DE DEBUG ────────────────────────────────────────────────────
DEBUG = True   # defina como False para silenciar todos os logs de debug

class C:
    """Códigos de cor ANSI."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    # rótulos
    CYAN    = "\033[96m"   # cabeçalhos de seção
    YELLOW  = "\033[93m"   # prompts
    GREEN   = "\033[92m"   # saída bruta do modelo
    MAGENTA = "\033[95m"   # dados interpretados / estruturados
    BLUE    = "\033[94m"   # entradas do cliente
    RED     = "\033[91m"   # erros / avisos
    DIM     = "\033[2m"    # divisores

_DIVIDER = f"{C.DIM}{'─' * 36}{C.RESET}"

def dbg(label: str, content: str, color: str = C.CYAN) -> None:
    """Imprime um bloco de debug rotulado em stderr."""
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


# ── Prompts do professor ─────────────────────────────────────────────────────
#
# Ordem de chamada em cada turno:
#   Fase 0  (apenas primeiro turno) — build_solution_steps_prompt
#   Fase 1  (todo turno)            — build_status_and_role_prompt  
#   Fase 2  (todo turno)            — build_teacher_prompt


# ── Fase 0: passos internos da solução (apenas ao iniciar a sessão) ───────────

def build_solution_steps_prompt(problem: str) -> str:
    inner = f"""
Você é um solucionador de problemas especialista. Produza uma solução interna e concisa para o problema abaixo.

### PROBLEMA
{problem}

Retorne apenas JSON com este esquema:
{{"solution_steps": ["1. passo um", "2. passo dois", "..."]}}

Regras:
- Numere cada passo "1.", "2.", "3.", etc.
- Cada passo é uma operação lógica ou cálculo, mostrado explicitamente com o desenvolvimento.
- O passo final deve incluir a resposta final de forma natural como parte do cálculo ou conclusão. Não adicione um passo separado reafirmando-a.
- Escreva as expressões na sua forma equivalente mais simples. Oculte termos nulos, omita operações triviais (ex. +0, *1), e reduza antes de apresentar.
- Não explique nem atue como tutor — isto é apenas para referência.
- A saída deve ser apenas JSON válido. Sem markdown."""
    return f"""<bos><|turn>user
{inner}<turn|>
<|turn>model
"""


# ── Fase 1: diagnóstico zero-shot do estado do aluno + seleção do papel do professor ─────

def build_status_and_role_prompt(
    problem: str,
    solution_steps: str,
    student_input: str,
    prior_status: str = "",
    prior_teacher_reply: str = "",
    first_turn: bool = True,
) -> str:
    context_block = "" if first_turn else f"""
## CONTEXTO (apenas para referência)
STATUS_ANTERIOR: {prior_status}
RESPOSTA_ANTERIOR_DO_PROFESSOR: "{prior_teacher_reply}"
"""
    label = "COMENTÁRIO_DO_ALUNO" if first_turn else "RESPOSTA_DO_ALUNO"
    status_rule = "" if first_turn else "- Baseie o student_status apenas em RESPOSTA_DO_ALUNO. Você não deve conectar ao STATUS_ANTERIOR, nem elogiar ou afirmar o aluno por nenhuma informação fornecida na RESPOSTA_ANTERIOR_DO_PROFESSOR."

    inner = f"""Você é um controlador de diagnóstico para um professor de IA. Diagnostique o aluno e selecione uma função de resposta.

## PROBLEMA
{problem}

## PASSOS DA SOLUÇÃO (referência correta)
{solution_steps}
{context_block}
## {label}
"{student_input}"

## SAÍDA 
Retorne apenas JSON com este esquema:
{{"student_status": "", "role": ""}}

IMPORTANTE:
Antes de escrever `student_status`, faça uma auditoria passo a passo da expressão do aluno em relação a `## PASSOS DA SOLUÇÃO`.

Para CADA termo ou valor no diálogo do aluno:
1. Identifique a qual termo ou valor de referência em `## PASSOS DA SOLUÇÃO` ele corresponde.
2. Verifique se o valor, sinal, coeficiente, operador e expoente coincidem exatamente.
3. Trate qualquer valor deslocado, coeficiente faltando, sinal incorreto, expoente incorreto ou operação incorreta como um erro, mesmo se o número aparecer em outro lugar na solução correta.

Não julgue a correção com base apenas na similaridade numérica. Um número correto usado na estrutura errada ainda é incorreto.

student_status: 1-2 frases, na terceira pessoa. Descreva o que o aluno demonstrou, tentou ou expressou em ## {label}. Compare todos os valores termo a termo com o passo relevante em ## PASSOS DA SOLUÇÃO para confirmar a correção ou identificar erros específicos, incluindo adesão correta a fórmulas. Não deduza entendimento completo por afirmações parciais ou vagas. Apenas marque um conceito como compreendido se o aluno o demonstrar explicitamente. Mencione o que permanece não resolvido ou ambíguo. Se a entrada estiver em branco, o aluno não sabe por onde começar. MANTENHA O TEXTO GERADO EM PORTUGUÊS DO BRASIL.
role — escolha exatamente uma:
- session_close: a resposta do aluno coincide com a resposta final conforme ## PASSOS DA SOLUÇÃO
- inject_info: o aluno tem uma lacuna de conhecimento que deve ser preenchida antes de prosseguir
- confirm_and_advance: o último passo do aluno está correto, mas o problema ainda não foi concluído
- partial_worked_step: o aluno pede ajuda para montar ou realizar o próximo passo
- redirect: o aluno cometeu um erro que deve ser corrigido antes de prosseguir. 

Regras:
{status_rule}
- Se houver qualquer cálculo ou raciocínio incorreto, o redirect tem precedência sobre todas as outras funções, incluindo session_close e confirm_and_advance.
- Se a resposta do aluno contiver o valor da solução final (equivalente ao resultado da última entrada em ## PASSOS DA SOLUÇÃO), mesmo embutido em uma expressão maior, a função deve ser session_close.
- Uma expressão do aluno está correta se for matematicamente equivalente ao passo esperado, mesmo que simplificada. Omitir termos nulos, pular operações triviais ou reordenar operações comutativas são válidos. Não marque como erro termos ausentes cujo valor seja zero ou cuja omissão não altere o resultado.
- Compare os sinais de todos os valores usados pelo aluno com os sinais dos valores em ## PASSOS DA SOLUÇÃO. Se houver divergência de sinais, a função deve ser redirect.
- Retorne APENAS o objeto JSON. Não produza nada após a chave de fechamento."""

    return f"""<bos><|turn>user
{inner}<turn|>
<|turn>model
"""


# ── Fase 2: resposta do professor, entrada no mesmo formato do fine-tuning QLoRA  ────────

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
            raise ValueError("Nenhum objeto JSON encontrado")
    return json.loads(cleaned)


# ── Verificações pré-sessão ────────────────────────────────────────────────
#
#  1. build_problem_validation_prompt  — isto é um problema acadêmico válido?
#  2. call_llama_vision                — imagem → texto do problema extraído
#  Endpoints: POST /validate-problem e POST /extract-problem

def build_problem_validation_prompt(problem: str) -> str:
    inner = f"""Você é um assistente de validação para um sistema educacional de IA.
Classifique a entrada abaixo em exatamente uma das três categorias.

ENTRADA:
{problem}

Categorias:
- "valid": um problema determinístico com uma resposta correta clara — matemática, ciência, lógica, programação, regras gramaticais, história factual, etc.
- "non_deterministic": uma questão acadêmica genuína, mas sem uma única resposta correta — redação, análise literária, escrita criativa, humanidades baseadas em opinião, debates éticos, etc.
- "invalid": linguagem sem sentido, caracteres aleatórios, saudações, conversa fora do tópico, conteúdo ofensivo, ou não é uma pergunta.

Retorne apenas JSON: {{"result": "valid"}} ou {{"result": "non_deterministic"}} ou {{"result": "invalid"}}
Sem markdown, sem texto extra."""
    return f"""<bos><|turn>user
{inner}<turn|>
<|turn>model
"""


async def call_llama_vision(image_b64: str, mime_type: str, text_prompt: str, max_tokens: int = 512) -> str:
    """Envia uma imagem + prompt de texto ao Gemma via /v1/chat/completions (rota de visão)."""
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
    print(f"\n{C.GREEN}", end="", file=sys.stderr, flush=True)   # saída bruta do modelo is green

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


# ── Estáticos / index ───────────────────────────────────────────────────

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


# ── Transcrição (áudio Gemma 4 via llama-server) ───────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    tmp_in  = None
    tmp_wav = None
    try:
        # Salva o áudio recebido (webm no Chrome, mp4 no Safari — ffmpeg lida com ambos)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
            while chunk := await audio.read(1024 * 1024):
                f.write(chunk)
            tmp_in = f.name

        # Converte para WAV mono 16 kHz 
        tmp_wav = tmp_in + ".wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_wav,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("Falha na conversão de áudio para WAV pelo ffmpeg")

        # Codifica o WAV em Base64
        with open(tmp_wav, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        # Envia ao Gemma via llama-server
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": "Transcreva este áudio com exatidão. Retorne apenas as palavras faladas, nada mais."},
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


# ── Pré-sessão: validação do problema ──────────────────────────────────

@app.post("/validate-problem")
async def validate_problem(problem_text: str = Form(...)):
    """Verifica se o texto enviado é um problema acadêmico genuíno."""
    try:
        problem = problem_text.strip()
        if not problem:
            return {"result": "invalid"}

        prompt = build_problem_validation_prompt(problem)
        raw = await call_llama_completion(prompt, n_predict=128, temperature=0, thinking=False, label="PROMPT DE VALIDAÇÃO DO PROBLEMA")

        parsed = extract_json(raw)
        result = parsed.get("result", "invalid")
        dbg("RESULTADO DA VALIDAÇÃO DO PROBLEMA", f"result={result}", C.MAGENTA)
        return {"result": result}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Pré-sessão: extração de problema da imagem ────────────────────────────
#
# aumente max_px ou quality se o texto do problema não estiver sendo extraído com precisão
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
    """Lê uma imagem enviada e extrai qualquer problema/pergunta visível nela."""
    try:
        contents          = await image.read()
        contents, mime_type = compress_image(contents)
        image_b64         = base64.b64encode(contents).decode("utf-8")

        dbg("EXTRAIR PROBLEMA", f"mime={mime_type}  bytes={len(contents)}", C.BLUE)

        vision_prompt = (
            "Olhe para esta imagem e extraia exatamente o problema ou pergunta mostrada.\n"
            "Retorne apenas JSON:\n"
            '{"found": true, "problem": "o texto completo do problema extraído"}\n'
            "Se não houver problema ou pergunta clara visível, retorne:\n"
            '{"found": false, "problem": null}\n'
            "Sem markdown, sem texto extra."
        )

        raw    = await call_llama_vision(image_b64, mime_type, vision_prompt, max_tokens=1024)

        parsed  = extract_json(raw)
        found   = bool(parsed.get("found", False))
        problem = parsed.get("problem") if found else None
        dbg("RESULTADO DA EXTRAÇÃO DO PROBLEMA", f"found={found}  problem={problem}", C.MAGENTA)
        return {"found": found, "problem": problem}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Endpoint do professor ────────────────────────────────────────────────────
# Cadeia de chamadas por turno:
#   Fase 0  (apenas primeiro turno) — gera passos da solução; raciocínio sempre ativo (referência correta para o professor; precisão é crucial)
#   Fase 1  (todo turno)            — diagnostica estado do aluno + seleciona papel;
#                                raciocínio controlado pelo usuário via toggle da tela inicial
#   Fase 2  (todo turno)            — gera resposta do professor
#
# papéis do professor: inject_info · confirm_and_advance · partial_worked_step · redirect · session_close
# partial_worked_step acompanha quantos turnos consecutivos o aluno pediu
# ajuda no mesmo passo — o suporte escala low → medium → high a cada turno consecutivo
#
# session_state é um dict com o problema, passos da solução, estado anterior do aluno,
# última resposta do professor e contador de partial_worked_step — enviado de volta pelo frontend a cada turno.
# No primeiro turno ele é null; o cliente envia problem_text e student_comment.

@app.post("/teacher")
async def teacher(
    session_state: str     = Form(None), # JSON or null on first turn
    problem_text: str    = Form(None),   # apenas primeiro turno
    student_comment: str = Form(""),     # apenas primeiro turno — may be ""
    student_response: str = Form(None),  # turnos seguintes
    use_thinking: bool   = Form(True),   # toggle da tela inicial — ativa/desativa raciocínio
):
    try:
        state      = json.loads(session_state) if session_state else None
        first_turn = state is None

        # Em turnos seguintes o cliente não reenvia o toggle, então ele é lido
        # do state. No primeiro turno o cliente envia explicitamente.
        if not first_turn:
            use_thinking = state.get("use_thinking", True)

        # ── Resolver entradas ────────────────────────────────────────────
        if first_turn:
            problem        = (problem_text or "").strip()
            student_input  = (student_comment or "").strip()
            prior_status   = ""
            prior_reply    = ""
            solution_steps = ""   # preenchido pela Fase 0 abaixo
            
        else:
            problem        = state.get("formal_problem", "")
            prior_status   = state.get("prior_status", "")
            prior_reply    = state.get("last_teacher_reply", "")
            solution_steps = state.get("solution_steps", "")
            student_input  = (student_response or "").strip()
            

        # ── Fase 0: gerar passos da solução (apenas primeiro turno) ────────
        if first_turn:
            steps_prompt = build_solution_steps_prompt(problem)

            steps_raw = await call_llama_completion(steps_prompt, n_predict=256, temperature=0, thinking=True, label="FASE 0", color=C.YELLOW)

            try:
                steps_parsed   = extract_json(steps_raw)
                steps_list     = steps_parsed.get("solution_steps", [])
                solution_steps = "\n".join(steps_list)
            except Exception as e:
                dbg("ERRO DE PARSING NA FASE 0", f"{e}\n\nRaw output was:\n{steps_raw}", C.RED)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Parsing da Fase 0 falhou: {e}", "raw": steps_raw},
                )

            
        # ── Fase 1: diagnosticar estado do aluno + selecionar papel (mesclado) ───
        # Bypass: se o aluno não disser nada no comentário, é possível pular
        # totalmente a chamada ao LLM para reduzir latência. O papel padrão é inject_info.
        if first_turn and not student_input:
            student_status = "O aluno ainda não comentou e não sabe por onde começar."
            role    = "inject_info"
            dbg("FASE 1 PULADA (comentário vazio)", student_status, C.MAGENTA)
        else:
            phase1_prompt = build_status_and_role_prompt(
                problem=problem,
                solution_steps=solution_steps,
                student_input=student_input,
                prior_status=prior_status,
                prior_teacher_reply=prior_reply,
                first_turn=first_turn,
            )
            phase1_raw = await call_llama_completion(phase1_prompt, n_predict=192, temperature=0, thinking=use_thinking, label="FASE 1", color=C.CYAN)

            try:
                phase1_parsed  = extract_json(phase1_raw)
                student_status = phase1_parsed.get("student_status", prior_status)
                role           = phase1_parsed.get("role", "hint")
            except Exception as e:
                dbg("ERRO DE PARSING NA FASE 1", f"{e}\n\nRaw output was:\n{phase1_raw}", C.RED)
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Parsing da Fase 1 falhou: {e}", "raw": phase1_raw},
                )


        # ── Escalonamento adaptativo de suporte (apenas partial_worked_step) ────
        prior_count = (state or {}).get("partial_worked_count", 0)
        if role == "partial_worked_step":
            count  = prior_count + 1
            support = {1: "low", 2: "medium"}.get(count, "high")
        else:
            count  = 0
            support = None

        dbg("SAÍDA DA FASE 1", f"student_status: {student_status}\nrole: {role}\ncount: {count} | support: {support}", C.CYAN)

        # Próximo estado compartilhado (escrito antes da Fase 2 para session_close poder reutilizá-lo)
        next_state = {
            "formal_problem":        problem,
            "prior_status":          student_status,
            "last_teacher_reply":      "",        # preenchido após a Fase 2
            "solution_steps":        solution_steps,
            "partial_worked_count": count,
            "use_thinking":          use_thinking,
        }

        # ── Encerramento da sessão — pula chamada ao LLM para reduzir latência ───────
        if role == "session_close":
            close_msg = "Correto. Isso conclui o problema. Muito bem! 😊"
            dbg("ENCERRAMENTO DA SESSÃO", "", C.RED)
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

        # ── Fase 2: gerar resposta do professor ─────────────────────────────
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
            label="FASE 2",
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


# ── Geração de título (mantida em /v1/chat/completions) ──────────────────

@app.post("/generate-title")
async def generate_title(messages: str = Form(...)):
    try:
        history = json.loads(messages)

        # Usa o primeiro par usuário + assistente como contexto
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
                        "Você gera títulos ultra-curtos para chats. "
                        "Responda com APENAS 2-5 palavras que resumem o tópico da conversa. "
                        "Sem pontuação, sem aspas, sem explicação. Em Português."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Resuma esta conversa em 2-5 palavras:\n\n{prompt}",
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