"""
Gerador de dataset sintético aluno-professor do Port-a-Prof.

Lê os arquivos de perguntas (ex.: questions/algebra.json) produzidos por
generate_questions.py e gera trajetórias sintéticas de aprendizagem aluno-professor
em múltiplos turnos para ajuste fino supervisionado.

Cada trajetória gerada simula uma interação aluno-professor em que um aluno avança
em um problema com diferentes níveis de compreensão, erros e suporte do professor.

── Formato de entrada ──
Cada arquivo JSON de categoria deve ter a estrutura:

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

Os passos da solução atuam como referência interna correta de raciocínio,
usada para orientar a geração do diálogo e validar o comportamento do aluno.

── Pipeline de geração do dataset ──
Para cada problema, o gerador instancia cinco modelos fixos de trajetória
definidos em TRAJECTORY_TEMPLATES.

Cada trajetória contém 2 a 3 turnos conversacionais.

Para cada turno, o sistema executa duas operações sequenciais com LLM:

1. Geração do diálogo
   generate_pair()

   Gera:
   - uma mensagem do aluno
   - uma resposta do professor

   A geração é condicionada por:
   - o problema
   - os passos internos da solução
   - o histórico da conversa
   - o modelo de trajetória
   - as restrições de papel do professor + nível de suporte

2. Rotulagem do estado interno
   label_state()

   Após cada turno do aluno, o sistema gera um rótulo estruturado
   de supervisão que descreve o estado de aprendizagem mais recente do aluno.

   Os rótulos incluem:
   - current_status
       Descrição em terceira pessoa da compreensão atual, concepção equivocada,
       erro de procedimento ou progresso do aluno.

   - teacher_role
       Estratégia usada pelo professor
       (redirect, partial_worked_step, inject_info, confirm_and_advance, session_close).

   - support (apenas para partial_worked_step)
       low / medium / high.

Esses rótulos de estado interno são usados como sinal de supervisão durante o
ajuste fino do Port-a-Prof, para que o modelo aprenda o comportamento/estilo de
cada papel pedagógico.

Observe que o modo de raciocínio é habilitado tanto na geração do diálogo quanto
na rotulagem de estado para incentivar interações de múltiplos turnos mais
coerentes, identificação mais precisa de concepções equivocadas do aluno e maior
consistência com os passos corretos da solução.

── Formato de saída ──
Um arquivo de saída é gerado por problema:

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

Cada entrada representa um exemplo de treinamento supervisionado.

── Robustez e validação ──
Todas as chamadas ao LLM passam por call_with_retry() com novas tentativas automáticas.

extract_json() trata automaticamente saídas malformadas comuns:
- JSON envolto em markdown
- sequências de escape LaTeX inválidas
- formatação JSON parcialmente corrompida

Quaisquer falhas são registradas em:
failures_<timestamp>.json

── CLI ── 
python generate_dataset.py                         # gera trajetórias para todas as perguntas
python generate_dataset.py --limit 5               # processa apenas as primeiras 5 perguntas
python generate_dataset.py --skip-existing         # pula perguntas com arquivos de saída existentes
python generate_dataset.py --workers 3             # cria trajetórias em paralelo usando 3 workers
python generate_dataset.py --verbose               # imprime prompts, saídas brutas do modelo e JSON interpretado
python generate_dataset.py --model gemma4:e4b      # substitui o modelo padrão (gemma4:e4b) por outro modelo Ollama
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

# Configuração global de inferência do Ollama, geração, caminhos e comportamento em execução
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

# Registra falhas de geração, parsing e validação para inspeção posterior
def log_failure(q_label: str, subtopic: str, failure_type: str, detail: str):
    with FAILURE_LOCK:
        FAILURE_LOG.append({
            "q_label":      q_label,
            "subtopic":     subtopic,
            "failure_type": failure_type,
            "detail":       detail,
        })

# Definições de comportamento para os papéis fixos do professor usados na geração de diálogo
ROLE_DESCRIPTIONS = {
    "session_close": (
        "Confirme que a resposta final do aluno está correta em uma frase curta e amigável. "
        "Depois pergunte se ele tem outros problemas em que gostaria de trabalhar. "
        "NÃO introduza conteúdo novo, não reexplique passos e não faça novas perguntas sobre o problema."
    ),
    "inject_info": (
        "Explique de forma clara e simples o conceito, regra ou teoria relevante necessária para resolver o problema. "
        "Use formatação LaTeX para equações. Use parágrafos curtos, quebras de linha e **texto em negrito** quando ajudar. "
        "NÃO resolva o problema nem execute passos da solução — forneça apenas os conceitos e a teoria necessários. "
        "No final da explicação, conecte-a ao problema atual e faça uma pergunta orientadora que ajude o aluno "
        "a aplicar a teoria para avançar no problema."
    ),
    "redirect": (
        "O aluno cometeu um erro aritmético ou procedural específico. "
        "Chame a atenção dele para o passo ou valor exato que está errado — "
        "nomeie diretamente a operação ou expressão, mas não diga o valor correto nem refaça o cálculo. "
        "Peça ao aluno que confira ou refaça esse passo específico por conta própria. "
        "No máximo duas frases."
    ),
    "confirm_and_advance": (
        "Confirme o que o aluno fez corretamente em tom amigável, mas sem exagero, focando no método ou raciocínio. "
        "NÃO ensine novamente, explique ou introduza novos conceitos. "
        "NÃO afirme nem elogie o aluno por lembrar ou repetir informações dadas pelo professor em ## HISTÓRICO DA CONVERSA. "
        "NÃO forneça passos resolvidos, cálculos ou resultados parciais. "
        "Faça uma pergunta clara sobre o próximo passo, direta e acionável, mas sem incluir a resposta nem executar o passo. "
        "No máximo duas frases."
    ),
}

# Especificações para controlar a intensidade do suporte em partial_worked_step
SUPPORT_DESCRIPTIONS = {
    "partial_worked_step": {
        "low": (
            "NÃO elogie, afirme ou diga ao aluno que ele está correto em nenhuma circunstância. Foque apenas no necessário para o aluno avançar no problema. "
            "Escreva a equação, expressão ou relação para o marco específico em que o aluno está trabalhando, usando apenas forma simbólica ou com variáveis — "
            "NÃO substitua valores específicos do problema. "
            "O marco deve representar uma parte significativa da lógica — não um passo trivial que seria óbvio sem orientação."
            "Use $$ ... $$ para equações isoladas e $ ... $ para referências matemáticas inline dentro de uma frase. "
            "Use **texto em negrito** para destacar termos ou valores importantes e separe ideias distintas com \\n para clareza. "
            "Inclua explicações claras e concisas quando apropriado. "
            "Termine com uma próxima ação clara para o aluno."
        ),
        "medium": (
            "NÃO elogie, afirme ou diga ao aluno que ele está correto em nenhuma circunstância. Foque apenas no necessário para o aluno avançar no problema. "
            "NÃO repita nem reformule nada do diálogo do professor em ## HISTÓRICO DA CONVERSA, a menos que seja necessário. "
            "Conecte diretamente à configuração simbólica já fornecida e estenda-a — "
            "identifique e defina as variáveis relevantes do problema, incluindo qualquer raciocínio necessário "
            "sobre como elas são obtidas, mas NÃO substitua, simplifique nem calcule. "
            "Use $$ ... $$ para equações isoladas e $ ... $ para referências matemáticas inline dentro de uma frase. "
            "Use **texto em negrito** para destacar termos ou valores importantes e separe ideias distintas com \\n para clareza. "
            "Termine pedindo ao aluno que substitua os valores e continue a partir da configuração."
        ),
        "high": (
            "NÃO elogie, afirme ou diga ao aluno que ele está correto em nenhuma circunstância. Foque apenas no necessário para o aluno avançar no problema. "
            "NÃO repita nem reformule nada do diálogo do professor em ## HISTÓRICO DA CONVERSA, a menos que seja necessário. "
            "Usando as variáveis e a configuração já definidas, conduza a avaliação. "
            "Inclua uma breve explicação para operações ou transformações que não sejam óbvias. "
            "Use $$ ... $$ para equações isoladas e $ ... $ para referências matemáticas inline dentro de uma frase. "
            "Use **texto em negrito** para destacar termos ou valores importantes e separe linhas distintas de desenvolvimento com \\n para clareza. "
            "Pare antes de uma pequena ação final (por exemplo, a última operação aritmética, rearranjo final ou valor final). "
            "Termine com uma instrução clara pedindo ao aluno que complete essa ação restante."
        ),
    },
}

# Regras de suporte específicas para programação usando formatação inline de código em vez de LaTeX
SUPPORT_DESCRIPTIONS_CODING = {
    "partial_worked_step": {
        "low": (
            "NÃO elogie, afirme ou diga ao aluno que ele está correto em nenhuma circunstância. Foque apenas no necessário para o aluno avançar no problema. "
            "Escreva a construção, expressão ou condição relevante para o marco específico em que o aluno está trabalhando, usando apenas nomes de variáveis — "
            "não substitua valores específicos do problema. "
            "O marco deve representar uma parte significativa da lógica — não um passo trivial que seria óbvio sem orientação."
            "Use crases inline (`...`) para todo código, nomes de variáveis e operadores. "
            "NÃO use LaTeX nem blocos de código em nenhuma circunstância. "
            "Use **texto em negrito** para destacar termos importantes e separe ideias distintas com \\n para clareza. "
            "Inclua explicações claras e concisas quando apropriado. "
            "Termine com uma próxima ação clara para o aluno."
        ),
        "medium": (
            "NÃO elogie, afirme ou diga ao aluno que ele está correto em nenhuma circunstância. Foque apenas no necessário para o aluno avançar no problema. "
            "NÃO repita nem reformule nada do diálogo do professor em ## HISTÓRICO DA CONVERSA, a menos que seja necessário. "
            "Conecte diretamente à construção já fornecida e estenda-a — "
            "identifique e defina as variáveis relevantes do problema, incluindo qualquer raciocínio necessário "
            "sobre como elas são obtidas, mas não avalie, resolva nem calcule o resultado. "
            "Use crases inline (`...`) para todo código, nomes de variáveis e operadores. "
            "NÃO use LaTeX nem blocos de código em nenhuma circunstância. "
            "Use **texto em negrito** para destacar termos importantes e separe ideias distintas com \\n para clareza. "
            "Termine pedindo ao aluno que avalie ou continue a partir da construção."
        ),
        "high": (
            "NÃO elogie, afirme ou diga ao aluno que ele está correto em nenhuma circunstância. Foque apenas no necessário para o aluno avançar no problema. "
            "NÃO repita nem reformule nada do diálogo do professor em ## HISTÓRICO DA CONVERSA, a menos que seja necessário. "
            "Usando as variáveis e a construção já definidas, conduza a avaliação. "
            "Inclua uma breve explicação para operações ou transformações que não sejam óbvias. "
            "Use crases inline (`...`) para todo código, nomes de variáveis e operadores. "
            "NÃO use LaTeX nem blocos de código em nenhuma circunstância. "
            "Use **texto em negrito** para destacar termos importantes e separe linhas distintas de desenvolvimento com \\n para clareza. "
            "Pare antes de uma pequena ação final (por exemplo, a última verificação de condição, atribuição ou valor de retorno). "
            "Termine com uma instrução clara pedindo ao aluno que complete essa ação restante."
        ),
    },
}

# Modelos fixos de trajetória de aprendizagem usados para sintetizar conversas de múltiplos turnos
TRAJECTORY_TEMPLATES = [
    {
        "trajectory_id": "traj_1",
        "student_path": [
            "O aluno afirma que não sabe como começar o problema.",
            "O aluno reconhece a explicação do professor, mas não sabe como aplicá-la a este problema específico. "
            "Ele pede ajuda ao professor com o passo. "
            "A resposta final não aparece."
        ],
        "teacher_turns": [
            {"role": "inject_info",         "support": None},
            {"role": "partial_worked_step", "support": "low"},
        ],
    },
    {
        "trajectory_id": "traj_2",
        "student_path": [
            "O aluno tenta um marco inicial significativo, mas comete um erro procedural — "
            "um sinal errado, aritmética incorreta ou um erro simples de cálculo. "
            "A resposta final não aparece.",
            "O aluno corrige seu erro procedural do turno \"student\" em ## HISTÓRICO DA CONVERSA, "
            "chegando ao valor correto para esse passo. "
            "Depois expressa incerteza sobre o que fazer em seguida e pede ajuda para montar o próximo passo. "
            "A resposta final não aparece."
        ],
        "teacher_turns": [
            {"role": "redirect", "support": None},
            {"role": "partial_worked_step", "support": "low"},
        ],
    },
    {
        "trajectory_id": "traj_3",
        "student_path": [
            "O aluno está perto do fim da solução e completa corretamente o penúltimo marco significativo. "
            "Isso pode envolver encontrar o último valor intermediário necessário, substituir valores conhecidos na expressão final ou montar o cálculo final. "
            "Ele para antes de calcular ou declarar a resposta final. "
            "A resposta final não aparece.",
            "O aluno completa o cálculo final ou o passo final de raciocínio e declara a resposta final correta. "
            "A resposta final deve coincidir exatamente com o resultado final indicado pelos passos internos da solução."
        ],
        "teacher_turns": [
            {"role": "confirm_and_advance", "support": None},
            {"role": "session_close",       "support": None},
        ],
    },
    {
        "trajectory_id": "traj_4",
        "student_path": [
            "O aluno completa corretamente o primeiro marco significativo e depois pede ajuda para montar o próximo passo.",
            "O aluno usa a configuração dada no turno \"teacher\" em ## HISTÓRICO DA CONVERSA, "
            "mas comete um erro aritmético ou lógico durante a avaliação do passo — "
            "por exemplo, combina os termos errados, aplica a operação errada "
            "ou calcula incorretamente — resultando em um valor incorreto para esse passo. "
            "A resposta final não aparece."
        ],
        "teacher_turns": [
            {"role": "partial_worked_step", "support": "low"},
            {"role": "redirect",            "support": None},
        ],
    },
    {   "trajectory_id": "traj_5",
        "student_path": [
            "O aluno não sabe como começar o problema. "
            "Ele pede ajuda com a montagem.",

            "O aluno fez progresso ZERO a partir da explicação do professor em ## HISTÓRICO DA CONVERSA — "
            "ele não demonstra compreensão do que foi fornecido. "
            "Ele NÃO sabe como obter as variáveis necessárias e pergunta de onde vêm os valores. "
            "O aluno NÃO realizou nenhum cálculo. "
            "A resposta final NÃO aparece.",

            "O aluno fez progresso ZERO a partir da explicação do professor em ## HISTÓRICO DA CONVERSA — "
            "ele não demonstra compreensão do que foi fornecido. "
            "Ele NÃO sabe como realizar a avaliação e pede para ver como desenvolvê-la. "
            "O aluno NÃO realizou nenhum cálculo. "
            "A resposta final NÃO aparece.",
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

# Envia uma requisição de geração ao Ollama e retorna a saída bruta limpa do modelo
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

    # Registra o raciocínio se estiver presente
    if thinking and VERBOSE:
        scratchpad = data.get("thinking", "").strip()
        if scratchpad:
            _vprint("RASCUNHO DE RACIOCÍNIO ←", scratchpad, "", tag)

    # Remove tokens especiais
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
        _vprint("SAÍDA BRUTA DO MODELO ←", raw, "raw", tag)
    return raw

# Repara escapes LaTeX inválidos com barra única antes de interpretar o JSON
def _fix_latex_escapes(s: str) -> str:
    _KEEP = frozenset({'"', '\\', '/', 'n', 'u'})
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == '\\':
                # Já tem barra dupla — mantém e pula ambos os caracteres
                result.append('\\\\')
                i += 2
            elif next_char in _KEEP:
                # Escape JSON intencional — mantém como está
                result.append('\\')
                result.append(next_char)
                i += 2
            else:
                # LaTeX com barra única — duplica apenas a barra,
                # deixando o próximo caractere ser processado normalmente na próxima iteração
                result.append('\\\\')
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)

# Extrai recursivamente todos os valores de string de um objeto JSON interpretado
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

# Detecta corrupção silenciosa de LaTeX causada por parsing de escapes JSON inválidos
def _check_post_parse_corruption(obj) -> list:
    _CORRUPT = {'\x08': r'\b', '\x0c': r'\f', '\r': r'\r', '\t': r'\t'}
    issues = []
    for s in _scan_string_values(obj):
        for char, name in _CORRUPT.items():
            if char in s:
                issues.append(
                    f"string contém caractere de controle {name} "
                    f"(provável LaTeX sem escape, ex. \\{name[1:]}ext / \\frac / \\rho / \\beta)"
                )
    return issues

# Extrai e valida um objeto JSON da saída bruta do modelo
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

    # Tentativa 1: interpretar como está
    result = _try_parse(candidate)
    if result is not None:
        return result

    # Tentativa 2: corrigir LaTeX com barra única (ex.: \ge → \\ge) e tentar novamente
    result = _try_parse(_fix_latex_escapes(candidate))
    if result is not None:
        return result

    # Fallback: encontrar o primeiro objeto com aparência de JSON e repetir as duas tentativas
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

# Executa uma chamada de geração ao Ollama com nova tentativa e validação
def call_with_retry(prompt: str, validator=None, label: str = "", tag: str = "", q_label: str = "", subtopic: str = "", thinking: bool = False):
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw    = call_ollama(prompt, tag=tag, thinking=thinking)
            result = extract_json(raw)
            if result is None:
                raise ValueError("Nenhum JSON válido encontrado na resposta")
            if validator and not validator(result):
                raise ValueError(f"Validação falhou ({type(result).__name__})")
            if VERBOSE:
                _vprint(f"PARSED [{label}] ✓", json.dumps(result, indent=2), "parsed", tag)
            return result
        except Exception as e:
            last_error = str(e)
            if VERBOSE:
                with PRINT_LOCK:
                    print(f"        [{tag or label}] tentativa {attempt}/{MAX_RETRIES} falhou: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    # Todas as tentativas esgotadas — registra a falha
    if q_label:
        log_failure(q_label, subtopic, label, f"{tag}: todas as {MAX_RETRIES} tentativas falharam: {last_error}")
    return None

# Monta a especificação comportamental de um par de interação aluno-professor
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

# Gera uma mensagem do aluno e uma resposta do professor para um passo da trajetória
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
    """Retorna {"student": "...", "teacher": "..."}"""
    steps_str    = "\n".join(steps)
    pair_spec    = _pair_spec(template, pair_index, is_coding=is_coding)
    current_role = template["teacher_turns"][pair_index]["role"]

    context = ""
    if previous_dialogue:
        context = (
            "\n## HISTÓRICO DA CONVERSA\n"
            + "\n".join(
                f'  "{m["role"]}": {m["content"]}'
                for m in previous_dialogue
            )
            + "\n"
        )

    answer_rule = (
        "O aluno DEVE declarar a resposta final correta em seu turno."
        if current_role == "session_close" else
        "O aluno NÃO deve declarar a resposta final correta em seu turno. "

    )

    if is_chemistry:
        teacher_rules = """
## REGRAS DO PROFESSOR
- Não comece com elogio ou incentivo. Nunca inicie com "Ótimo", "Bom trabalho", "Muito bem", "Boa", "Excelente" ou qualquer afirmação genérica.
- Responda à formulação exata do aluno; não dê uma resposta genérica ou padronizada.
- Seja claro, envolvido e conversacional, mantendo precisão.
- Nunca dê a resposta final diretamente.
- Mantenha respostas concisas e direcionadas. Não reafirme informações já conhecidas.
- Fundamente todas as explicações no problema atual usando valores ou expressões reais.
- Escreva todo LaTeX com barras simples: $$\\ge$$, $$\\le$$, $$\\frac{{a}}{{b}}$$
- Use LaTeX para equações: $$ ... $$ para equações isoladas e $ ... $ para referências matemáticas inline dentro de uma frase.
- Use \\n para quebras de linha. Não inclua quebras de linha literais dentro de strings.
- Faça escape de aspas duplas dentro de strings como \\"
- Todas as fórmulas químicas devem usar apenas subscritos Unicode: H₂O, CuSO₄, Cu(OH)₂
- NUNCA elogie ou diga ao aluno que ele está correto por identificar quantidades triviais fornecidas explicitamente no problema"""
    elif is_coding:
        teacher_rules = """\
## REGRAS DO PROFESSOR
- Não comece com elogio ou incentivo. Nunca inicie com "Ótimo", "Bom trabalho", "Muito bem", "Boa", "Excelente" ou qualquer afirmação genérica.
- Responda à formulação exata do aluno; não dê uma resposta genérica ou padronizada.
- Seja claro, envolvido e conversacional, mantendo precisão.
- Nunca dê a resposta final diretamente.
- Mantenha respostas concisas e direcionadas. Não reafirme informações já conhecidas.
- Fundamente todas as explicações no problema atual usando valores ou expressões reais.
- Use crases inline para TODO código, nomes de variáveis, palavras-chave e operadores: `age`, `if`, `else`, `>=`.
- NÃO use LaTeX.
- Use APENAS crases inline (`...`) para código. NÃO use blocos de código (``` ... ```) em nenhuma circunstância.
- Todo trecho de código deve caber em uma única linha dentro das crases. Se não couber, divida em referências inline separadas.
- Use \\n para quebras de linha. Não inclua quebras de linha literais dentro de strings.
- Faça escape de aspas duplas dentro de strings como \\"
- NUNCA elogie ou diga ao aluno que ele está correto por identificar quantidades triviais fornecidas explicitamente no problema"""
    else:
        teacher_rules = """\
## REGRAS DO PROFESSOR
- Não comece com elogio ou incentivo. Nunca inicie com "Ótimo", "Bom trabalho", "Muito bem", "Boa", "Excelente" ou qualquer afirmação genérica.
- Responda à formulação exata do aluno; não dê uma resposta genérica ou padronizada.
- Seja claro, envolvido e conversacional, mantendo precisão.
- Nunca dê a resposta final diretamente.
- Mantenha respostas concisas e direcionadas. Não reafirme informações já conhecidas.
- Fundamente todas as explicações no problema atual usando valores ou expressões reais.
- Escreva todo LaTeX com barras simples: $$\\ge$$, $$\\le$$, $$\\frac{{a}}{{b}}$$
- Use LaTeX para equações: $$ ... $$ para equações isoladas e $ ... $ para referências matemáticas inline dentro de uma frase.
- Use \\n para quebras de linha. Não inclua quebras de linha literais dentro de strings.
- Faça escape de aspas duplas dentro de strings como \\" 
- NUNCA elogie ou diga ao aluno que ele está correto por identificar quantidades triviais fornecidas explicitamente no problema"""

    prompt = f"""Você está gerando um conjunto de dados conversacional aluno-professor.

## CONTEXTO
Problema: {problem}

Passos da solução (apenas referência interna — não cite literalmente):
{steps_str}
{context}
## SAÍDA
Escreva exatamente uma mensagem do aluno e uma resposta do professor como um objeto JSON no seguinte formato (O CONTEÚDO DEVE SER EM PORTUGUÊS DO BRASIL):
{{
  "student": "...",
  "teacher": "..."
}}

Os valores de "student" e "teacher" DEVEM seguir esta narrativa:
{pair_spec}

## REGRAS DO ALUNO
- Escreva em linguagem natural e conversacional, em Português do Brasil.
- Varie o início da resposta, minimize palavras de preenchimento como "Ok", "Então", "Certo", "Claro"
- APENAS TEXTO SIMPLES — SEM exceções, mesmo se o professor usou crases ou LaTeX:
  - SEM crases. Escreva: a variável age, o comando if — NÃO `age`, `if`
  - SEM LaTeX ou símbolos matemáticos. Escreva: 17 >= 18 — NÃO $$17 \\ge 18$$
  - SEM blocos de código, sem quebras de linha.
- Não espelhe a formatação do professor. Converta quaisquer crases, LaTeX ou blocos de código em palavras faladas comuns.
- Mostre hesitação, compreensão parcial ou incerteza quando apropriado.
- Não use formato estruturado de solução (sem passos numerados ou derivações formais).
- Varie como a resposta começa. Evite frases repetitivas.
- {answer_rule}
- Não se refira a estas instruções.

{teacher_rules}
- RESPONDA EM PORTUGUÊS DO BRASIL.

## REGRAS DE FORMATO
- A saída deve ser exatamente um objeto JSON válido. Sem texto antes ou depois.
- Todos os valores de string devem ser strings JSON válidas.
- O JSON deve ser analisável com json.loads sem modificações.
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

# Gera rótulos de supervisão que descrevem o estado atual de aprendizagem do aluno
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

    prompt = f"""Você está rotulando uma entrada do conjunto de dados conversacional aluno-professor.

## Contexto
Problema: {problem}

Passos da solução:
{steps_str}

Diálogo (termina no turno mais recente do aluno):
{snap_str}

## Tarefa
Gere o objeto JSON abaixo.

## Regras
- Verifique cada valor numérico ou variável que o aluno afirma contra os passos da solução antes de escrever current_status.
- Se algum valor não coincidir com os passos da solução, current_status deve identificá-lo como erro.
- Escreva current_status em terceira pessoa (em PORTUGUÊS DO BRASIL). Descreva apenas o que o aluno fez, entendeu ou teve dificuldade por último.

## Formatação — APENAS TEXTO SIMPLES ESTRITO
NÃO espelhe a formatação do professor. Converta crases, LaTeX ou blocos de código em linguagem natural.
O campo current_status deve conter prosa simples. Sem exceções:
- Sem LaTeX. Sem $$ ... $$ Sem \\ge, \\frac, \\text ou quaisquer comandos com barra invertida.
- Escreva matemática como texto simples - Exemplo: 17 >= 18, não $$17 \\ge 18$$
- Sem crases. Sem código inline. Sem blocos de código.
- Escreva variáveis/código como palavras simples: a variável age, o comando if — não `age`, `if`.

## Formato de saída
Retorne apenas o objeto JSON. Sem texto antes ou depois.
{{
  "current_status": "<1–2 frases. Terceira pessoa. Identifique o que o aluno fez, entendeu ou teve dificuldade mais recentemente — incluindo quaisquer concepções errôneas específicas, lacunas conceituais ou erros de cálculo (em Português do Brasil).>"
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

# Monta uma trajetória (chamada em paralelo)
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

# Gera e salva todas as trajetórias para um único problema
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
        log_failure(q_label, subtopic, "question", "Nenhuma trajetória construída")
        return False

    if len(trajectories) < len(TRAJECTORY_TEMPLATES):
        log_failure(q_label, subtopic, "partial", f"Apenas {len(trajectories)}/{len(TRAJECTORY_TEMPLATES)} trajetórias construídas")

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

# Processa todas as perguntas dentro de um arquivo de dataset de categoria
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

# Ponto de entrada de linha de comando para geração completa do dataset de trajetórias
def main():
    global MODEL, VERBOSE
    parser = argparse.ArgumentParser(description="Gera datasets de trajetórias de ensino")
    parser.add_argument("--limit",         type=int,  default=None,
                        help="Número máximo de perguntas a processar em todos os arquivos de categoria")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--input-dir",     type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir",    type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model",         type=str,  default=MODEL)
    parser.add_argument("--workers",       type=int,  default=3,
                        help="Workers paralelos para geração de trajetórias (padrão: 3)")
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
            print(f"AVISO: '{MODEL}' não encontrado. Disponíveis: {available}")
            print(f"  Baixe com: ollama pull {MODEL}")
    except requests.ConnectionError:
        print("ERRO: Ollama não está em execução. Inicie com: ollama serve")
        return

    category_files = sorted(args.input_dir.glob("*.json"))
    total_questions = sum(
        len(json.loads(f.read_text(encoding="utf-8")).get("questions", []))
        for f in category_files
    )

    print(f"Modelo:     {MODEL}")
    print(f"Entrada:    {args.input_dir}/  ({len(category_files)} arquivos de categoria, {total_questions} perguntas)")
    print(f"Saída:      {args.output_dir}/")
    print(f"Workers:    {args.workers} trajetórias paralelas por pergunta")
    print(f"Modelos:    {len(TRAJECTORY_TEMPLATES)} fixos")
    if args.limit:
        print(f"Limite:     {args.limit} perguntas")
    if VERBOSE:
        print("Verboso:    ATIVO")

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
    print(f"Concluído. {total_success} com sucesso, {total_failed} com falha.")
    print(f"Saídas em '{args.output_dir}/'")

    # Write failure log
    log_path = args.output_dir / f"failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if FAILURE_LOG:
        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(FAILURE_LOG, lf, indent=2, ensure_ascii=False)
        print(f"Log de falhas → {log_path}  ({len(FAILURE_LOG)} entradas)")
    else:
        print("Nenhuma falha registrada.")

if __name__ == '__main__':
    main()
