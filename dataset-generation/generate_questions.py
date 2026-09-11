"""
Gera perguntas-base de nível ensino médio e os passos de solução correspondentes
para produzir o dataset de treinamento do Port-a-Prof, usando Gemma 3 12B IT QAT via Ollama.

Saídas:
- coding.json
- calculus.json
- algebra.json
- chemistry.json
- physics.json
- probability.json
- geometry.json

Execute com:
python generate_questions.py
"""

# Imports necessários para saída JSON, chamadas à API do Ollama, timestamps e caminhos
import json
import requests
from datetime import datetime
from pathlib import Path

# Configuração global do endpoint local do Ollama, modelo e diretório de saída
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:12b-it-qat"
OUTPUT_DIR = Path("questions")  

# Categorias de matérias e subtópicos usados para orientar a geração sintética de perguntas
CATEGORIES: dict[str, list[str]] = {
    "programação": [
        "seleção simples de ramo if/else",
        "expressões booleanas básicas (sem aninhamento)",
        "laços fixos curtos com no máximo 3 iterações",
        "saída de função simples (sem laços)",
        "rastreamento de atribuições sequenciais",
        "operações de módulo"
    ],
    "cálculo": [
        "derivadas básicas",
        "derivada em um ponto",
        "avaliação de integrais definidas",
        "encontrar distância a partir da velocidade",
        "crescimento exponencial simples",
        "encontrar pontos críticos a partir de uma derivada simples"
    ],
    "álgebra": [
        "resolver uma equação quadrática por fatoração",
        "resolver uma equação linear com variáveis em ambos os lados",
        "resolver um sistema linear por eliminação dadas duas equações explícitas",
        "resolver um problema verbal de proporção usando multiplicação cruzada",
        "resolver uma equação com valor absoluto verificando raízes extranhas",
        "avaliar e simplificar uma expressão algébrica usando a ordem das operações",
    ],
    "química": [
        "cálculos de mol e massa molar",
        "conversão de massa para mols",
        "estequiometria a partir de equações balanceadas",
        "reagente limitante com razões molares simples",
        "cálculos simples de pH (ácidos/bases fortes)",
        "cálculos de diluição e concentração",
    ],
    "física": [
        "movimento com aceleração constante",
        "movimento vertical sob a gravidade",
        "segunda lei de Newton e forças",
        "cálculo do trabalho realizado (W = Fd)",
        "quantidade de movimento (p = mv)",
        "uso de v = u + at"
    ],
    "probabilidade": [
        "calcular probabilidade usando um espaço amostral",
        "calcular probabilidade usando eventos complementares",
        "calcular valor esperado para uma distribuição discreta de probabilidade",
        "encontrar a probabilidade de eventos independentes consecutivos",
        "calcular o número de arranjos usando permutações básicas",
        "calcular probabilidade condicional simples a partir de um problema verbal"
    ],
    "geometria": [
        "encontrar a circunferência de um círculo a partir de uma área dada",
        "determinar a hipotenusa usando a razão do cosseno",
        "determinar um ponto extremo ausente usando a fórmula do ponto médio",
        "encontrar o lado ausente de um triângulo retângulo usando o teorema de Pitágoras",
        "determinar o comprimento de um segmento de reta usando a fórmula da distância",
        "encontrar a soma total dos ângulos internos de um polígono",
    ]
}

# Adiciona regras mais estritas para perguntas de programação testarem rastreamento/avaliação, não escrita de programas
def category_extra_rules(category: str) -> str:
    if category.lower() != "programação":
        return ""

    return """
- A pergunta DEVE exigir que o aluno avalie, rastreie, preveja ou determine o resultado da lógica fornecida.
- A pergunta NÃO deve exigir que o aluno escreva, desenhe, implemente ou crie um programa.
- A pergunta deve ser totalmente autossuficiente: inclua todas as variáveis, valores e código ou pseudocódigo necessários para resolvê-la.
Formatação de código:
- Use crases simples apenas para código inline em uma única linha.
- Formato inline: "question": "Dado `x = 5; y = x + 2`, qual é o valor de y?"
- Todas as crases devem ser abertas e fechadas corretamente.
"""

# Monta o prompt enviado ao Gemma para gerar uma pergunta de uma categoria/subtópico
def build_prompt(category: str, subtopic: str) -> str:
    extra_rules = category_extra_rules(category)

    return f"""Você é um educador especialista. Gere um único problema de {category} com base neste subtópico: {subtopic}.

Regras para a pergunta:
- Escreva-a em 1 a 3 frases concisas — uma única declaração de problema independente em Português do Brasil.
- O problema deve ser determinístico com um caminho de solução claro.
- O problema deve naturalmente exigir vários cálculos sequenciais ou etapas de raciocínio para chegar à resposta.
- Use números específicos, variáveis e contexto para não haver ambiguidade.
{extra_rules}
Regras para a solução:
- Forneça uma lista numerada de passos — "1.", "2.", "3.", etc. em Português do Brasil.
- Cada passo é uma operação lógica ou cálculo, mostrado explicitamente com o desenvolvimento.
- O passo final deve incluir a resposta final naturalmente como parte do cálculo ou conclusão.
- Não inclua um passo separado que apenas reafirme a resposta final.

Retorne APENAS um objeto JSON válido (sem marcadores markdown, sem explicação) com exatamente estas chaves:
  "subtopic" : string        — o subtópico fornecido acima (em Português do Brasil)
  "question" : string        — a declaração concisa do problema (1–3 frases)
  "solution" : array[string] — passos de solução ordenados, cada um mostrando o raciocínio
"""

# Envia uma requisição de geração ao Ollama e interpreta o objeto JSON de pergunta retornado
def fetch_question(category: str, subtopic: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(category, subtopic)}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.8,
            "num_predict": 1024,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    content = response.json()["message"]["content"]
    parsed = json.loads(content)

    if isinstance(parsed, list):
        parsed = parsed[0]

    return parsed

# Salva todas as perguntas geradas para uma categoria em um arquivo JSON estruturado
def save(category: str, questions: list[dict]) -> None:
    output = {
        "category": category,
        "model": MODEL,
        "generated_at": datetime.now().isoformat(),
        "total": len(questions),
        "questions": questions,
    }
    path = OUTPUT_DIR / f"{category}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Salvo → {path}\n")

# Executa o pipeline completo de geração do dataset em todas as categorias e subtópicos
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_categories = len(CATEGORIES)
    total_questions = sum(len(v) for v in CATEGORIES.values())

    print(f"\nGerando {total_questions} perguntas com {MODEL} via Ollama\n")
    start = datetime.now()

    for cat_num, (category, subtopics) in enumerate(CATEGORIES.items(), 1):
        print(f"[{cat_num}/{total_categories}] {category.upper()}")
        questions = []

        for q_num, subtopic in enumerate(subtopics, 1):
            print(f"  Q{q_num} - {subtopic} ... ", end="", flush=True)
            try:
                question = fetch_question(category, subtopic)
                questions.append(question)
                print("✓")
            except Exception as e:
                print(f"✗ ({e})")

        save(category, questions)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"   Tudo concluído em {elapsed:.1f}s")
    print(f"   Arquivos: {', '.join(f'{c}.json' for c in CATEGORIES)}")


if __name__ == "__main__":
    main()
