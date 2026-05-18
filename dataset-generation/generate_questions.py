"""
Generates high-school level seed questions and accompanying solution steps for
producing Port-a-Prof's training dataset, using Gemma 3 12B IT QAT via Ollama.

Outputs:
- coding.json
- calculus.json
- algebra.json
- chemistry.json
- physics.json
- probability.json
- geometry.json

Run with:
python generate_questions.py
"""

# Imports required for JSON output, Ollama API calls, timestamps, and file paths
import json
import requests
from datetime import datetime
from pathlib import Path

# Global configuration for the local Ollama endpoint, model, and output directory
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:12b-it-qat"
OUTPUT_DIR = Path("questions")  

# Subject categories and subtopics used to guide synthetic question generation
CATEGORIES: dict[str, list[str]] = {
    "coding": [
        "simple if/else branch selection",
        "basic boolean expressions (without nesting)",
        "short fixed loops with at most 3 iterations",
        "simple function output (with no loops)",
        "tracing sequential assignments",
        "modulo operations"
    ],
    "calculus": [
        "basic derivatives",
        "derivative at a point",
        "evaluating definite integrals",
        "finding distance from velocity",
        "simple exponential growth",
        "finding critical points from a simple derivative"
    ],
    "algebra": [
        "solving a quadratic by factoring",                                      
        "solving a linear equation with variables on both sides",                
        "solving a linear system by elimination given two explicit equations", 
        "solving a proportion word problem using cross multiplication",          
        "solving an absolute value equation with extraneous root checking",     
        "evaluating and simplifying an algebraic expression using order of operations", 
    ],
    "chemistry": [
        "mole calculations and molar mass",
        "converting from mass to moles",
        "stoichiometry from balanced equations",
        "limiting reagent with simple mole ratios",
        "simple pH calculations (strong acids/bases)",
        "dilution and concentration calculations",
    ],
    "physics": [
        "constant-acceleration motion",
        "vertical motion under gravity",
        "Newton's second law and forces",
        "calculating work done (W = Fd)",
        "momentum (p = mv)",
        "using v = u + at"
    ],
    "probability": [
        "calculating probability using a sample space",
        "calculating probability using complementary events",
        "calculating expected value for a discrete probability distribution",
        "finding the probability of consecutive independent events",
        "calculating the number of arrangements using basic permutations",
        "calculating simple conditional probability from a word problem scenario"
    ],
    "geometry": [
        "finding the circumference of a circle from a given area",
        "determining the hypotenuse using cosine ratio",
        "determining a missing endpoint using the midpoint formula",
        "finding the missing side of a right triangle using Pythagoras theorem",
        "determining the length of a line segment using the distance formula",
        "finding the total interior angle sum of a polygon",
    ]
}

# Adds stricter generation rules for coding questions so they test tracing/evaluation, not program writing
def category_extra_rules(category: str) -> str:
    if category.lower() != "coding":
        return ""

    return """
- The question MUST require the student to evaluate, trace, predict, or determine the result of given logic.
- The question should NOT require the student to write, design, implement, or create a program.
- The question must be fully self-contained: include all variables, values, and code or pseudocode required to solve it.
Code formatting:
- Use single backticks for inline, single-line code only.
- Inline format: "question": "Given `x = 5; y = x + 2`, what is y?"
- All backticks must be properly opened and closed.
"""

# Builds the prompt sent to Gemma for one category/subtopic question-generation request
def build_prompt(category: str, subtopic: str) -> str:
    extra_rules = category_extra_rules(category)

    return f"""You are an expert educator. Generate a single {category} problem based on this subtopic: {subtopic}.

Rules for the question:
- Write it as 1 to 3 concise sentences — a single, self-contained problem statement.
- The problem must be deterministic with a clear solution path.
- The problem must naturally require multiple sequential calculations or reasoning steps to reach the answer.
- Use specific numbers, variables, and context so it is unambiguous.
{extra_rules}
Rules for the solution:
- Provide a numbered list of steps — "1.", "2.", "3.", etc.
- Each step is one logical operation or calculation, shown explicitly with working.
- The final step must include the final answer naturally as part of the calculation or conclusion.
- Do not include a separate step that restates the final answer.

Return ONLY a valid JSON object (no markdown fences, no explanation) with exactly these keys:
  "subtopic" : string        — the subtopic provided above
  "question" : string        — the concise problem statement (1–3 sentences)
  "solution" : array[string] — ordered solution steps, each showing the working
"""

# Sends one generation request to Ollama and parses the returned JSON question object
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

# Saves all generated questions for one category into a structured JSON file
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
    print(f"  Saved → {path}\n")

# Runs the full dataset generation pipeline across all categories and subtopics
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_categories = len(CATEGORIES)
    total_questions = sum(len(v) for v in CATEGORIES.values())

    print(f"\nGenerating {total_questions} questions with {MODEL} via Ollama\n")
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
    print(f"   All done in {elapsed:.1f}s")
    print(f"   Files: {', '.join(f'{c}.json' for c in CATEGORIES)}")


if __name__ == "__main__":
    main()