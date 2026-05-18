# Port-a-Prof: Deeper learning, wherever you are 💭
### Expanding access to AI-powered education, with no compromise to learning quality.
*An entry to [The Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)*

---

Port-a-Prof is an offline AI learning assistant designed to foster deep engagement with schoolwork.

Powered by a QLoRA fine-tuned variant of Gemma 4 E2B IT, it runs entirely on-device, keeping student data private while supporting text, audio, and image inputs.

The vision behind Port-a-Prof is to provide high-quality learning assistance to students regardless of location, connectivity, or financial circumstance — including through device access initiatives where donated devices are distributed to communities with Port-a-Prof pre-installed, making personalised learning support accessible to students who have historically been left out of it.

This repository includes:
- **Dataset generation** — scripts to generate seed questions and a multi-turn student-teacher dialogue dataset for fine-tuning, via Ollama
- **Fine-tuning** — a QLoRA training notebook built on Hugging Face Transformers, PEFT, TRL, and bitsandbytes
- **App** — a proof-of-concept interface built with FastAPI and a single-page HTML frontend, containerised with Docker

 
---
 
## Project Structure
 
```
PORT-A-PROF/
├── app/                        # Proof-of-concept app
│   ├── models/                 # GGUF model files (see below)
│   ├── static/                 # Static assets
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── index.html
|   |── logo.png
│   ├── port-a-prof.py          # FastAPI backend
│   └── README.md               # App setup instructions
│
├── dataset-generation/         # Synthetic training data pipeline
│   ├── dataset/                # Generated trajectory output
│   ├── questions/              # Generated seed questions
│   ├── generate_questions.py   # Step 1: generate seed questions via Ollama
│   └── generate_dataset.py     # Step 2: generate student-teacher dialogues via Ollama
│
├── fine-tuning/                # QLoRA fine-tuning
│   ├── dataset/
│   └── port-a-prof_QLoRA.ipynb
│
├── README.md
└── requirements.txt
```
 
---
 
## Quickstart
 
### 1. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
> **PyTorch** must be installed separately — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
 
---
 
### 2. Dataset Generation (requires Ollama)
 
[Ollama](https://ollama.com) must be running locally before running either generation script.
 
The following models are used by default — these can be changed to suit your preferences:
 
- **[gemma3:12b-it-qat](https://ollama.com/library/gemma3:12b-it-qat)** — seed question and solution generation (`generate_questions.py`)
- **[gemma4:e4b](https://ollama.com/library/gemma4:e4b)** — student-teacher dialogue generation (`generate_dataset.py`)
```bash
# Step 1 — generate seed questions
python dataset-generation/generate_questions.py
 
# Step 2 — generate student-teacher dialogues
python dataset-generation/generate_dataset.py
```
 
---
 
### 3. Fine-tuning
 
The fine-tuned model is designed for use as part of the Port-a-Prof system pipeline, not in isolation — it expects structured input to function correctly.
 
### Input Format
 
The model is trained on structured prompts with the following schema:
 
```
<bos><|turn>user
## PROBLEM
{problem text}
 
## STUDENT_ATTEMPT
{student's attempt or question}
 
## STATUS
{assessment of where the student is}
 
## TEACHER_ROLE
{partial_worked_step | redirect | confirm_and_advance | session_close | partial_worked_step}
 
## SUPPORT                          ← only present when TEACHER_ROLE is partial_worked_step
{low | medium | high}<turn|>
<|turn>model
```
 
- **PROBLEM** — the original question or task posed to the student
- **STUDENT_ATTEMPT** — the student's work, response, or where they are stuck
- **STATUS** — a diagnostic summary of the student's current understanding
- **TEACHER_ROLE** — the instructional strategy the model should adopt
- **SUPPORT** — the level of support to provide (`low`, `medium`, or `high`) in a partial worked step

### Getting the Model
 
You can either:
 
1. **Fine-tune it yourself** — open `fine-tuning/port-a-prof_QLoRA.ipynb` and follow the cells. The notebook produces a merged, full-precision model saved to `./port_a_prof_finetuned` in Hugging Face safetensors format, which can then be quantised and converted to GGUF using [llama.cpp](https://github.com/ggerganov/llama.cpp).
2. **Download the pre-converted GGUF** — grab the Q4_K_M quantised model directly from [Hugging Face](https://huggingface.co/bianca-lilyyy128/port-a-prof-Q4_K_M).
To run the app you will also need the Gemma 4 E2B multimodal projector, available [here](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/main/mmproj-F16.gguf).

---
 
### 4. Running the App
 
See **`app/README.md`** for full instructions on configuring and running the Docker container.
 
(Dependencies are handled by the Dockerfile — no separate `pip install` required.)

---

### License

This project is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0) in accordance with competition requirements.

The underlying Gemma base model and derivative weights remain subject to Google's Gemma Terms of Use.