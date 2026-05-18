# Port-a-Prof: Deeper Learning, Wherever You Are 💭

An offline AI teaching app powered by a fine-tuned Gemma 4 E2B model. Supports text, image, and voice input. Runs entirely on your machine — no cloud use, no API keys! 

---

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running (For reference I use Docker version 28.3.2, build 578ccf6)
- Internet connection for the **first run only** (to pull the llama.cpp image)

---

## Setup

### 1. Add the model files
**Models required:**
- `port-a-prof-Q4_K_M.gguf` — fine-tuned teaching model ([download](https://huggingface.co/bianca-lilyyy128/port-a-prof-Q4_K_M))
- `mmproj-F16.gguf` — Gemma 4 E2B multimodal projector ([download](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/main/mmproj-F16.gguf))


These must be downloaded and placed inside the `models/` folder:
```
models/
├── port-a-prof-Q4_K_M.gguf  ← fine-tuned teaching model
└── mmproj-F16.gguf          ← Gemma 4 E2B multimodal projector
```

### 2. Start the app
Open a terminal, navigate to this folder, and start the app:
```
cd path/to/app
docker compose up
```

Wait about **1 minute** for the model to load into memory. You'll know it's ready when you see the chat template print out followed by:
```
llama-server  | srv          main: model loaded
llama-server  | srv          main: server is listening on http://0.0.0.0:8080
llama-server  | srv          update_slots: all slots are idle
```

### 3. Open the app
Visit **http://localhost:8001** in your browser.

> Note: the terminal will also print `http://0.0.0.0:8080` — this is an internal address for the model server and is not meant to be visited directly. Always use **http://localhost:8001**.

---

## Starting and stopping

| Situation | Command |
|---|---|
| Normal start | `docker compose up` |
| After changing `port-a-prof.py`, `index.html`, or `Dockerfile` (Docker caches images so won't pick up changes without this) | `docker compose up --build` |
| Stop the app | Press `Ctrl+C` in the terminal |

---

## Viewing logs

The startup terminal mixes both containers' output, and colour coding may not render correctly during startup depending on your terminal. For a clearer view, open a second terminal window once the app is running and follow the Port-a-Prof logs:

```bash
docker logs -f port-a-prof
```

This is **highly recommended** — the app logs are colour-coded and formatted to clearly show each phase of the teaching pipeline (solution generation, student diagnosis, role selection, and teacher response), along with intermediate outputs and thinking blocks for interpretability.

For the model server:
```bash
docker logs -f llama-server
```

---

## Using the app

1. **Enter a problem** — type it, speak it, or upload a photo
2. **Optionally describe where you're stuck** in the second field
3. **Thinking Mode** — when on, Port-a-Prof uses Gemma 4 E2B's native thinking tokens to reason through the problem internally before responding. This gives more accurate results for complex problems but is slower. Turn it off for simpler problems where speed matters more.
4. Press **Start learning** and wait for the first response
5. Work through the problem step by step — Port-a-Prof will guide you without just giving the answer

---

## Offline use

After the first run, the app works **fully offline**. The only thing that requires internet is pulling the llama.cpp Docker image on first start, which is cached automatically after that.

---

## Acknowledgements

Thanks so much for trying Port-a-Prof 😊 This project is built on [Gemma 4 E2B](https://ai.google.dev/gemma) — Google's open-weight edge model that opens up so many exciting possibilities for improving accessibility, particularly in education.

Thanks also to the teams behind the tools that make it run:

| Library | Use |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Local model inference server |
| [KaTeX](https://katex.org) | Maths notation rendering |
| [marked](https://marked.js.org) | Markdown rendering |
| [DM Sans](https://fonts.google.com/specimen/DM+Sans) | UI font |