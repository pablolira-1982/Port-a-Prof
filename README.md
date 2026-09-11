# Port-a-Prof: Aprendizado mais profundo, onde você estiver 💭
### Expandindo o acesso à educação com IA, sem comprometer a qualidade do aprendizado.
*Um projeto para o com dedicação para os necessitados.*

---

Port-a-Prof é um assistente de aprendizado offline com IA projetado para promover um engajamento profundo com o trabalho escolar.

Alimentado por uma variante do Gemma 4 E2B IT ajustada com QLoRA, ele funciona inteiramente no dispositivo, mantendo os dados do aluno privados enquanto suporta entradas de texto, áudio e imagem.

A visão por trás do Port-a-Prof é fornecer assistência de aprendizado de alta qualidade a alunos, independentemente de sua localização, conectividade ou circunstância financeira — inclusive por meio de iniciativas de acesso a dispositivos, nas quais dispositivos doados são distribuídos para comunidades com o Port-a-Prof pré-instalado, tornando o suporte de aprendizado personalizado acessível a alunos que historicamente ficaram de fora.

Este repositório inclui:
- **Geração de dataset** — scripts para gerar perguntas iniciais e um conjunto de dados de diálogos aluno-professor de múltiplos turnos para ajuste fino (fine-tuning), via Ollama.
- **Fine-tuning** — um notebook de treinamento QLoRA construído no Hugging Face Transformers, PEFT, TRL e bitsandbytes.
- **App** — uma interface de prova de conceito construída com FastAPI e um frontend HTML de página única, em contêiner com Docker.

---

## Estrutura do Projeto

```
PORT-A-PROF/
├── app/                        # App de prova de conceito
│   ├── models/                 # Arquivos de modelo GGUF (veja abaixo)
│   ├── static/                 # Arquivos estáticos
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── index.html
│   ├── logo.png
│   ├── port-a-prof.py          # Backend FastAPI
│   └── README.md               # Instruções de configuração do App
│
├── dataset-generation/         # Pipeline de dados de treinamento sintético
│   ├── dataset/                # Saída das trajetórias geradas
│   ├── questions/              # Perguntas iniciais geradas
│   ├── generate_questions.py   # Passo 1: gerar perguntas iniciais via Ollama
│   └── generate_dataset.py     # Passo 2: gerar diálogos aluno-professor via Ollama
│
├── fine-tuning/                # Ajuste fino (fine-tuning) QLoRA
│   ├── dataset/
│   └── port-a-prof_QLoRA.ipynb
│
├── README.md
└── requirements.txt
```

---

## Início Rápido

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

> **PyTorch** deve ser instalado separadamente — veja [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

---

### 2. Geração de Dataset (requer Ollama)

O [Ollama](https://ollama.com) deve estar rodando localmente antes de executar qualquer script de geração.

Os seguintes modelos são usados por padrão — eles podem ser alterados para se adequarem às suas preferências:

- **[gemma3:12b-it-qat](https://ollama.com/library/gemma3:12b-it-qat)** — geração de perguntas e soluções iniciais (`generate_questions.py`)
- **[gemma4:e4b](https://ollama.com/library/gemma4:e4b)** — geração de diálogo aluno-professor (`generate_dataset.py`)
```bash
# Passo 1 — gerar perguntas iniciais
python dataset-generation/generate_questions.py

# Passo 2 — gerar diálogos aluno-professor
python dataset-generation/generate_dataset.py
```

---

### 3. Fine-tuning (Ajuste Fino)

O modelo com ajuste fino é projetado para uso como parte do pipeline do sistema Port-a-Prof, não isoladamente — ele espera uma entrada estruturada para funcionar corretamente.

### Formato de Entrada

O modelo é treinado em prompts estruturados com o seguinte esquema:

```
<bos><|turn>user
## PROBLEM
{texto do problema}

## STUDENT_ATTEMPT
{tentativa ou dúvida do aluno}

## STATUS
{avaliação de onde o aluno está}

## TEACHER_ROLE
{partial_worked_step | redirect | confirm_and_advance | session_close | partial_worked_step}

## SUPPORT                          ← presente apenas quando TEACHER_ROLE for partial_worked_step
{low | medium | high}<turn|>
<|turn>model
```

- **PROBLEM** — a pergunta original ou tarefa apresentada ao aluno
- **STUDENT_ATTEMPT** — o trabalho do aluno, resposta ou onde ele travou
- **STATUS** — um resumo diagnóstico da compreensão atual do aluno
- **TEACHER_ROLE** — a estratégia de instrução que o modelo deve adotar
- **SUPPORT** — o nível de suporte a fornecer (`low`, `medium` ou `high`) em uma etapa parcialmente resolvida

### Obtendo o Modelo

Você pode optar por:

1. **Fazer o fine-tuning você mesmo** — abra `fine-tuning/port-a-prof_QLoRA.ipynb` e siga as células. O notebook produz um modelo mesclado e de precisão total, salvo em `./port_a_prof_finetuned` no formato safetensors do Hugging Face, que pode então ser quantizado e convertido para GGUF usando o [llama.cpp](https://github.com/ggerganov/llama.cpp).
2. **Baixar o GGUF pré-convertido** — pegue o modelo quantizado Q4_K_M diretamente do [Hugging Face](https://huggingface.co/bianca-lilyyy128/port-a-prof-Q4_K_M).
Para executar o app, você também precisará do projetor multimodal Gemma 4 E2B, disponível [aqui](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/main/mmproj-F16.gguf).

---

### 4. Executando o App

Veja o **`app/README.md`** para instruções completas sobre como configurar e executar o contêiner Docker.

(As dependências são tratadas pelo Dockerfile — nenhum `pip install` separado é necessário.)

---

## ❤️ Apoie este projeto

Se este projeto foi útil para você e quiser apoiar o desenvolvimento, pode fazer uma doação voluntária via PIX.

### 🇧🇷 PIX

<p align="center">
  <img src="./qrcode-chave-pix.png" width="250" alt="QR Code PIX">
</p>

**Chave PIX:**

```text
b3652001-9daa-4131-ad05-6ea1f59e1721
```

---

### Licença

Este projeto é lançado sob a licença Creative Commons Attribution 4.0 International License (CC BY 4.0), de acordo com os requisitos da competição.

O modelo base subjacente do Gemma e seus pesos derivados permanecem sujeitos aos Termos de Uso do Gemma do Google.
