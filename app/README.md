# Port-a-Prof: Aprendizado mais profundo, onde você estiver 💭

Um app offline de ensino com IA alimentado por um modelo Gemma 4 E2B com fine-tuning (ajuste fino). Suporta entrada de texto, imagem e voz. Roda inteiramente na sua máquina — sem uso de nuvem, sem chaves de API! 

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando (Para referência, uso o Docker versão 28.3.2, build 578ccf6)
- Conexão com a Internet **apenas na primeira execução** (para baixar a imagem do llama.cpp)

---

## Configuração

### 1. Adicionar os arquivos do modelo
**Modelos necessários:**
- `port-a-prof-Q4_K_M.gguf` — modelo de ensino treinado ([download](https://drive.google.com/file/d/1iSB_zN36mrmRCr8SUT2fMxHmAAWviwVe/view?usp=sharing))
- `mmproj-F16.gguf` — projetor multimodal Gemma 4 E2B ([download](https://drive.google.com/file/d/1tWI93dfLVuj3vkBu_ZjY7iQbDZYD-3lx/view?usp=sharing))

Esses arquivos devem ser baixados e colocados dentro da pasta `models/`:
```
models/
├── port-a-prof-Q4_K_M.gguf  ← modelo de ensino treinado
└── mmproj-F16.gguf          ← projetor multimodal Gemma 4 E2B
```

### 2. Iniciar o app
Abra um terminal, navegue até esta pasta e inicie o app:
```bash
cd caminho/para/o/app
docker compose up
```

Aguarde cerca de **1 minuto** para que o modelo seja carregado na memória. Você saberá que ele está pronto quando vir a impressão do template de chat, seguida de:
```
llama-server  | srv          main: model loaded
llama-server  | srv          main: server is listening on http://0.0.0.0:8080
llama-server  | srv          update_slots: all slots are idle
```

### 3. Abrir o app
Acesse **http://localhost:8001** no seu navegador.

> Nota: o terminal também exibirá `http://0.0.0.0:8080` — este é um endereço interno do servidor do modelo e não se destina a ser acessado diretamente. Sempre use **http://localhost:8001**.

---

## Iniciando e parando

| Situação | Comando |
|---|---|
| Início normal | `docker compose up` |
| Após alterar `port-a-prof.py`, `index.html` ou `Dockerfile` (o Docker faz cache das imagens, então não vai pegar as alterações sem isso) | `docker compose up --build` |
| Parar o app | Pressione `Ctrl+C` no terminal |

---

## Visualizando logs

O terminal de inicialização mistura a saída dos dois contêineres e a codificação de cores pode não renderizar corretamente durante a inicialização, dependendo do seu terminal. Para uma visão mais clara, abra uma segunda janela do terminal assim que o app estiver rodando e siga os logs do Port-a-Prof:

```bash
docker logs -f port-a-prof
```

Isto é **altamente recomendado** — os logs do app são codificados por cores e formatados para mostrar claramente cada fase do pipeline de ensino (geração da solução, diagnóstico do aluno, seleção do papel e resposta do professor), além de saídas intermediárias e blocos de raciocínio (thinking) para interpretação.

Para o servidor do modelo:
```bash
docker logs -f llama-server
```

---

## Usando o app

1. **Insira um problema** — digite-o, fale-o ou envie uma foto.
2. **Opcionalmente, descreva onde você travou** no segundo campo.
3. **Modo Raciocínio (Thinking Mode)** — quando ativado, o Port-a-Prof usa os tokens de raciocínio nativos do Gemma 4 E2B para pensar sobre o problema internamente antes de responder. Isso gera resultados mais precisos para problemas complexos, mas é mais lento. Desligue-o para problemas mais simples onde a velocidade importa mais.
4. Clique em **Começar a aprender** e espere pela primeira resposta.
5. Trabalhe o problema passo a passo — o Port-a-Prof irá guiá-lo sem apenas dar a resposta final.

---

## Uso offline

Após a primeira execução, o app funciona **totalmente offline**. A única coisa que requer internet é o download inicial da imagem Docker do llama.cpp, que será feito cache automaticamente.

---

## Agradecimentos

Muito obrigado por testar o Port-a-Prof 😊 Este projeto é construído sobre o [Gemma 4 E2B](https://ai.google.dev/gemma) — modelo aberto do Google que possibilita muitas oportunidades incríveis para melhorar a acessibilidade, particularmente na educação.

Obrigado também às equipes por trás das ferramentas que o fazem funcionar:

| Biblioteca | Uso |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Servidor de inferência do modelo local |
| [KaTeX](https://katex.org) | Renderização de notação matemática |
| [marked](https://marked.js.org) | Renderização de Markdown |
| [DM Sans](https://fonts.google.com/specimen/DM+Sans) | Fonte da UI |