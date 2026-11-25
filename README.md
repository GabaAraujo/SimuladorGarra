# SimuladorGarra 3D — Simulador de Garra Robótica em Produção (Python MuJoCo + WebGL WebSocket 60 FPS)

Simulador 3D de garra robótica paralela construído em arquitetura **Cliente-Servidor em Tempo Real**:

- **Backend (`backend/`)**: Python + **MuJoCo** executando a dinâmica de corpos rígidos, limites de juntas, cones elípticos de atrito e sensores táteis/força a **500 Hz**, transmitindo o estado sincronizado a **60 FPS** via WebSocket RFC 6455 (`TCP_NODELAY`).
- **Frontend (`frontend/`)**: Visualizador **3D WebGL** responsivo no navegador com iluminação Blinn-Phong, câmera orbital interativa, interpolação suave entre quadros físicos (**LERP** posicional + **SLERP** em quatérnios), detecção de timeout com fallback gracioso para o último estado conhecido e build **single-file** (`frontend/dist/index.html`) pronto para qualquer host estático.

> **Decisão Arquitetural**: Não compilamos o MuJoCo para WebAssembly (WASM). Manter a física em Python + C++ MuJoCo no servidor evita binários pesados no cliente, permite escalar para GPUs/clusters de aprendizado por reforço ou acoplar hardware real (**Sim-to-Real**) mantendo o frontend leve (~38 KB) e universal.

---

## 1. Estrutura do Projeto

```text
SimuladorGarra/
├── backend/                        # Simulação física Python + MuJoCo & Servidor 60 FPS
│   ├── models/
│   │   └── gripper.mjcf            # Modelo MJCF completo (limites, atrito, atuadores e sensores)
│   ├── mujoco_env.py               # Carregamento do .mjcf, integração física e leitura de sensores
│   ├── gripper_controller.py       # 8 funções de movimento (move_right/left/..., close, open)
│   ├── websocket_server.py         # Servidor HTTP + WebSocket 60 FPS em porta única
│   ├── config.py                   # Caminhos de modelo, especificações da garra e rede
│   └── main.py                     # Ponto de entrada do servidor
│
├── frontend/                       # Visualização 3D no navegador (zero dependências externas)
│   ├── index.html                  # Interface web com viewport 3D, HUD e painel de controle
│   ├── css/style.css               # Layout responsivo industrial
│   ├── js/renderer.js              # Cena 3D WebGL, malha articulada da garra e interpolação LERP/SLERP
│   ├── js/websocket_client.js      # Cliente WebSocket, métricas RTT/FPS, timeout e fallback
│   ├── js/controls.js              # Botões das 8 funções, atalhos WASD/QE, sliders speed/force
│   ├── js/controls.py              # Bindings declarativos equivalentes em Python (UI/automação)
│   └── dist/index.html             # Build single-file autônomo para deploy estático
│
├── scenarios/                      # Cenários automatizados de teste via WebSocket
│   ├── test_basic_movement.py      # Valida as 8 funções via WebSocket e schema de estado
│   ├── test_real_time_sync.py      # Aferição de stream 60 FPS e latência comando -> render (<100ms)
│   ├── test_object_interaction.py  # Pick & Place físico real no MuJoCo via WebSocket
│   └── test_stress.py              # Alta frequência (160+ cmds), 0 quedas e estabilidade de FPS
│
├── scripts/
│   └── bundle_frontend.py          # Gera o arquivo único frontend/dist/index.html
├── main.py                         # CLI principal (`python main.py --port 8000`)
├── requirements.txt                # Dependências Python do backend
├── package.json                    # Scripts de build e execução do frontend
├── PROTOCOL.md                     # Documentação completa do protocolo JSON WebSocket & REST
└── DEPLOYMENT.md                   # Guia de deploy em servidor remoto, Nginx WSS e frontend estático
```

---

## 2. Instalação e Posicionamento do Modelo MuJoCo (`.mjcf`)

### 2.1 Instalar Dependências Python

No **Windows (PowerShell)**, use o Python Launcher (`py`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Esses comandos usam o ambiente virtual diretamente, sem precisar ativá-lo. Se `python3` abrir a Microsoft Store ou informar que o Python não foi encontrado, use `py`: no Windows, `python3` pode apontar apenas para um atalho da Store. Para listar as versões instaladas, execute `py -0p`. O comando acima requer Python 3.11 instalado.

No **Linux/macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Caso utilize `uv`: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`)*

### 2.2 Posicionamento do Modelo `.mjcf`

O motor nativo atual constrói a cena UR5e + Robotiq em `backend/clinic_model.py`. A câmera `vision_top` também é criada nesse construtor. O arquivo legado `backend/models/gripper.mjcf`, validado como XML na inicialização e usado como referência pelo fallback, descreve:
- Juntas prismáticas limitadas (`joint_x`, `joint_y`, `joint_z`, `joint_finger_left`, `joint_finger_right`) com acoplamento simétrico `<equality>`.
- Pastilhas de contato com atrito elíptico elevado (`condim="4"`, `impratio="10"`).
- Sensores `<framepos>`, `<framequat>`, `<force>`, `<torque>` e sensores táteis `<touch>` em cada dedo.

O parâmetro `--model` seleciona o XML validado, mas atualmente não substitui a cena UR5e nativa. Para alterar essa cena, modifique o construtor em `backend/clinic_model.py`.

---

## 3. Como Executar o Backend e o Frontend

### 3.1 Iniciar o Servidor (Backend + Frontend Integrado na Porta 8000)

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe main.py --port 8000
```

Linux/macOS:

```bash
.venv/bin/python main.py --port 8000
```

Abra no navegador:
- **Interface 3D Interativa**: [http://localhost:8000](http://localhost:8000)
- **Versão Single-File Estática**: [http://localhost:8000/standalone](http://localhost:8000/standalone)
- **Healthcheck REST**: [http://localhost:8000/health](http://localhost:8000/health)

O backend escuta apenas `127.0.0.1` por padrão. Para acesso de outra máquina, configure `SIM_HOST` e um `SIM_CORS_ORIGIN` específico para a interface confiável. `SIM_CORS_ORIGIN=*` permite origens de navegador de qualquer domínio; use essa opção somente em uma rede controlada. O servidor não autentica clientes, portanto não o exponha diretamente à internet.

### 3.2 Gerar o Frontend Single-File para Hospedagem Estática Independente

```bash
python3 scripts/bundle_frontend.py
```

O arquivo gerado em `frontend/dist/index.html` contém todo o HTML, CSS e JavaScript WebGL embutidos e pode ser hospedado em qualquer servidor estático apontando para o backend via `?ws=ws://IP_DO_SERVIDOR:8000/ws`.

---

## 4. Controles na Interface 3D

- **Mouse Esquerdo + Arrastar**: Órbita 3D ao redor da bancada.
- **Mouse Direito (ou Shift) + Arrastar**: Panorâmica (Pan).
- **Scroll do Mouse**: Zoom.
- **8 Comandos de Movimento (Botões ou Teclado)**:
  - `W` / `S`: `move_forward` (+Y) / `move_backward` (-Y)
  - `A` / `D`: `move_left` (-X) / `move_right` (+X)
  - `E` / `Q`: `move_up` (+Z) / `move_down` (-Z)
  - `Espaço` ou `C`: `close` (fechar dedos com força controlada)
  - `O`: `open` (abrir dedos)
- **Sliders**: Ajuste em tempo real de `speed` (`0.05` – `1.50 m/s`) e `force` (`1` – `100 N`).
- **Pegar Instrumento / Pegar Fio Dental**: enviam um comando ao backend, que solicita a visão, calcula o plano e executa a entrega. Exigem visão e captura habilitadas.
- **Clínica 3D**: modelos em `assets/objetos/odontologia/` (cadeira, bandeja, kit, instrumento) via Three.js + `/api/scene-manifest`.

---

## 5. Executando os Cenários de Teste (`scenarios/`)

```bash
# Executar todos os 4 cenários WebSocket com relatório detalhado de métricas
.venv/bin/python main.py --scenario all

# Ou executar via pytest / unittest diretamente
.venv/bin/pytest scenarios/ -v
```

Consulte [`PROTOCOL.md`](./PROTOCOL.md) para detalhes das mensagens JSON e [`DEPLOYMENT.md`](./DEPLOYMENT.md) para instruções de produção.

## 6. Plano de Integração do YOLO

O documento [`YOLO_INTEGRATION.md`](./YOLO_INTEGRATION.md) descreve a arquitetura, as alterações por arquivo e os critérios de validação. A implementação inclui câmera MuJoCo `vision_top`, worker de visão, preview com caixas, projeção para o mundo, captura cancelável e ferramentas de dataset/treino/avaliação. A qualidade dos pesos e da preensão precisa ser validada para cada versão.

No Windows (PowerShell), instale o conjunto opcional e habilite o serviço de visão em espera:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe main.py --vision --port 8000
```

A câmera e o YOLO ficam em espera até um comando. Os botões “Pegar Instrumento”, “Pegar Fio Dental” e “Pegar ferramenta” solicitam uma nova detecção por classe ao backend; ele valida o alvo, calcula o plano e só então movimenta o braço. Há uma revalidação pontual antes da descida. Consultar status não produz imagens e o preview mantém apenas a última coleta. Sem pesos válidos ou sem alvo elegível, a operação falha sem iniciar movimento. O modo manual continua disponível. Para criar um detector odontológico:

```powershell
.\.venv\Scripts\python.exe scripts/generate_vision_dataset.py --output datasets/odontologia-new --count 300
.\.venv\Scripts\python.exe scripts/calibrate_vision.py --dataset datasets/odontologia-new --output models/vision/calibration-new.json
.\.venv\Scripts\python.exe scripts/train_yolo.py --data datasets/odontologia-new/data.yaml --output models/vision/odontologia-new.pt --epochs 50 --device cpu
.\.venv\Scripts\python.exe scripts/evaluate_vision.py --dataset datasets/odontologia-new --model models/vision/odontologia-new.pt --calibration models/vision/calibration-new.json
```

O gerador preserva datasets existentes: use `--output` com outro diretório se já houver dados; os outros scripts aceitam `--dataset` ou `--data` para apontar para essa versão. O treinamento padrão usa a arquitetura compacta local `models/vision/yolo_dental.yaml`, com `imgsz=320` e sem baixar pesos iniciais. Também aceita `--model yolo11n.yaml` ou um `.pt` local para transferência de aprendizado. Treinar do zero exige avaliar quantidade de dados e número de épocas. Use `--vision-imgsz` se os pesos tiverem sido treinados com outra resolução.

Os pesos e a calibração entregues têm um perfil experimental validado no simulador, descrito em [`VISION_VALIDATION.md`](./VISION_VALIDATION.md). Para habilitar a captura pelo painel, execute com `--vision --vision-enable-pick`. Novos pesos/calibrações saem dos scripts com `validated_for_pick: false`; registre a validação após medir erro e preensão, conforme [`models/vision/README.md`](./models/vision/README.md). Sem manifesto, calibração e flag habilitados, a captura fica bloqueada. `--vision-device`, `--vision-fps`, `--vision-confidence`, `--vision-model`, `--vision-calibration` e `--vision-max-age-ms` ajustam a execução.

Para ensaiar a preensão antes de aprovar novos artefatos, use `scripts/evaluate_vision_pick.py --class-name dental_tool` (ou `floss_holder`), acrescentando `--trials 3 --position-jitter 0.015` para variar posições. O script usa cópias temporárias no simulador, registra o resultado e preserva os manifestos originais. Um sucesso na pose inicial não substitui testes em posições variadas.

Se aparecer “Observação expirada”, confira `age_ms`, `processing_ms`, `inference_ms` e `max_age_ms` em `/api/vision/status`. O limite padrão é 300 ms e inclui captura, inferência e entrega ao controlador. A CPU usa duas threads, aplicadas antes do carregamento e reaplicadas antes da inferência porque a inicialização do Ultralytics pode redefinir esse limite. Após atualizar o código, pare o backend com Ctrl+C, reinicie e recarregue a página:

```powershell
.\.venv\Scripts\python.exe main.py --vision --vision-enable-pick --vision-threads 2 --port 8000
```

O primeiro comando carrega as dependências e aquece o modelo; aguarde sua conclusão antes de solicitar novamente. Se a carga da máquina ainda fizer os quadros excederem 300 ms, a bancada estática do simulador pode ser ensaiada com `--vision-max-age-ms 1000`. Esse ajuste permite observações mais antigas e exige nova avaliação se os objetos estiverem em movimento; ele não corrige ausência de ferramenta, oclusão ou calibração inválida.

Verificações:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_vision.py scenarios/ -q
.\.venv\Scripts\python.exe main.py --scenario vision
.\.venv\Scripts\python.exe scripts/bundle_frontend.py
```

## 7. Instruções em linguagem natural

O painel de visão aceita frases de até 300 caracteres, como “me passe o instrumento dental”. O interpretador converte a frase somente em uma tarefa com uma das classes autorizadas (`dental_tool` ou `floss_holder`); pedidos ambíguos ou não reconhecidos geram uma pergunta e não iniciam movimento.

O Gemini é opcional. No PowerShell, configure a chave apenas para a sessão atual e, se necessário, escolha o modelo e o limiar de confiança:

```powershell
$env:GEMINI_API_KEY = "SUA_CHAVE"
$env:GEMINI_MODEL = "gemini-2.5-flash"
$env:NL_CONFIDENCE_THRESHOLD = "0.65"
```

Não salve a chave no repositório nem em exemplos `.env`. Ela é enviada ao Gemini pelo header `x-goog-api-key`. Sem `GEMINI_API_KEY`, ou quando a chamada falha, a interpretação usa o fallback local por palavras-chave. As associações de sinônimos estão centralizadas em [`backend/nl_interpreter.py`](./backend/nl_interpreter.py) e marcadas `# REVISAR`, pois ainda dependem de confirmação das classes odontológicas.

Inicie o backend com a visão e a captura habilitadas (os pesos e a calibração precisam estar validados para preensão):

```powershell
.\.venv\Scripts\python.exe main.py --vision --vision-enable-pick --vision-pick-max-attempts 2 --port 8000
```

Linux/macOS, usando a configuração padrão de tentativas:

```bash
python main.py --vision --vision-enable-pick
```

O mesmo endpoint pode ser usado sem a interface:

```powershell
curl.exe -i -X POST http://localhost:8000/api/instruction `
  -H "Content-Type: application/json" `
  -d '{"text":"Me passe o instrumento dental"}'
```

Um pedido que casa com mais de uma ferramenta retorna `accepted:false` e uma pergunta de esclarecimento; não move a garra:

```powershell
curl.exe -i -X POST http://localhost:8000/api/instruction `
  -H "Content-Type: application/json" `
  -d '{"text":"Pegue o instrumento dental e o fio dental"}'
```

Respostas aceitas iniciam o mesmo `vision_pick` dos botões do painel. O servidor retorna a tarefa, confiança, fonte (`gemini` ou `keyword`), status atual e erro opcional. `409` indica outra captura em andamento, `429` limita solicitações concorrentes ou muito próximas, `403` indica origem de navegador não permitida e `503` indica visão/captura desabilitada. O endpoint aceita no máximo 300 caracteres e 8 KiB de corpo; demais corpos HTTP são limitados a 1 MiB e mensagens WebSocket a 16 KiB. O WebSocket também aceita `{"type":"instruction","text":"..."}`; consulte [`PROTOCOL.md`](./PROTOCOL.md) para os schemas. Tentativas e resultados finais são acrescentados a `artifacts/nl_log.jsonl` com segredos comuns mascarados.

Falhas de detecção, revalidação ou preensão podem repetir até `--vision-pick-max-attempts`; entre tentativas, o robô abre a garra, retorna à pose segura e faz uma nova detecção. O status informa `attempt`, `max_attempts`, `last_failure` e `recovered`. Cancelamento e erros de simulação não iniciam outra tentativa.

Execute os testes offline e a avaliação de palavras-chave:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ scenarios/ -q
.\.venv\Scripts\python.exe scripts/evaluate_nl.py --mode keyword
```

O avaliador lê as 56 frases de `datasets/nl_commands.json`, separa 30 de desenvolvimento e 26 de teste, e grava métricas por split em `artifacts/nl_eval_<timestamp>.json`. Todas as linhas estão marcadas para revisão humana; os resultados atuais são exploratórios e não demonstram qualidade independente. Para avaliar o Gemini, defina `GEMINI_API_KEY` e use `--mode gemini`. Para incluir capturas simuladas de ponta a ponta, passe `--e2e N`, por exemplo `--mode keyword --e2e 2`; esse ensaio exige detector, pesos e calibração validados.
