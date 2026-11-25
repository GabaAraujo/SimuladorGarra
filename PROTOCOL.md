# Especificação do Protocolo WebSocket e REST (`PROTOCOL.md`)

Este documento define o contrato de comunicação em tempo real entre o **Backend Python + MuJoCo** (`backend/websocket_server.py`) e o **Frontend 3D no Navegador** (`frontend/js/websocket_client.js`).

## Extensão de visão YOLO: fluxo por comando

O comando do braço inicia a percepção. Habilitar `--vision`, consultar status ou abrir a interface não captura imagens. O worker espera uma solicitação explícita e executa uma coleta finita: até `max(3, stable_frames * 2)` quadros, encerrando antes se o alvo ficar estável. `--vision-fps` limita a cadência apenas durante essa coleta.

Fluxo: comando → detecção da ferramenta → validação do retorno → cálculo do plano → execução. O braço solicita uma segunda coleta pontual para revalidar a ferramenta após aproximar-se. Não existe captura contínua enquanto aguarda ou movimenta o objeto.

```json
{"command":"vision_pick","class_name":"dental_tool"}
```

`class_name` aceita `dental_tool` ou `floss_holder`. O cliente não precisa enviar quadro, detecção ou coordenadas. O backend obtém uma observação nova, resolve o alvo internamente e publica `plan` antes de comandar os atuadores. Ausência, ambiguidade, timeout ou geometria inválida encerram a operação sem iniciar movimento.

Para diagnóstico sem movimento, há um comando explícito separado:

```json
{"command":"vision_detect","class_name":"floss_holder"}
```

`vision_cancel` cancela a operação e/ou solicitação de visão; `reset` também invalida a revisão. Uma inferência que já começou pode terminar, mas seu resultado cancelado não é publicado nem usado por outra operação.

### Instruções em linguagem natural

O endpoint aceita uma frase de até 300 caracteres e só inicia uma captura quando a tarefa estruturada identifica uma classe permitida sem pedir esclarecimento. O Gemini é opcional; sem a chave ou diante de indisponibilidade, o backend usa o interpretador local por palavras-chave.

```http
POST /api/instruction
Content-Type: application/json

{"text":"Me passe o instrumento dental"}
```

A resposta mantém a interpretação, a aceitação, o status atual de `vision_pick` e um erro opcional:

```json
{"task":{"intent":"fetch","class_name":"dental_tool","confidence":0.82,"needs_clarification":false,"clarification":null,"source":"keyword"},"accepted":true,"status":{"type":"vision_pick_status","stage":"detecting"},"error":null}
```

Um pedido ambíguo retorna HTTP 200 com `accepted:false` e a pergunta em `task.clarification`; a entrada inválida retorna 400, corpo acima do limite retorna 413, origem de navegador não permitida retorna 403, captura ocupada retorna 409, solicitação concorrente ou enviada em menos de um segundo retorna 429 e visão/captura desabilitada retorna 503. Nenhuma tarefa com esclarecimento movimenta o braço. O endpoint aceita até 300 caracteres e 8 KiB de corpo; outros corpos HTTP têm limite de 1 MiB e mensagens WebSocket de 16 KiB. O servidor registra tentativa, interpretação, resultado final e duração em `artifacts/nl_log.jsonl` com segredos mascarados.

O servidor escuta em `127.0.0.1` e aceita apenas a mesma origem de navegador por padrão. Configure `SIM_HOST` e um `SIM_CORS_ORIGIN` específico para acesso remoto; `SIM_CORS_ORIGIN=*` permite todas as origens de navegador. A API não tem autenticação e não deve ser exposta diretamente à internet.

O WebSocket aceita o formato equivalente `{"type":"instruction","text":"..."}` e responde com `type:"instruction_result"`, os mesmos campos da resposta HTTP e `http_status` com o status correspondente.

| Interface | Contrato |
| --- | --- |
| `GET /api/vision/status` e `/api/vision/detections` | Somente leitura; nunca acionam a câmera. Incluem `operation`, `request_active`, `request_id`, `requested_class`, `frames_captured`, `allowed_classes` e `can_request_pick` |
| `GET /api/vision/frame?frame_id=125` | JPEG de uma coleta já solicitada; não produz nova imagem; 404 se saiu do cache ou houve reset |
| WS `vision` | Atualizações da solicitação e resultados; inclui estado `idle`/`detecting` e estados de resultado `ready`/`stale`/`error`, além de `disabled` |
| WS `vision_pick_status` | `operation_id`, classe solicitada, etapa, erro, alvo estimado, plano, plano restante e campos `attempt`, `max_attempts`, `last_failure` e `recovered` |

O `ack` confirma recebimento. A conclusão é informada pela operação: `detecting`, `planning`, `approaching`, `revalidating`, `aligning`, `descending`, `closing`, `lifting`, `delivering`, `lowering`, `retreating` e `recovering`, com terminais `completed`, `failed` ou `cancelled`. Falhas de detecção, revalidação e preensão podem iniciar nova tentativa; a recuperação abre a garra, retorna à pose segura e solicita uma observação nova. Cancelamento e erros de simulação não são repetidos.

As detecções mantêm caixa em pixels, classe, confiança, estabilidade e candidato em metros. O cache contém no máximo quatro quadros. A idade usa relógio monotônico e continua sendo validada antes de usar um resultado. Uma imagem antiga pode permanecer no preview como última observação; isso não mantém a visão capturando.

O alinhamento/descida usa o ponto revalidado por até seis segundos sob possível oclusão, pressupondo bancada estática. O contrato físico `state` e os comandos manuais existentes permanecem disponíveis.

---

## 1. Transporte e Conexão

- **Endpoint WebSocket Padrão**: `ws://<HOST>:<PORT>/ws` (em TLS/produção: `wss://<DOMAIN>/ws`)
- **Versão do Protocolo**: RFC 6455 (WebSocket Version 13), quadros UTF-8 JSON (`opcode = 0x1`)
- **Otimização de Latência**: `TCP_NODELAY` habilitado no servidor e escrita protegida por lock assíncrono por cliente
- **Taxa de Transmissão Contínua**: **60 FPS** (`16.67 ms` por quadro físico, 8 sub-passos MuJoCo de `0.002 s` por quadro)
- **Keepalive / Detecção de Desconexão**:
  - Servidor envia quadros `PING (0x9)` após `30s` de inatividade de comandos.
  - Cliente monitora watchdog de `2500ms`: se o fluxo 60 FPS cessar, ativa **Modo Fallback Gracioso** mantendo o último estado válido renderizado na tela e agenda reconexão com *exponential backoff*.

---

## 2. Mensagens Cliente → Servidor (Comandos)

### 2.1 Comandos das 8 Funções Principais da Garra

Formato canônico:

```json
{
  "command": "move_right",
  "speed": 0.5,
  "force": 50
}
```

Campos suportados:

| Campo | Tipo | Obrigatório | Faixa / Padrão | Descrição |
| :--- | :--- | :---: | :--- | :--- |
| `command` | `string` | Sim | — | Uma das 8 ações (`move_right`, `move_left`, `move_forward`, `move_backward`, `move_down`, `move_up`, `close`, `open`), além de `reset`, `move_to` ou `get_state`. |
| `speed` | `number` | Não | `0.05` – `1.50` (`0.50` m/s) | Velocidade linear alvo do movimento cartesiano ou dos dedos. |
| `force` | `number` | Não | `1.0` – `100.0` (`50.0` N) | Força máxima de atuação/preensão configurada nos servos dos dedos MuJoCo. |
| `duration` | `number` | Não | `0.01` – `5.0` (`0.16` s) | Duração do pulso incremental de deslocamento ($\Delta d = \text{speed} \times \text{duration}$). |
| `req_id` | `string` | Não | — | Identificador único ecoado no `ack` para correlação de latência RTT. |
| `client_ts` | `number` | Não | Epoch ms | Timestamp do cliente ecoado para telemetria comando → renderização. |

---

## 3. Mensagens Servidor → Cliente (Estado em Tempo Real)

Tanto os quadros contínuos de 60 FPS (`"type": "stream"`) quanto as confirmações imediatas de comando (`"type": "ack"`) e o pacote inicial de boas-vindas (`"type": "hello"`) transportam o objeto canônico `"state"`:

### 3.1 Exemplo Mínimo / Canônico de Resposta

```json
{
  "type": "ack",
  "command": "move_right",
  "req_id": "move_right-1763228400",
  "fps": 60.0,
  "state": {
    "pos": [0.08, 0.0, 0.238],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "force": 50.0,
    "commanded_force": 50.0,
    "contact_force": 0.0,
    "speed": 0.5,
    "aperture": 0.08,
    "is_closed": false,
    "is_grasping": false,
    "grasped_object": null,
    "velocity": [0.12, 0.0, 0.0],
    "sensors": {
      "left_touch": 0.0,
      "right_touch": 0.0,
      "wrist_force": [0.0, 0.0, 0.0],
      "wrist_torque": [0.0, 0.0, 0.0]
    },
    "objects": [
      {
        "name": "red_cube",
        "label": "Cubo Vermelho",
        "type": "box",
        "size": [0.024, 0.024, 0.024],
        "color": "#e63d42",
        "pos": [0.16, 0.1, 0.024],
        "quat": [1.0, 0.0, 0.0, 0.0]
      }
    ],
    "drop_zone": {
      "pos": [-0.2, 0.16, 0.0015],
      "size": [0.065, 0.065, 0.0015]
    },
    "sim_time": 4.256,
    "seq": 255,
    "server_ts": 1763228400125.4,
    "engine": "mujoco-4.0.1"
  }
}
```

### 3.2 Definição do Schema `state`

- `pos`: `[x, y, z]` em metros no referencial mundial MuJoCo (`Z-up`), extraído do sensor `<framepos name="tcp_pos"/>`.
- `quat`: `[w, x, y, z]` quatérnio unitário de orientação 3D do TCP, extraído do sensor `<framequat name="tcp_quat"/>`.
- `force`: força efetiva ativa em Newtons (`N`).
- `contact_force`: soma real das leituras dos sensores táteis `<touch name="left_finger_touch"/>` e `<touch name="right_finger_touch"/>` em Newtons.
- `aperture`: distância livre atual entre as faces internas dos dedos paralelos em metros (`0.000` a `0.080`).
- `is_grasping` / `grasped_object`: indica se um objeto manipulável está fisicamente sustentado pelos dedos.
- `objects`: lista de corpos rígidos dinâmicos na cena com pose `[x, y, z]` e quatérnio `[w, x, y, z]` para sincronização de malhas 3D.

### 3.3 Quadro de Erro (`type == "error"`)

Caso um comando inválido seja enviado, a conexão **não é encerrada**; o servidor responde com diagnóstico e o estado íntegro atual:

```json
{
  "type": "error",
  "error": "Comando desconhecido 'fly_away'. Suportados: ['close', 'move_backward', ...]",
  "state": { "pos": [0.0, 0.0, 0.238], "quat": [1.0, 0.0, 0.0, 0.0], "force": 50.0 }
}
```

---

## 4. Endpoints HTTP REST Auxiliares

Na mesma porta TCP (`8000` por padrão), o servidor expõe endpoints REST com verificação de origem same-origin por padrão:

- `GET /health` — Status operacional, motor físico ativo, FPS medido, clientes conectados e uptime.
- `GET /api/state` — Snapshot imediato `{"state": ...}` via HTTP.
- `POST /api/command` — Executa comando JSON (mesmo payload do WebSocket) e retorna `{"state": ...}`.
- `POST /api/instruction` — Interpreta texto em uma tarefa permitida e inicia `vision_pick` somente quando não houver ambiguidade.
- `POST /api/reset` — Reinicia a simulação MuJoCo para o estado inicial.
