# Auditoria da interface de linguagem natural e recuperação

**Data:** 25/11/2026
**Branch:** `feature-nl-gemini`
**Commit das correções:** `8032e78` — `fix: endurecer interpretacao natural e endpoints`

## Escopo e ambiente

- Windows, Python 3.14.6, pytest 9.2.0 e MuJoCo 4.0.1.
- Configuração padrão observada: `127.0.0.1:8000`, CORS `same-origin`. A porta 8000 estava livre antes e depois das verificações.
- `GEMINI_API_KEY` não estava definida. Nenhum peso `.pt` foi encontrado em `models/vision/`.
- Os comandos `node`, `chrome` e `msedge` não estavam disponíveis no `PATH`; não foi possível fazer uma inspeção visual do painel no navegador.
- `git fetch origin` atualizou `origin/main` de `3382cbd` para `aedbac9`. A referência `origin/feature-vision-yolo` não existe nesse remoto. Comparei com a referência local `miguel/feature-vision-yolo`: `HEAD...miguel/feature-vision-yolo` ficou em `8 0`, sem commits ou arquivos exclusivos do lado de `miguel`. Não houve merge.

## Execuções e resultados

| Verificação | Resultado |
| --- | --- |
| Cenário isolado `scenarios/test_basic_movement.py -k eight_commands_over_websocket` | `1 passed in 0.67s` |
| Suíte completa `tests/ scenarios/` | `72 passed in 12.32s` |
| Build do bundle | `scripts/bundle_frontend.py` gerou 80.767 bytes; `git diff --exit-code -- frontend/dist/index.html` não encontrou diferença |
| Avaliação offline | 56 frases: desenvolvimento `30/30`, teste `26/26`; 56/56 marcadas para revisão manual; precisão e recall de esclarecimento `100%` nesse conjunto |
| Stream durante inferência e log lentos simulados | Base `58.2 FPS`, intervalo p95 `30.9 ms`; durante a chamada `61.2 FPS`, intervalo p95 `29.9 ms`; 47 quadros em cada janela; resposta HTTP em `768.3 ms` |

A suíte foi executada com o Python 3.14.6 instalado diretamente e `--basetemp` dentro do workspace, com o plugin de cache desabilitado. O diretório temporário padrão do pytest falhou com `PermissionError` neste ambiente; o uso de `--basetemp` evitou esse problema sem alterar os testes. Comando executado:

```powershell
python -m pytest -p no:cacheprovider --basetemp=.pytest-release-check tests/ scenarios/ -q
```

O ensaio de stream usou um interpretador simulado que dormiu 600 ms e uma gravação de log simulada de 150 ms. Ele mede a chamada de interpretação e o caminho de log sem bloquear o loop do servidor; **não** representa uma chamada real ao Gemini. A implementação executa interpretação e gravação JSONL em `asyncio.to_thread`.

O avaliador gravou o resultado em [`nl_eval_audit_20261125.json`](../artifacts/nl_eval_audit_20261125.json). As métricas perfeitas são apenas uma verificação de que as regras atuais concordam com os rótulos escritos para este conjunto. Todas as frases seguem marcadas `revisar: true`; as classes, sinônimos e os exemplos foram escritos no projeto e ainda não receberam revisão humana independente. Não use `100%` como estimativa de desempenho fora do conjunto. O relatório legado de 40 frases não foi usado como evidência.

## Diagnóstico do cenário que travava

Havia pares de processos de teste anteriores ainda ativos após a interrupção do lançador Python do Windows. As linhas de comando foram conferidas e apenas processos que executavam os testes deste repositório foram encerrados. Depois de limpar esses processos, o cenário ainda reproduzia o travamento durante o encerramento do servidor.

O cenário completava os comandos e recebia os ACKs. O travamento ocorria em `server.stop()` enquanto `asyncio.Server.wait_closed()` aguardava o transporte WebSocket do cliente. `AsyncWSClient.close()` enviava um quadro de fechamento e fechava o TCP imediatamente, sem esperar a resposta de fechamento do servidor. O cliente agora aguarda a resposta por até um segundo, responde a ping durante esse período e força o fechamento do transporte se o prazo expirar. O cenário isolado passou em 0,67 s após a correção; a suíte final passou com 72 testes.

## Revisão por severidade

### Alta

- **Escuta e controle entre origens — corrigido por padrão.** O backend escutava em `0.0.0.0` e emitia CORS `*`. O padrão agora é `127.0.0.1` com verificação da origem HTTP e WebSocket `same-origin`. Acesso remoto exige configurar `SIM_HOST` e `SIM_CORS_ORIGIN` explicitamente. A API continua sem autenticação para clientes não navegador; não a exponha diretamente à internet. `SIM_CORS_ORIGIN=*` permite qualquer origem de navegador.

### Média

- **Prompt e comandos não permitidos — endurecidos.** A frase do usuário agora vai como dado JSON separado de `systemInstruction`; negações, alguns padrões de injeção e pedidos de movimento/coordenadas recebem esclarecimento no interpretador local. A resposta do Gemini continua limitada à lista de classes permitidas e validada antes de iniciar `vision_pick`. O formato de `systemInstruction` está previsto na [referência REST do Gemini GenerateContent](https://ai.google.dev/api/generate-content). Instruções de sistema não eliminam todos os ataques de prompt; a chave não estava disponível para testar o modelo real.
- **Tamanho e leitura de corpos — corrigidos.** A instrução aceita até 8 KiB de corpo e 300 caracteres; outros corpos HTTP têm limite de 1 MiB e leitura com prazo de 10 s. `Transfer-Encoding` não suportado e cabeçalhos de tamanho duplicados são recusados. Quadros/mensagens WebSocket são limitados a 16 KiB e quadros de controle inválidos são recusados. JSON e UTF-8 malformados retornam erro de entrada.
- **Abuso de `/api/instruction` — limitado.** Uma interpretação por vez; solicitações concorrentes ou com intervalo inferior a um segundo retornam `429`. Captura ativa continua retornando `409`.
- **Segredos em registros — mitigados.** A chave Gemini é enviada somente no cabeçalho `x-goog-api-key`. Texto de instrução e esclarecimento são limitados e mascaram a chave configurada, padrões Google API key, Bearer e campos comuns como `token`, `secret`, `password` e `senha`. O log JSONL permanece ignorado pelo Git.
- **Caminhos estáticos — corrigidos.** A verificação anterior usava comparação de prefixo de caminho, que podia aceitar uma pasta irmã cujo nome começasse pelo diretório permitido. O servidor agora exige que o caminho resolvido seja relativo à raiz do frontend ou dos assets.

### Baixa

- **Fechamento WebSocket de teste — corrigido.** O handshake de fechamento agora é completado com limite de tempo. O cliente também tolera o encerramento abrupto do transporte.
- **Recuperação da captura visual — revisão aprovada pelos testes existentes.** Falhas recuperáveis de detecção, revalidação ou preensão podem tentar novamente até `pick_max_attempts`; cada tentativa reobserva o alvo e a recuperação retorna à pose segura. Cancelamento e erros gerais da simulação não iniciam nova tentativa. O modo de uma tentativa também está coberto por teste.
- **Interface — revisão estática.** O texto retornado da API é renderizado por `textContent`; o `innerHTML` da criação do painel contém marcação fixa, sem interpolar a instrução. O bundle foi regenerado e permaneceu idêntico. Sem navegador instalado, não foi feita validação visual.

## Não verificado nesta execução

- Chamada real ao Gemini: pulada porque `GEMINI_API_KEY` não estava definida.
- Captura visual ponta a ponta com detector: pulada porque não há pesos `.pt` em `models/vision/`.
- Renderização visual do frontend: pulada porque não havia navegador disponível.
- Uso remoto da API sem autenticação: continua fora do escopo seguro; mantenha o backend em loopback ou use controles de rede apropriados antes de habilitar um bind remoto.

## Prontidão para relato acadêmico

Os testes de regressão e a medição local com atraso simulado podem ser relatados como verificação de implementação, deixando explícito que a inferência foi simulada. Os resultados de classificação não sustentam uma alegação de acurácia ou generalização: todas as 56 frases precisam de revisão humana e o Gemini real e a captura visual com pesos não foram executados. Até essa validação, apresente o sistema como protótipo e mantenha essas limitações junto aos resultados.
