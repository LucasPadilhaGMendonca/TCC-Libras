# SignForest — Reconhecimento de LIBRAS com MediaPipe e Random Forest

Projeto desenvolvido para o Trabalho de Conclusão de Curso (TCC) focado no reconhecimento de sinais estáticos do alfabeto datilológico da Língua Brasileira de Sinais (**LIBRAS**). O pipeline processa vídeos de fontes públicas, extrai pontos-chave anatômicos das mãos via MediaPipe, treina um classificador Random Forest e exporta o modelo treinado para inferência em tempo real via extensão de navegador (Chrome/Edge) usando ONNX Runtime Web.

A extensão soletra automaticamente: ao segurar um sinal estático por tempo suficiente, a letra é confirmada e digitada sozinha no campo de texto focado na página — sem precisar clicar em nada por letra.

---

## 🏛️ Arquitetura do Sistema

O fluxo de processamento e dados está estruturado em 6 etapas:

```text
FONTES DE VÍDEO (LIBRAS)
│
├── 1. V-LIBRASIL (UFPE)
├── 2. Dicionário Virtual LIBRAS (YouTube)
├── 3. Gilbervan Soares (YouTube)
└── 4. Incluir Tecnologia (YouTube)
          ↓
     [ dataset/videos/<CLASSE>/*.mp4 ]
          ↓  (training/extract_frames.py @ 8 FPS)
     [ dataset/frames/<CLASSE>/*.jpg ]
          ↓  (training/create_dataset.py)
   MediaPipe Hands (Detecção de 1 mão)
          ↓
   21 Landmarks anatômicos (X, Y)
          ↓
   Normalização das Coordenadas:
   (x - min(X)), (y - min(Y))
          ↓
     42 Características (Features)
          ↓
     [ dataset/data.pickle ]  (features + labels + groups)
          ↓  (training/train_classifier.py)
     Random Forest Classifier (Scikit-Learn)
          ↓
     [ training/model.p ]
          ↓  (training/export_model.py)
     [ extension/assets/onnx/model.onnx ]
          ↓
   EXTENSÃO CHROME/EDGE (Sidepanel / WebCam / ONNX Runtime Web)
```

## 📁 Estrutura de Diretórios

```text
TCC-Libras/
│
├── dataset/
│   ├── videos/                  # Vídeos brutos organizados por subpastas (A/, B/, C/...)
│   ├── frames/                  # Frames extraídos automaticamente (extract_frames.py)
│   └── data.pickle              # features (42), labels e groups (vídeo de origem, p/ split sem vazamento)
│
├── training/
│   ├── extract_frames.py        # Extrai frames dos vídeos em taxa controlada (8 FPS)
│   ├── create_dataset.py        # Processa frames com MediaPipe e gera o data.pickle
│   ├── train_classifier.py      # Treina o modelo Random Forest e gera model.p
│   ├── hyperparameter_search.py # Validação cruzada e busca em grade de parâmetros
│   ├── export_model.py          # Converte o modelo Scikit-Learn (.p) para formato ONNX
│   ├── inference_reference.py   # Script de teste e referência para inferência local
│   ├── requirements.txt         # Dependências do ambiente Python
│   └── tests/                   # Testes pytest (lógica de features)
│
├── extension/                   # Extensão Chrome/Edge (Manifest V3)
│   ├── assets/
│   │   ├── mediapipe/           # @mediapipe/tasks-vision vendorizado (vision_bundle.mjs, wasm/, hand_landmarker.task)
│   │   └── onnx/
│   │       ├── labels.json      # Mapeamento das classes do modelo
│   │       ├── ort.min.js / ort-wasm.wasm  # ONNX Runtime Web vendorizado
│   │       └── model.onnx       # Modelo exportado (gerado localmente, não versionado)
│   ├── lib/
│   │   ├── features.mjs         # Normalização de landmarks → 42 features (espelha create_dataset.py)
│   │   ├── spelling.mjs         # Estabilização de gesto → confirmação de letra (soletração automática)
│   │   └── *.test.mjs           # Testes (node --test)
│   ├── background.js            # Service worker da extensão
│   ├── content.js               # Insere/apaga caracteres no campo focado da página
│   ├── manifest.json            # Manifesto de configuração da extensão
│   ├── sidepanel.css            # Estilos da interface lateral
│   ├── sidepanel.html           # Interface do painel lateral (captura da webcam)
│   └── sidepanel.js             # Captura, MediaPipe, ONNX Runtime e soletração automática
│
├── .gitignore
└── README.md
```

## 📚 Base de Dados e Fontes Utilizadas

O treinamento utiliza exclusivamente fontes de LIBRAS:

- **V-LIBRASIL (UFPE)**: Base acadêmica com mais de 4.000 vídeos de múltiplos intérpretes.
- **Dicionário Virtual LIBRAS (YouTube)**: Acervo com mais de 400 vídeos voltados a serviços e utilidade pública.
- **Gilbervan Soares (YouTube)**: Canal voltado ao ensino e divulgação de LIBRAS com mais de 200 vídeos.
- **Incluir Tecnologia (YouTube)**: Canal com mais de 1.000 vídeos dedicados ao aprendizado e sinalização em LIBRAS.

**Nota de escopo:** Fontes em Língua de Sinais Americana (ASL) foram completamente desconsideradas para garantir fidelidade às variações e particularidades do alfabeto em LIBRAS.

**Letras fora do escopo:** H, J, K, X, Y, Z não estão no dataset porque seus sinais em LIBRAS envolvem movimento, e o pipeline (MediaPipe Hands sobre frames isolados) só reconhece poses estáticas.

**Estado atual do dataset:** cada classe tem frames vindos de **um único vídeo/intérprete**. `train_classifier.py` detecta essa condição automaticamente e usa um split aleatório por frame, avisando no console que a acurácia reportada não mede generalização real (frames do mesmo vídeo são muito correlacionados). Ao adicionar 2+ vídeos por classe, o script passa a usar `StratifiedGroupKFold` (nenhum vídeo aparece simultaneamente em treino e teste), o que produz uma métrica confiável.

## 🚀 Guia de Execução (pipeline de treino)

### 1. Instalação do Ambiente

```powershell
pip install -r training/requirements.txt
```

### 2. Organização dos Vídeos

Coloque os vídeos organizados por classe dentro de `dataset/videos/`:

```text
dataset/videos/
├── A/
│   ├── video1.mp4
│   └── video2.mp4
├── B/
│   └── video1.mp4
└── ...
```

### 3. Extração Automática de Frames

```powershell
python training/extract_frames.py
```

### 4. Extração de Features (MediaPipe)

Gera o arquivo `dataset/data.pickle` contendo as 42 features normalizadas por frame:

```powershell
python training/create_dataset.py
```

### 5. Treinamento do Modelo

Treina o classificador RandomForestClassifier e gera o relatório de métricas:

```powershell
python training/train_classifier.py
```

### 6. Exportação para a Extensão (ONNX)

Gera o arquivo `extension/assets/onnx/model.onnx` compatível com a extensão:

```powershell
python training/export_model.py
```

## 🧩 Carregando a Extensão (Chrome e Edge)

Pré-requisito: rode o passo 6 acima (ou baixe um `model.onnx` já treinado) antes de carregar — sem ele, `sidepanel.js` falha ao iniciar a câmera.

**Chrome**: acesse `chrome://extensions/` → ative o **Modo do desenvolvedor** → **Carregar sem compactação** → selecione a pasta `extension/`.

**Edge**: acesse `edge://extensions/` → ative o **Modo de desenvolvedor** → **Carregar sem pacote (Load unpacked)** → selecione a pasta `extension/`. A extensão usa `chrome.sidePanel` (API Manifest V3 espelhada pelo Edge Chromium); em ambientes corporativos gerenciados, políticas de grupo podem bloquear extensões não publicadas na loja ou o acesso à webcam — isso está fora do controle do código e deve ser verificado com o administrador da máquina de teste.

Depois de carregada, clique no ícone da extensão para abrir o painel lateral, clique em **Iniciar câmera** e comece a soletrar.

### Como funciona a soletração automática

1. O sinal precisa ficar estável (mesma letra, confiança suficiente) por um número mínimo de frames consecutivos antes de ser confirmado — evita digitar ruído/transições entre sinais.
2. Depois de confirmada, a mesma letra só é aceita de novo depois que a mão sai do quadro/sinal por alguns frames — evita repetição infinita enquanto a mão fica parada.
3. Cada letra confirmada é **acrescentada** ao campo de texto focado na página (não substitui o conteúdo existente).
4. Os botões **Espaço**, **Apagar** e **Limpar texto digitado** cobrem os comandos que ainda não têm um sinal dedicado no dataset; **Pausar soletração** para a digitação automática sem desligar a câmera.

## 🔒 Privacidade e dados

Toda a captura de webcam e a inferência (MediaPipe + ONNX Runtime Web) rodam **inteiramente no navegador do usuário** — nenhum quadro de vídeo, landmark de mão ou texto reconhecido é enviado a servidores externos. Isso é relevante porque landmarks de mão derivados de vídeo se enquadram como dado biométrico/sensível sob a LGPD (Lei 13.709/18); o processamento 100% local remove a maior parte da superfície de exposição desse dado.

O manifesto declara `host_permissions: ["<all_urls>"]` e um `content_script` em `<all_urls>` porque o objetivo do produto é preencher campos de texto em qualquer site que o usuário esteja usando no momento — não um site específico. Essa é uma permissão ampla; se o escopo do TCC for restringido a um conjunto conhecido de sites, vale trocar por `matches` específicos ou por injeção sob demanda via `activeTab` + `chrome.scripting`.

*(Isto é uma descrição técnica da arquitetura, não um parecer jurídico formal sobre conformidade LGPD.)*

## ✅ Testes

```powershell
# Pipeline Python (lógica de features, validações)
pip install -r training/requirements.txt
pytest training/tests

# Extensão (features.mjs, spelling.mjs) — requer Node.js instalado
node --test extension/lib
```
