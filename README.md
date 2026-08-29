Markdown
# SignForest — Reconhecimento de LIBRAS com MediaPipe e Random Forest

Projeto desenvolvido para o Trabalho de Conclusão de Curso (TCC) focado no reconhecimento de sinais estáticos do alfabeto datilológico da Língua Brasileira de Sinais (**LIBRAS**). O pipeline processa vídeos de fontes públicas, extrai pontos-chave anatômicos das mãos via MediaPipe, treina um classificador Random Forest e exporta o modelo treinado para inferência em tempo real via extensão do Google Chrome usando ONNX Runtime Web.

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
     [ dataset/data.pickle ]
          ↓  (training/train_classifier.py)
     Random Forest Classifier (Scikit-Learn)
          ↓
     [ training/model.p ]
          ↓  (training/export_model.py)
     [ extension/assets/onnx/model.onnx ]
          ↓
   EXTENSÃO GOOGLE CHROME (Sidepanel / WebCam / ONNX Runtime Web)
📁 Estrutura de Diretórios
Plaintext
SignForest-Extension-adaptado/
│
├── dataset/
│   ├── videos/                  # Vídeos brutos organizados por subpastas (A/, B/, C/...)
│   ├── frames/                  # Frames extraídos automaticamente (ignorado no git)
│   ├── processed/               # Dados intermediários de processamento
│   └── data.pickle              # Dicionário serializado contendo features (42) e labels
│
├── training/
│   ├── extract_frames.py        # Extrai frames dos vídeos em taxa controlada (8 FPS)
│   ├── create_dataset.py        # Processa frames com MediaPipe e gera o data.pickle
│   ├── train_classifier.py      # Treina o modelo Random Forest e gera model.p
│   ├── hyperparameter_search.py # Validação cruzada e busca em grade de parâmetros
│   ├── export_model.py          # Converte o modelo Scikit-Learn (.p) para formato ONNX
│   ├── inference_reference.py   # Script de teste e referência para inferência local
│   └── requirements.txt         # Dependências do ambiente Python
│
├── extension/                   # Extensão Google Chrome (Manifest V3)
│   ├── assets/
│   │   ├── mediapipe/           # Dependências locais do MediaPipe Tasks Vision
│   │   └── onnx/
│   │       ├── labels.json      # Mapeamento das classes do modelo
│   │       └── model.onnx       # Modelo exportado para inferência no navegador
│   ├── background.js            # Service worker da extensão
│   ├── content.js               # Script de conteúdo injetado
│   ├── manifest.json            # Manifesto de configuração da extensão
│   ├── sidepanel.css            # Estilos da interface lateral
│   ├── sidepanel.html           # Interface do painel lateral (captura da webcam)
│   └── sidepanel.js             # Lógica de captura, MediaPipe e ONNX Runtime no navegador
│
├── .gitignore
└── README.md
📚 Base de Dados e Fontes Utilizadas
O treinamento utiliza exclusivamente fontes de LIBRAS:

V-LIBRASIL (UFPE): Base acadêmica com mais de 4.000 vídeos de múltiplos intérpretes.

Dicionário Virtual LIBRAS (YouTube): Acervo com mais de 400 vídeos voltados a serviços e utilidade pública.

Gilbervan Soares (YouTube): Canal voltado ao ensino e divulgação de LIBRAS com mais de 200 vídeos.

Incluir Tecnologia (YouTube): Canal com mais de 1.000 vídeos dedicados ao aprendizado e sinalização em LIBRAS.

Nota de escopo: Fontes em Língua de Sinais Americana (ASL) foram completamente desconsideradas para garantir fidelidade às variações e particularidades do alfabeto em LIBRAS.

🚀 Guia de Execução
1. Instalação do Ambiente
Instale as dependências listadas no requirements.txt:

Bash
pip install -r training/requirements.txt
2. Organização dos Vídeos
Coloque os vídeos organizados por classe dentro de dataset/videos/:

Plaintext
dataset/videos/
├── A/
│   ├── video1.mp4
│   └── video2.mp4
├── B/
│   └── video1.mp4
└── ...
3. Extração Automática de Frames
Bash
python training/extract_frames.py
4. Extração de Features (MediaPipe)
Gera o arquivo dataset/data.pickle contendo as 42 features normalizadas por frame:

Bash
python training/create_dataset.py
5. Treinamento do Modelo
Treina o classificador RandomForestClassifier e gera o relatório de métricas:

Bash
python training/train_classifier.py
6. Exportação para a Extensão (ONNX)
Gera o arquivo extension/assets/onnx/model.onnx compatível com a extensão:

Bash
python training/export_model.py
🧩 Carregamento da Extensão no Chrome
Abra o Google Chrome e acesse chrome://extensions/.

Ative o Modo do desenvolvedor no canto superior direito.

Clique em Carregar sem compactação.

Selecione a pasta extension/ do projeto.

Abra o painel lateral da extensão para iniciar o reconhecimento em tempo real via webcam.