// Máquina de estados que transforma uma sequência ruidosa de predições por
// frame em letras confirmadas, evitando (a) ruído/transição entre sinais e
// (b) repetição infinita da mesma letra enquanto o usuário mantém a mão parada.
//
// Regra: uma letra só é confirmada depois de aparecer de forma estável (mesma
// label, confiança acima do limiar) por `stableFramesRequired` frames
// seguidos. Depois de confirmada, a mesma letra só pode ser confirmada de
// novo depois que `releaseFramesRequired` frames consecutivos sem mão
// detectada forem observados (o usuário precisa "soltar" o sinal).
export class SpellingEngine {
  constructor({
    stableFramesRequired = 12,
    confidenceThreshold = 0.6,
    releaseFramesRequired = 4
  } = {}) {
    this.stableFramesRequired = stableFramesRequired;
    this.confidenceThreshold = confidenceThreshold;
    this.releaseFramesRequired = releaseFramesRequired;

    this.streakLabel = null;
    this.streakCount = 0;
    this.releaseCount = 0;
    this.lastConfirmed = null;
  }

  get progress() {
    return Math.min(1, this.streakCount / this.stableFramesRequired);
  }

  get pendingLabel() {
    return this.streakLabel;
  }

  reset() {
    this.streakLabel = null;
    this.streakCount = 0;
    this.releaseCount = 0;
    this.lastConfirmed = null;
  }

  // Chamado quando nenhuma mão foi detectada no frame atual.
  processNoHand() {
    this.streakLabel = null;
    this.streakCount = 0;
    this.releaseCount += 1;

    if (this.releaseCount >= this.releaseFramesRequired) {
      this.lastConfirmed = null;
    }

    return null;
  }

  // Chamado com a predição do frame atual ({ label, confidence }).
  // confidence pode ser null quando o modelo não expõe probabilidades.
  // Retorna a letra confirmada neste frame, ou null.
  processDetection({ label, confidence }) {
    this.releaseCount = 0;

    const confidenceOk = confidence == null || confidence >= this.confidenceThreshold;

    if (!confidenceOk) {
      this.streakLabel = null;
      this.streakCount = 0;
      return null;
    }

    if (this.streakLabel === label) {
      this.streakCount += 1;
    } else {
      this.streakLabel = label;
      this.streakCount = 1;
    }

    if (this.streakCount >= this.stableFramesRequired && label !== this.lastConfirmed) {
      this.lastConfirmed = label;
      this.streakCount = 0;
      return label;
    }

    return null;
  }
}
