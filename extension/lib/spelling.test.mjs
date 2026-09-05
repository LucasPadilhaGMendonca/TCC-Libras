export class SpellingEngine {
  constructor(options = {}) {
    this.stableFramesRequired = options.stableFramesRequired ?? 8;
    this.confidenceThreshold = options.confidenceThreshold ?? 0.6;
    this.releaseFramesRequired = options.releaseFramesRequired ?? 2;

    this.streak = 0;
    this.pendingLabel = null;
    this.lastConfirmed = null;
    this.releaseStreak = 0;
    this.progress = 0;
  }

  processDetection(prediction) {
    if (!prediction || !prediction.label) {
      return this.processNoHand();
    }

    // 1. Rejeita predição abaixo da confiança mínima configurada
    if (prediction.confidence !== null && prediction.confidence !== undefined) {
      if (prediction.confidence < this.confidenceThreshold) {
        return null;
      }
    }

    const current = prediction.label;

    // 2. Bloqueia repetição contínua do mesmo sinal enquanto não houver soltura da mão
    if (current === this.lastConfirmed) {
      this.progress = 0;
      return null;
    }

    // Ao detectar mão válida, zera a contagem de soltura
    this.releaseStreak = 0;

    // 3. Atualiza ou reinicia streak
    if (this.pendingLabel === current) {
      this.streak++;
    } else {
      this.pendingLabel = current;
      this.streak = 1;
    }

    this.progress = Math.min(1, this.streak / this.stableFramesRequired);

    // 4. Confirmação do sinal
    if (this.streak >= this.stableFramesRequired) {
      const confirmed = this.pendingLabel;
      this.lastConfirmed = confirmed;
      this.pendingLabel = null;
      this.streak = 0;
      this.progress = 0;
      return confirmed;
    }

    return null;
  }

  processNoHand() {
    this.streak = 0;
    this.pendingLabel = null;
    this.progress = 0;

    this.releaseStreak++;
    if (this.releaseStreak >= this.releaseFramesRequired) {
      this.lastConfirmed = null; // Libera para confirmar a mesma palavra novamente
    }

    return null;
  }

  reset() {
    this.streak = 0;
    this.pendingLabel = null;
    this.lastConfirmed = null;
    this.releaseStreak = 0;
    this.progress = 0;
  }
}