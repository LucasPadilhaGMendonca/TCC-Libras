export class SpellingEngine {
  constructor(options = {}) {
    this.stableFramesRequired = options.stableFramesRequired ?? options.stabilityThreshold ?? 8;
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

    // Descarta detecções abaixo do limiar de confiança
    if (prediction.confidence !== null && prediction.confidence !== undefined) {
      if (prediction.confidence < this.confidenceThreshold) {
        return null;
      }
    }

    const current = prediction.label;

    // Impede repetição enquanto a mesma pose estiver sendo sustentada
    if (current === this.lastConfirmed) {
      this.progress = 0;
      return null;
    }

    // Detecção ativa reinicia a contagem de soltura
    this.releaseStreak = 0;

    if (this.pendingLabel === current) {
      this.streak++;
    } else {
      this.pendingLabel = current;
      this.streak = 1;
    }

    this.progress = Math.min(1, this.streak / this.stableFramesRequired);

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
      this.lastConfirmed = null; // Libera confirmação do mesmo caractere
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