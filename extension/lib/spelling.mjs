export class SpellingEngine {
  constructor(options = {}) {
    this.stabilityThreshold = options.stabilityThreshold ?? 8; // Quadros consecutivos para confirmar
    this.history = [];
    this.lastConfirmed = null;
    this.progress = 0;
  }

  processDetection(prediction) {
    if (!prediction || !prediction.label) {
      return this.processNoHand();
    }

    const current = prediction.label;

    // Se acabou de confirmar esse sinal e ainda está segurando a mão na mesma posição, não digita de novo
    if (current === this.lastConfirmed) {
      this.progress = 0;
      return null;
    }

    if (this.history.length > 0 && this.history[this.history.length - 1] === current) {
      this.history.push(current);
    } else {
      this.history = [current];
    }

    this.progress = Math.min(1, this.history.length / this.stabilityThreshold);

    if (this.history.length >= this.stabilityThreshold) {
      this.lastConfirmed = current;
      this.history = [];
      this.progress = 0;
      return current;
    }

    return null;
  }

  processNoHand() {
    this.history = [];
    this.progress = 0;
    this.lastConfirmed = null; // Libera para fazer o mesmo sinal novamente após abaixar as mãos
    return null;
  }

  reset() {
    this.history = [];
    this.progress = 0;
    this.lastConfirmed = null;
  }
}