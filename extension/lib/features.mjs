/**
 * Normaliza os landmarks de uma mão em relação ao pulso (ponto 0).
 * Retorna sempre 42 floats (21 marcos * 2 eixos).
 */
function normalizeHand(landmarks) {
  if (!landmarks || landmarks.length === 0) {
    return new Array(42).fill(0.0);
  }

  const base = landmarks[0];
  const features = [];

  for (let i = 0; i < landmarks.length; i++) {
    features.push(landmarks[i].x - base.x);
    features.push(landmarks[i].y - base.y);
  }

  return features;
}

/**
 * Constrói o vetor bimanual de entrada com exatamente 84 valores.
 * @param {Array} multiHandLandmarks Array de mãos retornado pelo MediaPipe
 * @returns {Array<number>|null} Vetor de 84 floats ou null se nenhuma mão for visível
 */
export function buildFeatures(multiHandLandmarks) {
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
    return null;
  }

  const hand1 = multiHandLandmarks[0];
  const hand2 = multiHandLandmarks.length > 1 ? multiHandLandmarks[1] : null;

  const featuresHand1 = normalizeHand(hand1);
  const featuresHand2 = normalizeHand(hand2);

  const combined = [...featuresHand1, ...featuresHand2];
  return combined.length === 84 ? combined : null;
}