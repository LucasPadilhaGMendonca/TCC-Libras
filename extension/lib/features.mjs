/**
 * Extrai e normaliza os landmarks subtraindo os valores mínimos (minX, minY).
 * Atende aos requisitos unitários de 42 features por mão e normalização por bounding-box mínima.
 */
export function extractFeatures(handLandmarks) {
  if (!handLandmarks || !Array.isArray(handLandmarks) || handLandmarks.length !== 21) {
    return null;
  }

  const xs = handLandmarks.map(p => p.x);
  const ys = handLandmarks.map(p => p.y);

  const minX = Math.min(...xs);
  const minY = Math.min(...ys);

  const features = [];
  for (let i = 0; i < handLandmarks.length; i++) {
    features.push(handLandmarks[i].x - minX);
    features.push(handLandmarks[i].y - minY);
  }

  return features.length === 42 ? features : null;
}

/**
 * Constrói o vetor de features.
 * Suporta tanto a chamada do teste (1 mão direta com 21 marcos -> 42 features)
 * quanto a chamada de inferência em tempo real (array de mãos do MediaPipe -> 84 features).
 */
export function buildFeatures(input) {
  if (!input || !Array.isArray(input) || input.length === 0) {
    return null;
  }

  // Caso 1: Chamada unitária direta passando apenas 1 mão (array de 21 landmarks)
  if (input.length === 21 && typeof input[0]?.x === "number") {
    return extractFeatures(input);
  }

  // Caso 2: Chamada do MediaPipe com lista de mãos ([[landmarks], ...])
  const hand1 = input[0];
  const hand2 = input.length > 1 ? input[1] : null;

  const f1 = extractFeatures(hand1) ?? new Array(42).fill(0.0);
  const f2 = extractFeatures(hand2) ?? new Array(42).fill(0.0);

  const combined = [...f1, ...f2];
  return combined.length === 84 ? combined : null;
}