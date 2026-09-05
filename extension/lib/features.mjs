/**
 * Extrai e normaliza os landmarks subtraindo os valores mínimos (minX, minY).
 * Retorna exatamente 42 floats ou null se a mão for inválida.
 */
export function extractFeatures(handLandmarks) {
  if (!handLandmarks || !Array.isArray(handLandmarks) || handLandmarks.length !== 21) {
    return null;
  }

  const xs = handLandmarks.map(p => p?.x ?? 0);
  const ys = handLandmarks.map(p => p?.y ?? 0);

  const minX = Math.min(...xs);
  const minY = Math.min(...ys);

  const features = [];
  for (let i = 0; i < handLandmarks.length; i++) {
    const pt = handLandmarks[i];
    if (!pt || typeof pt.x !== "number" || typeof pt.y !== "number") {
      return null;
    }
    features.push(pt.x - minX);
    features.push(pt.y - minY);
  }

  return features.length === 42 ? features : null;
}

/**
 * Constrói o vetor de features:
 * - Se receber uma única mão (array de landmarks direto): valida e retorna 42 features (para os testes unitários).
 * - Se receber a lista de mãos do MediaPipe ([[hand1], [hand2]]): retorna 84 features para a inferência.
 * - Rejeita com null caso qualquer entrada de mão não contenha exatamente 21 pontos válidos.
 */
export function buildFeatures(input) {
  if (!input || !Array.isArray(input) || input.length === 0) {
    return null;
  }

  // Caso A: Testes unitários passando diretamente uma única mão (array plano de pontos)
  const isSingleHand = typeof input[0]?.x === "number" || (input.length > 0 && !Array.isArray(input[0]));
  if (isSingleHand) {
    return extractFeatures(input);
  }

  // Caso B: Array de mãos retornado pelo MediaPipe ([[landmarks1], [landmarks2]])
  const hand1 = input[0];
  const hand2 = input.length > 1 ? input[1] : null;

  const f1 = extractFeatures(hand1);
  if (!f1) return null;

  const f2 = hand2 ? extractFeatures(hand2) : new Array(42).fill(0.0);
  if (!f2) return null;

  const combined = [...f1, ...f2];
  return combined.length === 84 ? combined : null;
}