// Mesma lógica de normalização usada em training/create_dataset.py:
// para cada um dos 21 landmarks da mão, calcula (x - min(X), y - min(Y)),
// gerando 42 features na ordem [x0,y0,x1,y1,...,x20,y20].
export function buildFeatures(hand) {
  if (!hand || hand.length !== 21) return null;

  const xs = hand.map(p => p.x);
  const ys = hand.map(p => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);

  const features = [];

  for (const p of hand) {
    features.push(p.x - minX);
    features.push(p.y - minY);
  }

  return features.length === 42 ? features : null;
}
