import { test } from "node:test";
import assert from "node:assert/strict";
import { buildFeatures } from "./features.mjs";

function point(x, y) {
  return { x, y };
}

test("retorna 42 valores para 21 landmarks", () => {
  const hand = Array.from({ length: 21 }, (_, i) => point(0.1 * i, 0.2 * i));

  const features = buildFeatures(hand);

  assert.ok(features);
  assert.equal(features.length, 42);
});

test("normaliza subtraindo o mínimo de x e de y", () => {
  const hand = [
    point(0.5, 0.9),
    point(0.2, 0.4),
    point(0.8, 0.6),
    ...Array.from({ length: 18 }, () => point(0.5, 0.5))
  ];

  const features = buildFeatures(hand);

  assert.ok(Math.abs(features[0] - 0.3) < 1e-9);
  assert.ok(Math.abs(features[1] - 0.5) < 1e-9);

  const xs = features.filter((_, i) => i % 2 === 0);
  const ys = features.filter((_, i) => i % 2 === 1);
  assert.ok(Math.min(...xs) < 1e-9);
  assert.ok(Math.min(...ys) < 1e-9);
});

test("rejeita mãos sem exatamente 21 landmarks", () => {
  const hand = Array.from({ length: 5 }, () => point(0.1, 0.1));

  assert.equal(buildFeatures(hand), null);
});

test("rejeita mão ausente/undefined", () => {
  assert.equal(buildFeatures(undefined), null);
  assert.equal(buildFeatures([]), null);
});
