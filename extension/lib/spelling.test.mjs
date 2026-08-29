import { test } from "node:test";
import assert from "node:assert/strict";
import { SpellingEngine } from "./spelling.mjs";

function feed(engine, label, confidence, times) {
  let last = null;
  for (let i = 0; i < times; i++) {
    last = engine.processDetection({ label, confidence });
  }
  return last;
}

test("confirma a letra após atingir o número de frames estáveis", () => {
  const engine = new SpellingEngine({ stableFramesRequired: 5, confidenceThreshold: 0.6 });

  const results = [];
  for (let i = 0; i < 5; i++) {
    results.push(engine.processDetection({ label: "A", confidence: 0.9 }));
  }

  assert.deepEqual(results, [null, null, null, null, "A"]);
});

test("não confirma quando a confiança está abaixo do limiar", () => {
  const engine = new SpellingEngine({ stableFramesRequired: 3, confidenceThreshold: 0.6 });

  const confirmed = feed(engine, "B", 0.3, 10);

  assert.equal(confirmed, null);
});

test("não repete a mesma letra sem um 'release' (sem mão) entre confirmações", () => {
  const engine = new SpellingEngine({
    stableFramesRequired: 3,
    confidenceThreshold: 0.6,
    releaseFramesRequired: 2
  });

  const first = feed(engine, "C", 0.9, 3);
  assert.equal(first, "C");

  // continua segurando o mesmo sinal: não deve confirmar de novo
  const stillHolding = feed(engine, "C", 0.9, 10);
  assert.equal(stillHolding, null);

  // solta a mão (frames sem detecção) o suficiente para liberar
  engine.processNoHand();
  engine.processNoHand();

  const second = feed(engine, "C", 0.9, 3);
  assert.equal(second, "C");
});

test("trocar de sinal antes de estabilizar reinicia a contagem", () => {
  const engine = new SpellingEngine({ stableFramesRequired: 4, confidenceThreshold: 0.6 });

  engine.processDetection({ label: "D", confidence: 0.9 });
  engine.processDetection({ label: "D", confidence: 0.9 });
  engine.processDetection({ label: "E", confidence: 0.9 }); // troca, reinicia streak

  assert.equal(engine.pendingLabel, "E");
  assert.ok(engine.progress < 1);
});

test("progress reflete a fração do streak atual", () => {
  const engine = new SpellingEngine({ stableFramesRequired: 4, confidenceThreshold: 0.6 });

  engine.processDetection({ label: "F", confidence: 0.9 });
  engine.processDetection({ label: "F", confidence: 0.9 });

  assert.equal(engine.progress, 0.5);
});
