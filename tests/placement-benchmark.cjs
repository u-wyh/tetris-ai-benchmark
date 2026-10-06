const { performance } = require("node:perf_hooks");
const { game } = require("./game-harness.cjs");

const cases = [[12345, 0], [54321, 1], [2026, 2], [8675309, 3]];
const durations = [];
const counts = [];

for (const [seed, lockedPieces] of cases) {
  const g = game();
  g.run(`resetGame({ seed: ${seed} })`);
  for (let i = 0; i < lockedPieces; i++) {
    const direction = i % 2 === 0 ? -1 : 1;
    for (let step = 0; step < 4; step++) g.run(`move(${direction})`);
    g.run("hardDrop()");
  }
  if (g.run("state") !== "playing") throw new Error(`Seed ${seed} ended before sampling`);
  for (let repeat = 0; repeat < 3; repeat++) {
    const start = performance.now();
    const count = g.run("getActionMask().filter(Boolean).length");
    durations.push(performance.now() - start);
    counts.push(count);
  }
}

const sorted = [...durations].sort((a, b) => a - b);
const averageMs = durations.reduce((sum, ms) => sum + ms, 0) / durations.length;
const medianMs = (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
const averagePlacements = counts.reduce((sum, count) => sum + count, 0) / counts.length;
console.log(JSON.stringify({
  samples: durations.length,
  averageMs: Number(averageMs.toFixed(2)),
  medianMs: Number(medianMs.toFixed(2)),
  averagePlacements: Number(averagePlacements.toFixed(1)),
  minPlacements: Math.min(...counts),
  maxPlacements: Math.max(...counts),
  masksPerSecond: Number((1000 / averageMs).toFixed(1))
}, null, 2));
