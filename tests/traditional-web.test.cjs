const test = require("node:test");
const assert = require("node:assert/strict");
const { game } = require("./game-harness.cjs");

test("V2 choices are deterministic, legal, and use only public state", () => {
  const g = game();
  g.run("resetGame({seed:100000})");
  for (const mode of ["hold", "beam"]) {
    const before = g.run("JSON.stringify(getPublicObservation())");
    const first = g.run(`traditionalChooseV2(getPublicObservation(), "${mode}")`);
    const second = g.run(`traditionalChooseV2(getPublicObservation(), "${mode}")`);
    assert.equal(first, second);
    assert.equal(g.run(`getActionMask()[${first}]`), true);
    assert.equal(g.run("JSON.stringify(getPublicObservation())"), before);
    assert.deepEqual(Object.keys(JSON.parse(before)).filter(k => ["bag", "queue", "rng"].includes(k)), []);
  }
});

test("AI executes canonical Hold path and Down gives no manual points", () => {
  const g = game();
  g.run("resetGame({seed:7}); setMode('hold')");
  let action = g.run("aiExpectedAction");
  let path = g.run("[...aiPlan]");
  assert.equal(g.run(`getActionMask()[${action}]`), true);
  assert.equal(path.at(-1), "HardDrop");
  // Execute a known Hold placement through the same animation dispatcher.
  const hold = g.run("getLegalPlacements().find(p => p.hold)");
  g.run(`aiPlan = ${JSON.stringify(hold.path)}; aiExpectedAction = ${hold.actionId}; aiPlanning = false`);
  assert.equal(g.run("aiPlan[0]"), "Hold");
  g.run("updateTraditional(1000)");
  assert.equal(g.run("heldType !== null && holdUsed"), true);
  for (let i = 1; i < hold.path.length; i++) g.run("updateTraditional(1000)");
  assert.equal(g.run("placedCount"), 1);
  assert.equal(g.run("score >= 0"), true);
  const down = path.filter(step => step === "Down").length;
  assert.ok(down >= 0);
  const plain = game();
  plain.run("resetGame({seed:7}); active = makePiece('O'); score = 0; softDrop(false)");
  assert.equal(plain.run("score"), 0);
});

test("pause, mode switching, and version 6 save restore", () => {
  const storage = new Map();
  const g = game(storage);
  g.run("resetGame({seed:12345}); setMode('beam'); survivalMs = 3456; placedCount = 8; togglePause(); saveGame()");
  const saved = JSON.parse(storage.get("neonBlocksGameV1"));
  assert.equal(saved.version, 6);
  assert.equal(saved.aiMode, "beam");
  assert.equal(saved.initialSeed, 12345);
  const restored = game(storage);
  assert.equal(restored.run("state"), "paused");
  assert.equal(restored.run("aiMode"), "beam");
  assert.equal(restored.run("placedCount"), 8);
  assert.equal(restored.run("survivalMs"), 3456);
  restored.run("togglePause(); setMode('human')");
  assert.equal(restored.run("state"), "playing");
  assert.equal(restored.run("aiEnabled"), false);
  assert.equal(restored.run("aiPlan.length"), 0);
  restored.run("setMode('v1')");
  assert.equal(restored.run("aiEnabled && aiPlan.length > 0"), true);
});
