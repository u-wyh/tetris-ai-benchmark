const test = require("node:test");
const assert = require("node:assert/strict");
const { game } = require("./game-harness.cjs");

function placements(g) {
  return JSON.parse(g.run("JSON.stringify(getLegalPlacements())"));
}

function mask(g) {
  return Array.from(g.run("getActionMask()"));
}

function fullState(g) {
  return g.run("JSON.stringify({board,active,queue,bag,heldType,holdUsed,initialSeed,gameplayRngState,aiRngState,score,level,lines,lockElapsed,lockResetCount,lockStarted,state})");
}

test("all 1840 action IDs round-trip and invalid coordinates fail", () => {
  const g = game();
  assert.equal(g.run("Array.from({length: 1840}, (_, id) => encodeAction(decodeAction(id)) === id).every(Boolean)"), true);
  assert.equal(g.run("encodeAction(0, 0, 0, -3)"), 0);
  assert.equal(g.run("encodeAction(1, 3, 9, 19)"), 1839);
  assert.equal(g.run("JSON.stringify(decodeAction(1839))"), '{"hold":1,"rotation":3,"x":9,"y":19}');
  assert.throws(() => g.run("encodeAction(0, 0, 10, 0)"), /1840-action space/);
  assert.throws(() => g.run("decodeAction(1840)"), /0 to 1839/);
});

test("mask has 1840 booleans and empty-board placements use occupied bounds", () => {
  const g = game();
  g.run('resetGame({seed:12345}); active = makePiece("O"); holdUsed = true');
  const p = placements(g);
  const m = mask(g);
  assert.equal(m.length, 1840);
  assert.equal(m.every(value => typeof value === "boolean"), true);
  assert.equal(p.length, 9);
  assert.deepEqual(p.map(item => item.x), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(p.every(item => item.y === 18 && item.rotation === 0), true);
  assert.equal(m[g.run("encodeAction(0, 0, 9, 18)")], false);
  assert.equal(m[g.run("encodeAction(0, 0, 4, 18)")], true);
  assert.deepEqual(p.find(item => item.x === 4).occupiedCells,
    [{x:4,y:18},{x:5,y:18},{x:4,y:19},{x:5,y:19}]);
});

test("a geometric cavity with no legal entry is masked", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); holdUsed = true; board[16].fill("T"); board[19].fill("T")');
  assert.equal(g.run('collidesOnBoard(board, 4, 17, PIECES.O.matrix)'), false);
  assert.equal(mask(g)[g.run("encodeAction(0, 0, 4, 17)")], false);
});

test("T floor kick and I-specific SRS kick produce reachable paths", () => {
  const t = game();
  t.run('resetGame(); active = makePiece("T"); active.y = 18; holdUsed = true');
  const targetT = placements(t).find(item => item.rotation === 1 && item.x === 3 && item.y === 17);
  assert.deepEqual(targetT.path, ["RotateCW", "HardDrop"]);
  assert.equal(mask(t)[targetT.actionId], true);
  t.run('rotate(); hardDrop()');
  assert.deepEqual(targetT.occupiedCells, [{x:3,y:17},{x:3,y:18},{x:4,y:18},{x:3,y:19}]);
  assert.equal(t.run("board[19][3]"), "T");

  const i = game();
  i.run('resetGame(); active = makePiece("I"); active.y = 17; holdUsed = true');
  const targetI = placements(i).find(item => item.rotation === 1 && item.x === 6 && item.y === 16);
  assert.deepEqual(targetI.path, ["RotateCW", "HardDrop"]);
  i.run("rotate(); hardDrop()");
  assert.equal(i.run("board[19][6]"), "I");
});

test("O and repeated I/S/Z orientations keep only the lowest action ID per occupied cells", () => {
  for (const type of ["O", "I", "S", "Z"]) {
    const g = game();
    g.run(`resetGame(); active = makePiece("${type}"); holdUsed = true`);
    const p = placements(g);
    const keys = p.map(item => item.occupiedCells.map(cell => `${cell.x}:${cell.y}`).join("|"));
    assert.equal(new Set(keys).size, p.length);
    assert.equal(p.every(item => item.rotation < 2), true);
    assert.equal(mask(g).filter(Boolean).length, p.length);
  }
  const o = game();
  o.run('resetGame(); active = makePiece("O"); holdUsed = true');
  assert.equal(mask(o)[o.run("encodeAction(0, 1, 4, 18)")], false);
});

test("Hold branches use public Next or held piece without consuming the real queue", () => {
  const empty = game();
  empty.run('resetGame(); active = makePiece("T"); queue = ["I","J","L","O","S"]; heldType = null');
  const before = fullState(empty);
  const emptyHold = placements(empty).filter(item => item.hold === 1);
  assert.equal(emptyHold.length > 0, true);
  assert.equal(emptyHold.every(item => item.path[0] === "Hold"), true);
  assert.equal(emptyHold.some(item => item.occupiedCells.every(cell => cell.y === 19)), true);
  assert.equal(fullState(empty), before);

  const filled = game();
  filled.run('resetGame(); active = makePiece("T"); queue = ["I","J","L","O","S"]; heldType = "O"');
  const filledBefore = fullState(filled);
  const filledHold = placements(filled).filter(item => item.hold === 1);
  assert.equal(filledHold.length, 9);
  assert.equal(filledHold.every(item => item.rotation === 0), true);
  assert.equal(fullState(filled), filledBefore);
  filled.run("holdUsed = true");
  assert.equal(placements(filled).some(item => item.hold === 1), false);
  assert.equal(mask(filled).slice(920).some(Boolean), false);
});

test("identical occupied cells remain separate actions across Hold branches", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); heldType = "O"');
  const p = placements(g);
  assert.equal(p.length, 18);
  const normal = p.find(item => item.hold === 0 && item.x === 4);
  const held = p.find(item => item.hold === 1 && item.x === 4);
  assert.deepEqual(normal.occupiedCells, held.occupiedCells);
  const m = mask(g);
  assert.equal(m[normal.actionId], true);
  assert.equal(m[held.actionId], true);
});

test("queries leave real game, RNG, score, and lock state unchanged", () => {
  const g = game();
  g.run('resetGame({seed:12345}); board[19][0] = "T"; score = 123; lines = 4; lockElapsed = 200; lockResetCount = 3; lockStarted = true');
  const before = fullState(g);
  placements(g);
  mask(g);
  assert.equal(fullState(g), before);
});

test("the 15-reset budget changes reachability instead of allowing endless floor movement", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); active.y = 18; holdUsed = true; lockStarted = true; lockElapsed = 350; lockResetCount = 15');
  const target = g.run("encodeAction(0, 0, 7, 18)");
  assert.equal(mask(g)[target], false);
  g.run("lockResetCount = 14");
  assert.equal(mask(g)[target], true);
});

test("a reachable placement remains legal even when its lock causes Block Out", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); active.x = 4; active.y = 0; holdUsed = true; board[2][4] = "T"');
  const target = g.run("encodeAction(0, 0, 4, 0)");
  assert.equal(mask(g)[target], true);
  g.run("hardDrop()");
  assert.equal(g.run("state"), "over");
});

test("a reachable Lock Out remains legal in the action mask", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("I"); active.y = -2; holdUsed = true; for (let x = 3; x <= 6; x++) board[0][x] = "T"');
  const target = g.run("encodeAction(0, 0, 3, -1)");
  assert.equal(mask(g)[target], true);
  g.run("hardDrop()");
  assert.equal(g.run("state"), "over");
});

test("the legacy AI can suggest a placement isolated behind an impassable wall", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); holdUsed = true; for (let y = 0; y < 20; y++) board[y][6] = "T"');
  assert.equal(g.run('candidateMoves(board, "O", 0).some(move => move.x === 7)'), true);
  assert.equal(mask(g)[g.run("encodeAction(0, 0, 7, 18)")], false);
});

test("canonical placements, paths, and masks repeat exactly and pause yields no actions", () => {
  const g = game();
  g.run('resetGame({seed:12345}); active = makePiece("T"); active.y = 18; holdUsed = true');
  const before = g.run("JSON.stringify(getLegalPlacements())");
  assert.equal(g.run("JSON.stringify(getLegalPlacements())"), before);
  assert.deepEqual(mask(g), mask(g));
  assert.deepEqual(placements(g).find(item => item.rotation === 1 && item.x === 2 && item.y === 17).path,
    ["Left", "RotateCW", "HardDrop"]);
  g.run('state = "paused"');
  assert.equal(mask(g).some(Boolean), false);
  assert.equal(mask(g).length, 1840);
  g.run('state = "over"');
  assert.equal(mask(g).some(Boolean), false);
  assert.equal(mask(g).length, 1840);
});
