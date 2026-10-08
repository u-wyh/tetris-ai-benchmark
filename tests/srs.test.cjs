const test = require("node:test");
const assert = require("node:assert/strict");
const { game } = require("./game-harness.cjs");

function snapshot(g) {
  return g.run("JSON.stringify({ active, lockElapsed, lockResetCount, lockStarted, board, score, lines, queue, heldType, holdUsed })");
}

test("ordinary rotations follow 0 → R → 2 → L → 0", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.y = 5');
  for (const state of ["R", "2", "L", "0"]) {
    g.run("rotate()");
    assert.equal(g.run("active.rotation"), state);
    assert.equal(g.run("JSON.stringify(active.matrix)"), g.run(`JSON.stringify(pieceMatrix("T", "${state}"))`));
    assert.equal(g.run("active.x"), 3);
    assert.equal(g.run("active.y"), 5);
  }
});

test("JLSTZ kicks off left and right walls in table order", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.rotation = "R"; active.matrix = pieceMatrix("T", "R"); active.x = -1; active.y = 5');
  assert.equal(g.run("collides(active.x, active.y, active.matrix)"), false);
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "2");
  assert.equal(g.run("active.x"), 0);
  g.run('active.rotation = "L"; active.matrix = pieceMatrix("T", "L"); active.x = 8');
  assert.equal(g.run("collides(active.x, active.y, active.matrix)"), false);
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "0");
  assert.equal(g.run("active.x"), 7);
});

test("T floor kick and obstacle kick use the prescribed offsets", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.x = 4; active.y = 18');
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "R");
  assert.equal(g.run("active.x"), 3);
  assert.equal(g.run("active.y"), 17);
  g.run('board = createBoard(); active = makePiece("T"); active.x = 4; active.y = 5; board[7][5] = "J"');
  assert.equal(g.run("collides(active.x, active.y, active.matrix)"), false);
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "R");
  assert.equal(g.run("active.x"), 3);
  assert.equal(g.run("active.y"), 5);
});

test("I uses its separate two-cell kick and all four clockwise transitions", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("I"); active.x = 3; active.y = 5; board[5][5] = "T"');
  assert.equal(g.run("collides(active.x, active.y, active.matrix)"), false);
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "R");
  assert.equal(g.run("active.x"), 1);
  g.run('board = createBoard(); active = makePiece("I"); active.rotation = "R"; active.matrix = pieceMatrix("I", "R"); active.x = 7; active.y = 5');
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "2");
  assert.equal(g.run("active.x"), 6);
  g.run('board = createBoard(); active = makePiece("I"); active.y = 5');
  for (const state of ["R", "2", "L", "0"]) {
    g.run("rotate()");
    assert.equal(g.run("active.rotation"), state);
  }
});

test("O tracks rotation state without moving or changing its shape", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); active.x = 1; active.y = 17; lockElapsed = 100');
  const before = g.run("JSON.stringify(active.matrix)");
  for (const state of ["R", "2", "L", "0"]) {
    g.run("rotate()");
    assert.equal(g.run("active.rotation"), state);
    assert.equal(g.run("active.x"), 1);
    assert.equal(g.run("active.y"), 17);
    assert.equal(g.run("JSON.stringify(active.matrix)"), before);
    assert.equal(g.run("lockElapsed"), 100);
  }
});

test("failed rotation preserves the complete game state", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.x = 4; active.y = 8; lockElapsed = 123; board[10][5] = "I"; board[8][4] = "I"; board[7][4] = "I"; board[11][5] = "I"; board[12][4] = "I"');
  assert.equal(g.run("collides(active.x, active.y, active.matrix)"), false);
  const before = snapshot(g);
  g.run("rotate()");
  assert.equal(snapshot(g), before);
});

test("Hold returns the initial rotation state and a saved game restores rotation", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.y = 5; rotate(); holdPiece(); hardDrop(); holdPiece()');
  assert.equal(g.run("active.type"), "T");
  assert.equal(g.run("active.rotation"), "0");
  assert.equal(g.run("JSON.stringify(active.matrix)"), g.run('JSON.stringify(PIECES.T.matrix)'));
  g.run("rotate(); saveGame()");
  const restored = game(g.storage);
  assert.equal(restored.run("active.rotation"), "R");
  assert.equal(restored.run("heldType"), g.run("heldType"));
  assert.equal(restored.run("holdUsed"), true);
});

test("version 2 saves infer their existing matrix orientation", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.y = 5; rotate(); saveGame()');
  const saved = JSON.parse(g.storage.get("neonBlocksGameV1"));
  saved.version = 2;
  delete saved.active.rotation;
  g.storage.set("neonBlocksGameV1", JSON.stringify(saved));
  const restored = game(g.storage);
  assert.equal(restored.run("active.rotation"), "R");
});

test("movement, drops, line clearing, and AI still execute", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("I"); active.x = 3; active.y = 0; board[19].fill("T"); for (let x = 3; x <= 6; x++) board[19][x] = null');
  const x = g.run("active.x");
  g.run("move(-1); move(1); softDrop(true)");
  assert.equal(g.run("active.x"), x);
  assert.equal(g.run("score"), 1);
  g.run("hardDrop()");
  assert.equal(g.run("lines"), 1);
  assert.equal(g.run("score") >= 101, true);
  const ai = game();
  ai.run("resetGame(); setAI(true)");
  assert.equal(ai.run("aiPlan.length > 0"), true);
  for (let i = 0; i < 100; i++) ai.run("updateAI(100)");
  assert.equal(ai.run("aiEnabled"), true);
});

test("a clear spawn remains playable and an occupied spawn causes Block Out", () => {
  const g = game();
  g.run('resetGame(); board = createBoard(); activatePiece("T")');
  assert.equal(g.run("state"), "playing");
  g.run('board[0][4] = "I"; activatePiece("T")');
  assert.equal(g.run("state"), "over");
});

test("a piece locked wholly above the board causes Lock Out without merging or spawning", () => {
  const g = game();
  g.run('resetGame(); board = createBoard(); active = makePiece("I"); active.y = -2; for (let x = 3; x <= 6; x++) board[0][x] = "T"');
  assert.equal(g.run("collides(active.x, active.y, active.matrix)"), false);
  assert.equal(g.run("collides(active.x, active.y + 1, active.matrix)"), true);
  const boardBefore = g.run("JSON.stringify(board)");
  const queueBefore = g.run("JSON.stringify(queue)");
  g.run("advanceGame(450, 1000)");
  assert.equal(g.run("state"), "over");
  assert.equal(g.run("JSON.stringify(board)"), boardBefore);
  assert.equal(g.run("JSON.stringify(queue)"), queueBefore);
  assert.equal(g.run("lines"), 0);
});

test("a partially visible piece locks its visible cells and play continues", () => {
  const g = game();
  g.run('resetGame(); board = createBoard(); active = makePiece("I"); active.x = 0; active.y = -1; queue[0] = "O"; for (let x = 0; x < 4; x++) board[1][x] = "T"');
  g.run("advanceGame(450, 1000)");
  assert.equal(g.run("state"), "playing");
  assert.equal(g.run("active.type"), "O");
  assert.equal(g.run('board[0].slice(0, 4).every(cell => cell === "I")'), true);
});

test("Partial Lock still clears a completed top row", () => {
  const g = game();
  g.run('resetGame(); board = createBoard(); active = makePiece("I"); active.x = 0; active.y = -1; queue[0] = "O"; for (let x = 0; x < 4; x++) board[1][x] = "T"; for (let x = 4; x < 10; x++) board[0][x] = "T"');
  g.run("advanceGame(450, 1000)");
  assert.equal(g.run("state"), "playing");
  assert.equal(g.run("lines"), 1);
  assert.equal(g.run("score"), 100);
  assert.equal(g.run("board[0].every(cell => cell === null)"), true);
});

test("a piece swapped from Hold still causes Block Out on spawn collision", () => {
  const g = game();
  g.run('resetGame(); board = createBoard(); active = makePiece("I"); heldType = "T"; board[0][4] = "O"; holdPiece()');
  assert.equal(g.run("state"), "over");
  assert.equal(g.run("active.type"), "T");
  assert.equal(g.run("active.rotation"), "0");
});

function nextPieces(g, count) {
  return g.run(`Array.from({ length: ${count} }, () => { const type = active.type; spawnPiece(); return type; }).join("")`);
}

test("a fixed seed repeats the first 100 pieces and another seed differs", () => {
  const first = game();
  const second = game();
  const different = game();
  first.run("resetGame({ seed: 12345 })");
  second.run("resetGame({ seed: 12345 })");
  different.run("resetGame({ seed: 54321 })");
  const sequence = nextPieces(first, 100);
  assert.equal(nextPieces(second, 100), sequence);
  assert.notEqual(nextPieces(different, 100), sequence);
  for (let i = 0; i < 98; i += 7) {
    assert.equal([...sequence.slice(i, i + 7)].sort().join(""), "IJLOSTZ");
  }
});

test("seed zero is valid and an ordinary new game records an automatic seed", () => {
  const fixed = game();
  fixed.run("resetGame({ seed: 0 })");
  assert.equal(fixed.run("initialSeed"), 0);
  const ordinary = game();
  ordinary.run("resetGame(); saveGame()");
  const saved = JSON.parse(ordinary.storage.get("neonBlocksGameV1"));
  assert.equal(Number.isInteger(saved.initialSeed), true);
  assert.equal(saved.initialSeed >= 0 && saved.initialSeed <= 0xffffffff, true);
  assert.equal(saved.gameplayRngState, ordinary.run("gameplayRngState"));
  assert.throws(() => ordinary.run("resetGame({ seed: -1 })"), /unsigned 32-bit integer/);
});

test("the initial seed, gameplay state, and AI state survive save and restore", () => {
  const g = game();
  g.run("resetGame({ seed: 12345 }); shuffledBag(); shuffledBag(); aiRandom(); saveGame()");
  const saved = JSON.parse(g.storage.get("neonBlocksGameV1"));
  assert.equal(saved.version, 6);
  assert.equal(saved.initialSeed, 12345);
  assert.equal(saved.gameplayRngState, g.run("gameplayRngState"));
  assert.equal(saved.aiRngState, g.run("aiRngState"));
  const restored = game(g.storage);
  assert.equal(restored.run("initialSeed"), 12345);
  assert.equal(restored.run("gameplayRngState"), saved.gameplayRngState);
  assert.equal(restored.run("aiRngState"), saved.aiRngState);
  assert.equal(restored.run("Array.from({length: 20}, shuffledBag).flat().join(\"\")"),
    g.run("Array.from({length: 20}, shuffledBag).flat().join(\"\")"));
  assert.equal(restored.run("aiRandom()"), g.run("aiRandom()"));
});

test("refreshing a saved game preserves the subsequent piece sequence", () => {
  const g = game();
  g.run("resetGame({ seed: 12345 })");
  nextPieces(g, 12);
  g.run("saveGame()");
  const restored = game(g.storage);
  assert.equal(nextPieces(restored, 100), nextPieces(g, 100));
});

test("visual random calls cannot change the gameplay sequence", () => {
  const visual = game();
  const plain = game();
  visual.run("resetGame({ seed: 12345 })");
  plain.run("resetGame({ seed: 12345 })");
  visual.run('createLineParticles(5, Array(10).fill("T")); shake = 5; drawBoard()');
  assert.equal(nextPieces(visual, 100), nextPieces(plain, 100));
  assert.equal(visual.run("gameplayRngState"), plain.run("gameplayRngState"));
});

test("AI decisions use their own RNG and leave the 7-Bag sequence unchanged", () => {
  const ai = game();
  const plain = game();
  ai.run("resetGame({ seed: 12345 }); setAI(true); chooseAIMove(); chooseAIMove()");
  plain.run("resetGame({ seed: 12345 })");
  assert.equal(nextPieces(ai, 28), nextPieces(plain, 28));
  assert.equal(ai.run("gameplayRngState"), plain.run("gameplayRngState"));
  assert.notEqual(ai.run("aiRngState"), plain.run("aiRngState"));
});

test("Hold consumes RNG only when normal Next replenishment requires another bag", () => {
  const held = game();
  const normal = game();
  held.run("resetGame({ seed: 12345 }); holdPiece()");
  normal.run("resetGame({ seed: 12345 }); spawnPiece()");
  assert.equal(held.run("gameplayRngState"), normal.run("gameplayRngState"));
  assert.equal(held.run('JSON.stringify({queue, bag})'), normal.run('JSON.stringify({queue, bag})'));
  held.run("hardDrop()");
  normal.run("hardDrop()");
  const stateBeforeSwap = held.run("gameplayRngState");
  const queueBeforeSwap = held.run('JSON.stringify({queue, bag})');
  held.run("holdPiece()");
  assert.equal(held.run("gameplayRngState"), stateBeforeSwap);
  assert.equal(held.run('JSON.stringify({queue, bag})'), queueBeforeSwap);
  assert.equal(held.run("gameplayRngState"), normal.run("gameplayRngState"));
});

test("landing starts 450 ms of delay; soft drop uses the same delay", () => {
  const natural = game();
  natural.run('resetGame(); active = makePiece("O"); active.y = 18');
  natural.run("advanceGame(449, 1000)");
  assert.equal(natural.run("state"), "playing");
  assert.equal(natural.run("board[19][4]"), null);
  assert.equal(natural.run("lockElapsed"), 449);
  natural.run("advanceGame(1, 1001)");
  assert.equal(natural.run("board[19][4]"), "O");
  assert.equal(natural.run("lockResetCount"), 0);
  assert.equal(natural.run("lockElapsed"), 0);

  const soft = game();
  soft.run('resetGame(); active = makePiece("O"); active.y = 17; softDrop(true)');
  assert.equal(soft.run("active.y"), 18);
  assert.equal(soft.run("lockStarted"), true);
  soft.run("advanceGame(449, 1000)");
  assert.equal(soft.run("board[19][4]"), null);
  soft.run("advanceGame(1, 1001)");
  assert.equal(soft.run("board[19][4]"), "O");
});

test("successful grounded left, right, and SRS rotations reset the clock", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); active.y = 18; advanceGame(300, 1000)');
  g.run("move(-1)");
  assert.equal(g.run("lockResetCount"), 1);
  assert.equal(g.run("lockElapsed"), 0);
  g.run("advanceGame(300, 1300); move(1)");
  assert.equal(g.run("lockResetCount"), 2);
  assert.equal(g.run("lockElapsed"), 0);
  g.run("advanceGame(449, 1749)");
  assert.equal(g.run("board[19][4]"), null);
  g.run("advanceGame(1, 1750)");
  assert.equal(g.run("board[19][4]"), "O");

  const rotating = game();
  rotating.run('resetGame(); active = makePiece("T"); active.y = 18; advanceGame(300, 1000); rotate()');
  assert.equal(rotating.run("active.rotation"), "R");
  assert.equal(rotating.run("lockResetCount"), 1);
  assert.equal(rotating.run("lockElapsed"), 0);
});

test("failed actions and movement in the air cannot consume lock resets", () => {
  const blockedMove = game();
  blockedMove.run('resetGame(); active = makePiece("O"); active.x = 0; active.y = 18; advanceGame(300, 1000); move(-1)');
  assert.equal(blockedMove.run("lockResetCount"), 0);
  assert.equal(blockedMove.run("lockElapsed"), 300);

  const blockedRotation = game();
  blockedRotation.run('resetGame(); active = makePiece("T"); active.x = 4; active.y = 18; board[17][4] = "I"; advanceGame(300, 1000)');
  const before = snapshot(blockedRotation);
  blockedRotation.run("rotate()");
  assert.equal(snapshot(blockedRotation), before);
  assert.equal(blockedRotation.run("lockResetCount"), 0);
  assert.equal(blockedRotation.run("lockElapsed"), 300);

  const airborne = game();
  airborne.run('resetGame(); active = makePiece("O"); active.y = 5; move(-1); move(1); rotate()');
  assert.equal(airborne.run("lockResetCount"), 0);
  assert.equal(airborne.run("lockStarted"), false);
});

test("only 15 grounded actions can reset; the 16th move and later rotation cannot", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); active.y = 18');
  for (let i = 0; i < 15; i++) g.run(i % 2 === 0 ? "move(-1)" : "move(1)");
  assert.equal(g.run("lockResetCount"), 15);
  g.run("advanceGame(300, 1000); move(1)");
  assert.equal(g.run("active.x"), 4);
  assert.equal(g.run("lockResetCount"), 15);
  assert.equal(g.run("lockElapsed"), 300);
  g.run("rotate()");
  assert.equal(g.run("active.rotation"), "R");
  assert.equal(g.run("lockResetCount"), 15);
  assert.equal(g.run("lockElapsed"), 300);
  g.run("advanceGame(149, 1149)");
  assert.equal(g.run("board[19][4]"), null);
  g.run("advanceGame(1, 1150)");
  assert.equal(g.run("board[19][4]"), "O");
});

test("leaving the ground after 15 resets does not restore another delay", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("O"); active.y = 17; board[19][4] = "T"');
  for (let i = 0; i < 15; i++) g.run(i % 2 === 0 ? "move(-1)" : "move(1)");
  g.run("advanceGame(300, 1000); move(1); move(1)");
  assert.equal(g.run("lockResetCount"), 15);
  assert.equal(g.run("lockElapsed"), 300);
  assert.equal(g.run("collides(active.x, active.y + 1, active.matrix)"), false);
  g.run("advanceGame(150, 1150)");
  assert.equal(g.run("lockElapsed"), 450);
  assert.equal(g.run("state"), "playing");
  g.run("softDrop(true); advanceGame(1, 1151)");
  assert.equal(g.run("board[19][5]"), "O");
});

test("hard drop locks immediately despite the reset cap; Hold starts a fresh cycle", () => {
  const hard = game();
  hard.run('resetGame(); active = makePiece("O"); lockElapsed = 449; lockResetCount = 15; lockStarted = true; hardDrop()');
  assert.equal(hard.run("board[19][4]"), "O");
  assert.equal(hard.run("lockResetCount"), 0);
  assert.equal(hard.run("lockElapsed"), 0);
  assert.equal(hard.run("lockStarted"), false);

  const held = game();
  held.run('resetGame(); active = makePiece("O"); active.y = 18; advanceGame(300, 1000); move(-1); holdPiece()');
  assert.equal(held.run("lockResetCount"), 0);
  assert.equal(held.run("lockElapsed"), 0);
  assert.equal(held.run("lockStarted"), false);
});

test("refreshing during lock delay preserves timer and reset count", () => {
  const g = game();
  g.run('resetGame({ seed: 12345 }); active = makePiece("O"); active.y = 18; advanceGame(300, 1000); move(-1); advanceGame(200, 1200); saveGame()');
  const restored = game(g.storage);
  assert.equal(restored.run("lockResetCount"), 1);
  assert.equal(restored.run("lockElapsed"), 200);
  assert.equal(restored.run("lockStarted"), true);
  restored.run("advanceGame(250, 1450)");
  assert.equal(restored.run("board[19][3]"), "O");
});

test("version 4 saves start with an unused lock delay budget", () => {
  const g = game();
  g.run('resetGame({ seed: 12345 }); active = makePiece("O"); active.y = 18; saveGame()');
  const saved = JSON.parse(g.storage.get("neonBlocksGameV1"));
  saved.version = 4;
  delete saved.lockElapsed;
  delete saved.lockResetCount;
  delete saved.lockStarted;
  g.storage.set("neonBlocksGameV1", JSON.stringify(saved));
  const restored = game(g.storage);
  assert.equal(restored.run("lockElapsed"), 0);
  assert.equal(restored.run("lockResetCount"), 0);
  assert.equal(restored.run("lockStarted"), false);
});

test("the existing AI keeps locking pieces through the game loop", () => {
  const g = game();
  g.run("resetGame({ seed: 12345 }); setAI(true)");
  let locked = false;
  for (let i = 0; i < 300 && !locked; i++) {
    g.run(`advanceGame(50, ${1000 + i * 50})`);
    locked = g.run("board.some(row => row.some(Boolean))");
  }
  assert.equal(locked, true);
  assert.equal(g.run("aiEnabled"), true);
});

test("one through four cleared lines score 100/300/500/800 at level one", () => {
  for (const [count, points] of [[1, 100], [2, 300], [3, 500], [4, 800]]) {
    const g = game();
    g.run(`resetGame(); board = createBoard(); for (let y = 20 - ${count}; y < 20; y++) board[y].fill("T"); clearLines()`);
    assert.equal(g.run("score"), points);
    assert.equal(g.run("lines"), count);
    assert.equal(g.run("level"), 1);
  }
});

test("no line clear adds no score and level advances every ten cumulative lines", () => {
  const g = game();
  g.run('resetGame(); board = createBoard(); board[19][0] = "T"; clearLines()');
  assert.equal(g.run("score"), 0);
  assert.equal(g.run("lines"), 0);
  for (const count of [4, 4, 2]) {
    g.run(`board = createBoard(); for (let y = 20 - ${count}; y < 20; y++) board[y].fill("T"); clearLines()`);
  }
  assert.equal(g.run("lines"), 10);
  assert.equal(g.run("level"), 2);
  assert.equal(g.run("score"), 1900);
  g.run('board = createBoard(); board[19].fill("T"); clearLines()');
  assert.equal(g.run("score"), 2100);
  assert.equal(g.run("level"), 2);
});

test("a threshold-crossing clear uses the level before the clear", () => {
  const g = game();
  g.run('resetGame(); lines = 9; level = 1; board = createBoard(); board[18].fill("T"); board[19].fill("T"); clearLines()');
  assert.equal(g.run("lines"), 11);
  assert.equal(g.run("level"), 2);
  assert.equal(g.run("score"), 300);
  g.run('board[19].fill("T"); clearLines()');
  assert.equal(g.run("score"), 500);
});

test("soft drop awards one point per moved row; hard drop awards two", () => {
  const soft = game();
  soft.run('resetGame(); active = makePiece("O"); softDrop(true); softDrop(true); softDrop(true)');
  assert.equal(soft.run("score"), 3);
  soft.run("softDrop(false)");
  assert.equal(soft.run("score"), 3);
  soft.run("active.y = 18; softDrop(true)");
  assert.equal(soft.run("score"), 3);

  const hard = game();
  hard.run('resetGame(); active = makePiece("O"); active.y = 16; hardDrop()');
  assert.equal(hard.run("score"), 4);
  assert.equal(hard.run("board[19][4]"), "O");
  const zeroDistance = game();
  zeroDistance.run('resetGame(); active = makePiece("O"); active.y = 18; hardDrop()');
  assert.equal(zeroDistance.run("score"), 0);
});

test("Hold, SRS rotation, and lock delay do not award points", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.y = 18; holdPiece()');
  assert.equal(g.run("score"), 0);
  g.run('active = makePiece("T"); active.y = 18; rotate(); advanceGame(300, 1000)');
  assert.equal(g.run("score"), 0);
  g.run('board = createBoard(); heldType = "O"; holdUsed = false; holdPiece()');
  assert.equal(g.run("score"), 0);
});

test("seed choice does not change scoring for the same piece and actions", () => {
  const first = game();
  const second = game();
  first.run('resetGame({ seed: 12345 }); active = makePiece("O"); softDrop(true); hardDrop()');
  second.run('resetGame({ seed: 54321 }); active = makePiece("O"); softDrop(true); hardDrop()');
  assert.equal(first.run("score"), second.run("score"));
  assert.equal(first.run("score"), 35);
});

test("Block Out and Lock Out add no extra score", () => {
  const block = game();
  block.run('resetGame(); score = 17; board[0][4] = "I"; activatePiece("T")');
  assert.equal(block.run("state"), "over");
  assert.equal(block.run("score"), 17);

  const lock = game();
  lock.run('resetGame(); score = 17; board = createBoard(); active = makePiece("I"); active.y = -2; for (let x = 3; x <= 6; x++) board[0][x] = "T"; mergePiece()');
  assert.equal(lock.run("state"), "over");
  assert.equal(lock.run("score"), 17);
});

test("public observation exposes exactly three Next pieces from a longer queue", () => {
  const g = game();
  g.run('resetGame({ seed: 12345 }); queue = ["I", "J", "L", "O", "S", "T"]; bag = ["Z"]');
  const observation = JSON.parse(g.run("JSON.stringify(getPublicObservation())"));
  assert.deepEqual(observation.next, ["I", "J", "L"]);
  assert.equal(g.run("queue.length"), 6);
  assert.equal(g.run("NEXT_PREVIEW_COUNT"), 3);
  assert.equal(g.run("getPublicObservation().next.length"), 3);
});

test("public observation omits hidden queue, bag, RNG, seed, and save data", () => {
  const g = game();
  g.run('resetGame({ seed: 12345 }); queue = ["I", "J", "L", "O", "S"]; bag = ["Z"]; saveGame()');
  const observation = JSON.parse(g.run("JSON.stringify(getPublicObservation())"));
  assert.deepEqual(Object.keys(observation).sort(),
    ["board", "currentPiece", "gameOver", "hold", "holdAvailable", "level", "lines", "lock", "next", "phase", "score"].sort());
  assert.deepEqual(Object.keys(observation.lock).sort(), ["elapsedMs", "resetCount", "started"]);
  assert.equal(JSON.stringify(observation).includes('"bag"'), false);
  assert.equal(JSON.stringify(observation).includes('"gameplayRngState"'), false);
  assert.equal(JSON.stringify(observation).includes('"aiRngState"'), false);
  assert.equal(JSON.stringify(observation).includes('"initialSeed"'), false);
  assert.equal(JSON.stringify(observation).includes('"savedAt"'), false);
  assert.equal(JSON.stringify(observation).includes('"O","S"'), false);
});

test("public board, current piece, and Next are detached frozen snapshots", () => {
  const g = game();
  g.run('resetGame(); board[0][0] = "T"; active = makePiece("L"); queue = ["I", "J", "L", "O", "S"]');
  g.run('var view = getPublicObservation(); view.board[0][0] = "Z"; view.currentPiece.matrix[0][2] = 0; view.next[0] = "Z"');
  assert.equal(g.run("Object.isFrozen(view) && Object.isFrozen(view.board) && Object.isFrozen(view.board[0])"), true);
  assert.equal(g.run("Object.isFrozen(view.currentPiece) && Object.isFrozen(view.currentPiece.matrix[0]) && Object.isFrozen(view.next)"), true);
  assert.equal(g.run("board[0][0]"), "T");
  assert.equal(g.run("active.matrix[0][2]"), 1);
  assert.equal(g.run("queue[0]"), "I");
  const oldX = g.run("view.currentPiece.x");
  g.run("move(-1)");
  assert.equal(g.run("view.currentPiece.x"), oldX);
  assert.equal(g.run("active.x"), oldX - 1);
});

test("public current piece, Hold availability, score, level, lines, and lock state are accurate", () => {
  const g = game();
  g.run('resetGame(); active = makePiece("T"); active.rotation = "R"; active.matrix = pieceMatrix("T", "R"); active.x = 2; active.y = 3; heldType = "I"; holdUsed = false; score = 1234; lines = 12; level = 2; lockElapsed = 175; lockResetCount = 4; lockStarted = true');
  const visible = JSON.parse(g.run("JSON.stringify(getPublicObservation())"));
  assert.equal(visible.board.length, 20);
  assert.equal(visible.board[0].length, 10);
  assert.deepEqual(visible.currentPiece, {
    type: "T", matrix: JSON.parse(g.run('JSON.stringify(pieceMatrix("T", "R"))')),
    rotation: "R", x: 2, y: 3
  });
  assert.equal(visible.hold, "I");
  assert.equal(visible.holdAvailable, true);
  assert.equal(visible.score, 1234);
  assert.equal(visible.level, 2);
  assert.equal(visible.lines, 12);
  assert.equal(visible.gameOver, false);
  assert.deepEqual(visible.lock, { elapsedMs: 175, resetCount: 4, started: true });
  g.run("holdUsed = true");
  assert.equal(g.run("getPublicObservation().holdAvailable"), false);
  g.run('state = "over"');
  assert.equal(g.run("getPublicObservation().gameOver"), true);
  assert.equal(g.run("getPublicObservation().holdAvailable"), false);
  assert.equal(g.run("getPublicObservation().currentPiece"), null);
});

test("the existing AI continues using only queue[0] without the new observation", () => {
  const g = game();
  g.run('resetGame({ seed: 12345 }); queue = ["I", "J", "L", "O", "S"]; setAI(true)');
  assert.equal(g.run("chooseAIMove() !== null"), true);
  assert.equal(g.run("aiEnabled"), true);
  assert.equal(g.run("getPublicObservation().next.length"), 3);
});
