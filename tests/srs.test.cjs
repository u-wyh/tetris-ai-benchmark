const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../tetris-game/game.js"), "utf8");

function game(storage = new Map()) {
  const context2d = () => ({
    save() {}, restore() {}, clearRect() {}, fillRect() {}, beginPath() {},
    moveTo() {}, lineTo() {}, stroke() {}, translate() {}, roundRect() {}, fill() {},
    createLinearGradient() { return { addColorStop() {} }; }
  });
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      width: id === "gameCanvas" ? 320 : 144,
      height: id === "gameCanvas" ? 640 : id === "nextCanvas" ? 218 : 92,
      textContent: "", lastChild: { textContent: "" },
      classList: { add() {}, remove() {}, toggle() {} },
      setAttribute() {}, addEventListener() {}, getContext: context2d
    });
    return elements.get(id);
  };
  const sandbox = vm.createContext({
    document: {
      hidden: false,
      querySelector: selector => element(selector.slice(1)),
      querySelectorAll: () => [],
      addEventListener() {}
    },
    window: { addEventListener() {} },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value)
    },
    performance: { now: () => 1000 },
    requestAnimationFrame() {}, setInterval() {},
    location: { search: "" }, URLSearchParams
  });
  vm.runInContext(source, sandbox);
  return { run: code => vm.runInContext(code, sandbox), storage };
}

function snapshot(g) {
  return g.run("JSON.stringify({ active, lockElapsed, board, score, lines, queue, heldType, holdUsed })");
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
