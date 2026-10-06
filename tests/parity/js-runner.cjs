"use strict";

const fs = require("node:fs");
const { game } = require("../game-harness.cjs");

const request = JSON.parse(fs.readFileSync(0, "utf8"));
if (!Number.isInteger(request.seed) || request.seed < 0 || request.seed > 0xffffffff) {
  throw new Error("seed must be uint32");
}
const g = game();
g.run(`resetGame({seed:${request.seed}})`);

if (request.fixture === "single_clear" || request.fixture === "double_clear"
    || request.fixture === "level_up") {
  g.run('active = makePiece("O"); for (let x = 0; x < 10; x++) if (x !== 3 && x !== 4) board[19][x] = "Z"');
  if (request.fixture === "double_clear") g.run('for (let x = 0; x < 10; x++) if (x !== 3 && x !== 4) board[18][x] = "Z"');
  if (request.fixture === "level_up") g.run('lines = 9; level = 1');
} else if (request.fixture === "triple_clear" || request.fixture === "quad_clear") {
  g.run('active = makePiece("I"); for (let y = 16; y < 20; y++) for (let x = 0; x < 10; x++) if (x !== 4) board[y][x] = "Z"');
  if (request.fixture === "triple_clear") g.run('board[16].fill(null)');
} else if (request.fixture === "block_out") {
  g.run('active = makePiece("O"); board[2][4] = "Z"');
} else if (request.fixture === "lock_out") {
  g.run('active = makePiece("I"); active.y = -2; for (let x = 3; x <= 6; x++) board[0][x] = "Z"');
} else if (request.fixture === "partial_lock") {
  g.run('active = makePiece("O"); active.x = 0; active.y = -1; board[1][0] = "Z"');
} else if (request.fixture === "srs_floor") {
  g.run('active = makePiece("T"); active.y = 18');
} else if (request.fixture === "tuck") {
  g.run(`active = makePiece("T");
    ["....#.....", "......#...", "#.......#.", ".##.......",
     "...##.....", ".....##...", ".......##.", ".#.......#"]
      .forEach((row, index) => [...row].forEach((cell, x) => {
        if (cell === "#") board[index + 12][x] = "Z";
      }));`);
} else if (request.fixture !== undefined) {
  throw new Error(`Unknown fixture: ${request.fixture}`);
}

function snapshot(step) {
  return g.run(`(() => {
    const o = getPublicObservation();
    const result = {step:${step}, board:o.board, current:o.currentPiece, next3:o.next,
      hold:o.hold, holdAvailable:o.holdAvailable, score:o.score, lines:o.lines,
      level:o.level, gameOver:o.gameOver, actionMask:getActionMask()};
    if (${Boolean(request.includePlacements)}) result.placements = getLegalPlacements();
    return result;
  })()`);
}

function chooseAction(step) {
  const p = g.run("getLegalPlacements()");
  if (!p.length) return null;
  const strategy = request.strategy || "mixed";
  if (strategy === "left") return [...p].sort((a, b) => a.x - b.x || b.y - a.y || a.actionId - b.actionId)[0].actionId;
  if (strategy === "right") return [...p].sort((a, b) => b.x - a.x || b.y - a.y || a.actionId - b.actionId)[0].actionId;
  if (strategy === "hold") {
    const wanted = p.find(item => item.hold === (step % 2 === 0 ? 1 : 0) && item.rotation === step % 4);
    if (wanted) return wanted.actionId;
  }
  return p[(step * 37 + request.seed % 97) % p.length].actionId;
}

function applyAction(action) {
  if (!Number.isInteger(action) || action < 0 || action >= 1840) throw new Error(`Invalid action ${action}`);
  // Each placement path is a sequence of the reference game's ordinary operations.
  g.run(`(() => {
    const placement = getLegalPlacements().find(item => item.actionId === ${action});
    if (!placement) throw new Error("Illegal placement action ${action}");
    for (const step of placement.path) {
      if (step === "Hold") holdPiece();
      else if (step === "Left") move(-1);
      else if (step === "Right") move(1);
      else if (step === "Down") softDrop(false);
      else if (step === "RotateCW") rotate();
      else if (step === "HardDrop") hardDrop();
    }
  })()`);
}

const output = {seed: request.seed, actions: [], states: [snapshot(-1)]};
const actions = request.actions || [];
const count = request.steps === undefined ? actions.length : request.steps;
for (let step = 0; step < count; step++) {
  if (g.run('state') !== "playing") break;
  const action = request.steps === undefined ? actions[step] : chooseAction(step);
  if (action === null) break;
  applyAction(action);
  output.actions.push(action);
  output.states.push(snapshot(step));
}
process.stdout.write(JSON.stringify(output));
