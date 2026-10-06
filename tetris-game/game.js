"use strict";

const COLS = 10;
const ROWS = 20;
const CELL = 32;
const SAVE_KEY = "neonBlocksGameV1";

const PIECES = {
  I: { color: "#35dff2", matrix: [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]] },
  J: { color: "#4f74f9", matrix: [[1,0,0],[1,1,1],[0,0,0]] },
  L: { color: "#ff9f43", matrix: [[0,0,1],[1,1,1],[0,0,0]] },
  O: { color: "#ffd63d", matrix: [[1,1],[1,1]] },
  S: { color: "#55e681", matrix: [[0,1,1],[1,1,0],[0,0,0]] },
  T: { color: "#b96cff", matrix: [[0,1,0],[1,1,1],[0,0,0]] },
  Z: { color: "#ff5573", matrix: [[1,1,0],[0,1,1],[0,0,0]] }
};

const ROTATION_STATES = ["0", "R", "2", "L"];
// SRS offsets use positive y upward; the board's y axis points downward.
const SRS_KICKS_JLSTZ = {
  "0>R": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "R>2": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "2>L": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "L>0": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]]
};
const SRS_KICKS_I = {
  "0>R": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "R>2": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  "2>L": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "L>0": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]]
};

const canvas = document.querySelector("#gameCanvas");
const ctx = canvas.getContext("2d");
const nextCanvas = document.querySelector("#nextCanvas");
const nextCtx = nextCanvas.getContext("2d");
const holdCanvas = document.querySelector("#holdCanvas");
const holdCtx = holdCanvas.getContext("2d");
const overlay = document.querySelector("#overlay");
const overlayKicker = document.querySelector("#overlayKicker");
const overlayTitle = document.querySelector("#overlayTitle");
const overlayText = document.querySelector("#overlayText");
const startButton = document.querySelector("#startButton");
const soundButton = document.querySelector("#soundButton");
const aiButton = document.querySelector("#aiButton");
const aiBadge = document.querySelector("#aiBadge");
const aiStatus = document.querySelector("#aiStatus");
const scoreEl = document.querySelector("#score");
const highScoreEl = document.querySelector("#highScore");
const levelEl = document.querySelector("#level");
const linesEl = document.querySelector("#lines");
const speedEl = document.querySelector("#speed");

let board = createBoard();
let active = null;
let queue = [];
let bag = [];
let heldType = null;
let holdUsed = false;
let score = 0;
let lines = 0;
let level = 1;
let highScore = Number(localStorage.getItem("neonBlocksHighScore")) || 0;
let state = "ready";
let lastTime = 0;
let pendingSimulationMs = 0;
let backgroundTicker = null;
let lastAutosaveAt = 0;
let dropElapsed = 0;
let lockElapsed = 0;
let lockResetCount = 0;
let lockStarted = false;
let particles = [];
let flashRows = [];
let shake = 0;
let muted = false;
let audioContext = null;
let aiEnabled = false;
let aiPlan = [];
let aiElapsed = 0;
let aiRestartAt = 0;
let initialSeed = 0;
let gameplayRngState = 0;
let aiRngState = 0;

highScoreEl.textContent = formatNumber(highScore);

function createBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

function cloneMatrix(matrix) {
  return matrix.map(row => [...row]);
}

function rotateMatrix(matrix) {
  return matrix[0].map((_, i) => matrix.map(row => row[i]).reverse());
}

function pieceMatrix(type, rotation) {
  let matrix = cloneMatrix(PIECES[type].matrix);
  for (let i = 0; i < ROTATION_STATES.indexOf(rotation); i++) matrix = rotateMatrix(matrix);
  return matrix;
}

function rotationFromMatrix(type, matrix) {
  const signature = JSON.stringify(matrix);
  return ROTATION_STATES.find(rotation => JSON.stringify(pieceMatrix(type, rotation)) === signature) ?? null;
}

function generateSeed() {
  const value = new Uint32Array(1);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(value);
    return value[0];
  }
  return (Date.now() ^ Math.floor(performance.now() * 1000)) >>> 0;
}

function setSeed(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError("Seed must be an unsigned 32-bit integer");
  }
  initialSeed = seed;
  gameplayRngState = seed;
  aiRngState = (seed ^ 0x9e3779b9) >>> 0;
}

// Mulberry32: one uint32 state, with identical 32-bit arithmetic in every run.
function nextRandom(state) {
  const nextState = (state + 0x6d2b79f5) >>> 0;
  let value = nextState;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return { state: nextState, value: ((value ^ (value >>> 14)) >>> 0) / 4294967296 };
}

function gameplayRandom() {
  const result = nextRandom(gameplayRngState);
  gameplayRngState = result.state;
  return result.value;
}

function aiRandom() {
  const result = nextRandom(aiRngState);
  aiRngState = result.state;
  return result.value;
}

function shuffledBag() {
  const types = Object.keys(PIECES);
  for (let i = types.length - 1; i > 0; i--) {
    const j = Math.floor(gameplayRandom() * (i + 1));
    [types[i], types[j]] = [types[j], types[i]];
  }
  return types;
}

function fillQueue() {
  while (queue.length < 5) {
    if (!bag.length) bag = shuffledBag();
    queue.push(bag.shift());
  }
}

function makePiece(type) {
  const matrix = cloneMatrix(PIECES[type].matrix);
  return {
    type,
    matrix,
    rotation: "0",
    x: Math.floor((COLS - matrix[0].length) / 2),
    y: matrix.length === 4 ? -1 : 0
  };
}

function activatePiece(type) {
  active = makePiece(type);
  lockElapsed = 0;
  lockResetCount = 0;
  lockStarted = false;
  if (collides(active.x, active.y, active.matrix)) {
    endGame();
  } else if (aiEnabled) {
    planAI();
  }
}

function spawnPiece(resetHold = true) {
  fillQueue();
  const type = queue.shift();
  fillQueue();
  if (resetHold) holdUsed = false;
  drawNext();
  activatePiece(type);
}

function holdPiece() {
  if (state !== "playing" || aiEnabled || holdUsed) return;
  const outgoingType = active.type;
  const incomingType = heldType;
  heldType = outgoingType;
  holdUsed = true;
  dropElapsed = 0;
  drawHold();
  if (incomingType === null) spawnPiece(false);
  else activatePiece(incomingType);
}

function resetGame({ seed = generateSeed() } = {}) {
  setSeed(seed);
  board = createBoard();
  queue = [];
  bag = [];
  heldType = null;
  holdUsed = false;
  score = 0;
  lines = 0;
  level = 1;
  particles = [];
  flashRows = [];
  shake = 0;
  dropElapsed = 0;
  pendingSimulationMs = 0;
  aiPlan = [];
  aiElapsed = 0;
  aiRestartAt = 0;
  lastTime = performance.now();
  state = "playing";
  overlay.classList.add("hidden");
  drawHold();
  spawnPiece();
  updateStats();
  sound("start");
  saveGame();
}

function collides(x, y, matrix) {
  for (let row = 0; row < matrix.length; row++) {
    for (let col = 0; col < matrix[row].length; col++) {
      if (!matrix[row][col]) continue;
      const nx = x + col;
      const ny = y + row;
      if (nx < 0 || nx >= COLS || ny >= ROWS) return true;
      if (ny >= 0 && board[ny][nx]) return true;
    }
  }
  return false;
}

function mergePiece() {
  let lockedInBoard = false;
  active.matrix.forEach((row, y) => row.forEach((value, x) => {
    if (!value) return;
    const boardY = active.y + y;
    if (boardY >= 0) {
      board[boardY][active.x + x] = active.type;
      lockedInBoard = true;
    }
  }));

  if (!lockedInBoard) {
    endGame();
    return;
  }
  clearLines();
  spawnPiece();
}

function clearLines() {
  const cleared = [];
  for (let y = ROWS - 1; y >= 0; y--) {
    if (board[y].every(Boolean)) {
      cleared.push(y);
      const row = board.splice(y, 1)[0];
      board.unshift(Array(COLS).fill(null));
      createLineParticles(y, row);
      y++;
    }
  }

  if (!cleared.length) {
    sound("lock");
    return;
  }

  flashRows = cleared.map(y => ({ y, life: 180 }));
  shake = Math.min(9, 2 + cleared.length * 1.7);
  lines += cleared.length;
  const oldLevel = level;
  level = Math.floor(lines / 10) + 1;
  score += [0, 100, 300, 500, 800][cleared.length] * oldLevel;
  updateHighScore();
  updateStats();
  sound(cleared.length === 4 ? "tetris" : "clear");
}

function resetLockDelayAfterGroundedAction(wasGrounded) {
  if (!wasGrounded || lockResetCount >= 15) return;
  lockStarted = true;
  lockElapsed = 0;
  lockResetCount++;
}

function move(dx) {
  if (state !== "playing") return;
  const wasGrounded = collides(active.x, active.y + 1, active.matrix);
  if (!collides(active.x + dx, active.y, active.matrix)) {
    active.x += dx;
    resetLockDelayAfterGroundedAction(wasGrounded);
    sound("move");
  }
}

function softDrop(manual = false) {
  if (state !== "playing") return false;
  if (!collides(active.x, active.y + 1, active.matrix)) {
    active.y++;
    dropElapsed = 0;
    if (collides(active.x, active.y + 1, active.matrix)) lockStarted = true;
    if (manual) {
      score++;
      updateHighScore();
      updateStats();
    }
    return true;
  }
  return false;
}

function hardDrop() {
  if (state !== "playing") return;
  let distance = 0;
  while (!collides(active.x, active.y + 1, active.matrix)) {
    active.y++;
    distance++;
  }
  score += distance * 2;
  updateHighScore();
  updateStats();
  sound("drop");
  mergePiece();
}

function rotate() {
  if (state !== "playing") return;
  const wasGrounded = collides(active.x, active.y + 1, active.matrix);
  const from = active.rotation;
  const to = ROTATION_STATES[(ROTATION_STATES.indexOf(from) + 1) % ROTATION_STATES.length];
  if (active.type === "O") {
    active.rotation = to;
    resetLockDelayAfterGroundedAction(wasGrounded);
    return;
  }
  const rotated = pieceMatrix(active.type, to);
  const kicks = (active.type === "I" ? SRS_KICKS_I : SRS_KICKS_JLSTZ)[`${from}>${to}`];
  for (const [dx, dyUp] of kicks) {
    const x = active.x + dx;
    const y = active.y - dyUp;
    if (!collides(x, y, rotated)) {
      active.matrix = rotated;
      active.rotation = to;
      active.x = x;
      active.y = y;
      resetLockDelayAfterGroundedAction(wasGrounded);
      sound("rotate");
      return;
    }
  }
}

function ghostY() {
  let y = active.y;
  while (!collides(active.x, y + 1, active.matrix)) y++;
  return y;
}

function boardCollision(testBoard, x, y, matrix) {
  for (let row = 0; row < matrix.length; row++) {
    for (let col = 0; col < matrix[row].length; col++) {
      if (!matrix[row][col]) continue;
      const nx = x + col;
      const ny = y + row;
      if (nx < 0 || nx >= COLS || ny >= ROWS) return true;
      if (ny >= 0 && testBoard[ny][nx]) return true;
    }
  }
  return false;
}

function simulatePlacement(testBoard, matrix, x, startY, type) {
  let y = startY;
  if (boardCollision(testBoard, x, y, matrix)) return null;
  while (!boardCollision(testBoard, x, y + 1, matrix)) y++;

  const result = testBoard.map(row => [...row]);
  for (let row = 0; row < matrix.length; row++) {
    for (let col = 0; col < matrix[row].length; col++) {
      if (!matrix[row][col]) continue;
      const ny = y + row;
      if (ny < 0) return null;
      result[ny][x + col] = type;
    }
  }

  let cleared = 0;
  for (let row = ROWS - 1; row >= 0; row--) {
    if (result[row].every(Boolean)) {
      result.splice(row, 1);
      result.unshift(Array(COLS).fill(null));
      cleared++;
      row++;
    }
  }
  return { board: result, y, cleared };
}

function boardValue(testBoard, cleared = 0) {
  const heights = [];
  let holes = 0;
  let coveredHoles = 0;

  for (let x = 0; x < COLS; x++) {
    let first = ROWS;
    let blocksAbove = 0;
    for (let y = 0; y < ROWS; y++) {
      if (testBoard[y][x]) {
        if (first === ROWS) first = y;
        blocksAbove++;
      } else if (first !== ROWS) {
        holes++;
        coveredHoles += blocksAbove;
      }
    }
    heights.push(ROWS - first);
  }

  const aggregateHeight = heights.reduce((sum, height) => sum + height, 0);
  const bumpiness = heights.slice(1).reduce((sum, height, i) => sum + Math.abs(height - heights[i]), 0);
  const maxHeight = Math.max(...heights);
  let wells = 0;
  for (let x = 0; x < COLS; x++) {
    const left = x === 0 ? ROWS : heights[x - 1];
    const right = x === COLS - 1 ? ROWS : heights[x + 1];
    wells += Math.max(0, Math.min(left, right) - heights[x]);
  }

  return cleared * 8.2 - aggregateHeight * .47 - holes * 4.2 - coveredHoles * .16 - bumpiness * .28 - maxHeight * .18 - wells * .08;
}

function candidateMoves(testBoard, type, startY = 0) {
  const moves = [];
  let matrix = cloneMatrix(PIECES[type].matrix);
  const seen = new Set();

  for (let rotations = 0; rotations < 4; rotations++) {
    const signature = JSON.stringify(matrix);
    if (!seen.has(signature)) {
      seen.add(signature);
      const bounds = occupiedBounds(matrix);
      for (let x = -bounds.minX; x <= COLS - bounds.minX - bounds.width; x++) {
        const result = simulatePlacement(testBoard, matrix, x, startY, type);
        if (result) moves.push({ rotations, x, matrix: cloneMatrix(matrix), ...result });
      }
    }
    matrix = rotateMatrix(matrix);
  }
  return moves;
}

function chooseAIMove() {
  const moves = candidateMoves(board, active.type, active.y);
  let best = null;
  for (const move of moves) {
    let value = boardValue(move.board, move.cleared);
    const nextMoves = candidateMoves(move.board, queue[0], 0);
    if (nextMoves.length) {
      value += .58 * Math.max(...nextMoves.map(nextMove => boardValue(nextMove.board, nextMove.cleared)));
    }
    value += aiRandom() * .002;
    if (!best || value > best.value) best = { ...move, value };
  }
  return best;
}

function planAI() {
  if (!aiEnabled || state !== "playing" || !active) return;
  aiStatus.textContent = "正在分析";
  const moveChoice = chooseAIMove();
  if (!moveChoice) return;

  const horizontal = moveChoice.x - active.x;
  aiPlan = [
    ...Array(moveChoice.rotations).fill("rotate"),
    ...Array(Math.abs(horizontal)).fill(horizontal < 0 ? "left" : "right"),
    ...Array(Math.max(0, moveChoice.y - active.y)).fill("down"),
    "drop"
  ];
  aiElapsed = -150;
  aiStatus.textContent = `目标 ${moveChoice.x + 1} 列`;
}

function updateAI(delta) {
  if (!aiPlan.length) return;
  aiElapsed += delta;
  const action = aiPlan[0];
  const delay = action === "down" ? 24 : action === "drop" ? 85 : 72;
  if (aiElapsed < delay) return;
  aiElapsed = 0;
  aiPlan.shift();

  if (action === "left") move(-1);
  else if (action === "right") move(1);
  else if (action === "rotate") rotate();
  else if (action === "down") softDrop(true);
  else if (action === "drop") hardDrop();
}

function setAI(enabled) {
  aiEnabled = enabled;
  syncAIControls();

  if (!enabled) {
    aiPlan = [];
    aiStatus.textContent = "已关闭";
  } else if (state === "ready" || state === "over") {
    resetGame();
  } else if (state === "playing") {
    planAI();
  }
  saveGame();
}

function syncAIControls() {
  aiButton.classList.toggle("active", aiEnabled);
  aiButton.setAttribute("aria-pressed", String(aiEnabled));
  aiBadge.classList.toggle("hidden", !aiEnabled);
  aiButton.lastChild.textContent = aiEnabled ? "AI 运行中" : "AI 选手";
}

function dropInterval() {
  return Math.max(70, 900 * Math.pow(0.82, level - 1));
}

function togglePause() {
  if (state === "ready" || state === "over") return;
  if (state === "paused") {
    state = "playing";
    lastTime = performance.now();
    overlay.classList.add("hidden");
  } else {
    state = "paused";
    showOverlay("PAUSED", "已暂停", "按 P 或点击按钮继续", "继续");
  }
  saveGame();
}

function endGame() {
  state = "over";
  updateHighScore();
  sound("over");
  aiRestartAt = aiEnabled ? performance.now() + 1800 : 0;
  showOverlay("GAME OVER", "游戏结束", aiEnabled ? `AI 得分 ${formatNumber(score)} · 即将重试` : `本局得分 ${formatNumber(score)}`, "再来一局");
  saveGame();
}

function showOverlay(kicker, title, text, buttonText) {
  overlayKicker.textContent = kicker;
  overlayTitle.textContent = title;
  overlayText.textContent = text;
  startButton.textContent = buttonText;
  overlay.classList.remove("hidden");
}

function updateHighScore() {
  if (score <= highScore) return;
  highScore = score;
  localStorage.setItem("neonBlocksHighScore", String(highScore));
}

function updateStats() {
  scoreEl.textContent = formatNumber(score);
  highScoreEl.textContent = formatNumber(highScore);
  levelEl.textContent = level;
  linesEl.textContent = lines;
  speedEl.textContent = `${(900 / dropInterval()).toFixed(1)}×`;
}

function saveGame() {
  if (!active || state === "ready") return;
  try {
    localStorage.setItem(SAVE_KEY, JSON.stringify({
      version: 5,
      savedAt: Date.now(),
      initialSeed,
      gameplayRngState,
      aiRngState,
      board,
      active,
      queue,
      bag,
      heldType,
      holdUsed,
      lockElapsed,
      lockResetCount,
      lockStarted,
      score,
      lines,
      level,
      state,
      aiEnabled
    }));
    lastAutosaveAt = Date.now();
  } catch (_) {
    // Storage may be disabled; gameplay should continue normally.
  }
}

function validPieceType(type) {
  return typeof type === "string" && Object.prototype.hasOwnProperty.call(PIECES, type);
}

function validRngState(value) {
  return Number.isInteger(value) && value >= 0 && value <= 0xffffffff;
}

function validSavedGame(saved) {
  return [1, 2, 3, 4, 5].includes(saved?.version)
    && (saved.version < 4 || (validRngState(saved.initialSeed) && validRngState(saved.gameplayRngState) && validRngState(saved.aiRngState)))
    && (saved.version < 5 || (Number.isFinite(saved.lockElapsed) && saved.lockElapsed >= 0 && Number.isInteger(saved.lockResetCount) && saved.lockResetCount >= 0 && saved.lockResetCount <= 15 && typeof saved.lockStarted === "boolean"))
    && (saved.version === 1 || ((saved.heldType === null || validPieceType(saved.heldType)) && typeof saved.holdUsed === "boolean"))
    && Array.isArray(saved.board)
    && saved.board.length === ROWS
    && saved.board.every(row => Array.isArray(row) && row.length === COLS && row.every(cell => cell === null || validPieceType(cell)))
    && validPieceType(saved.active?.type)
    && Array.isArray(saved.active?.matrix)
    && saved.active.matrix.every(row => Array.isArray(row))
    && (saved.version < 3 || ROTATION_STATES.includes(saved.active.rotation))
    && rotationFromMatrix(saved.active.type, saved.active.matrix) !== null
    && (saved.version < 3 || JSON.stringify(pieceMatrix(saved.active.type, saved.active.rotation)) === JSON.stringify(saved.active.matrix))
    && Number.isInteger(saved.active.x)
    && Number.isInteger(saved.active.y)
    && Array.isArray(saved.queue)
    && saved.queue.every(validPieceType)
    && Array.isArray(saved.bag)
    && saved.bag.every(validPieceType)
    && ["playing", "paused", "over"].includes(saved.state);
}

function restoreGame() {
  try {
    const saved = JSON.parse(localStorage.getItem(SAVE_KEY));
    if (!validSavedGame(saved)) return false;

    board = saved.board.map(row => [...row]);
    active = { ...saved.active, matrix: cloneMatrix(saved.active.matrix), rotation: saved.version >= 3 ? saved.active.rotation : rotationFromMatrix(saved.active.type, saved.active.matrix) };
    queue = [...saved.queue];
    bag = [...saved.bag];
    if (saved.version >= 4) {
      initialSeed = saved.initialSeed;
      gameplayRngState = saved.gameplayRngState;
      aiRngState = saved.aiRngState;
    } else {
      setSeed(generateSeed());
    }
    heldType = saved.version >= 2 ? saved.heldType : null;
    holdUsed = saved.version >= 2 ? saved.holdUsed : false;
    fillQueue();
    drawHold();
    score = Math.max(0, Number(saved.score) || 0);
    lines = Math.max(0, Number(saved.lines) || 0);
    level = Math.floor(lines / 10) + 1;
    state = saved.state;
    aiEnabled = Boolean(saved.aiEnabled);
    particles = [];
    flashRows = [];
    aiPlan = [];
    aiElapsed = 0;
    dropElapsed = 0;
    lockElapsed = saved.version >= 5 ? saved.lockElapsed : 0;
    lockResetCount = saved.version >= 5 ? saved.lockResetCount : 0;
    lockStarted = saved.version >= 5 ? saved.lockStarted : false;
    pendingSimulationMs = 0;
    lastTime = performance.now();
    lastAutosaveAt = Date.now();
    syncAIControls();
    updateHighScore();
    updateStats();

    if (state === "playing") {
      overlay.classList.add("hidden");
      if (aiEnabled) {
        planAI();
        aiStatus.textContent = "已恢复对局";
      }
    } else if (state === "paused") {
      showOverlay("RESTORED", "对局已恢复", "刷新前处于暂停状态", "继续");
      if (aiEnabled) aiStatus.textContent = "等待继续";
    } else {
      aiRestartAt = aiEnabled ? performance.now() + 1800 : 0;
      showOverlay("GAME OVER", "游戏结束", `本局得分 ${formatNumber(score)}`, "再来一局");
    }
    return true;
  } catch (_) {
    return false;
  }
}

function formatNumber(value) {
  return value.toLocaleString("zh-CN");
}

function roundedRect(context, x, y, w, h, radius) {
  context.beginPath();
  context.roundRect(x, y, w, h, radius);
}

function drawBlock(context, x, y, size, type, alpha = 1, ghost = false) {
  const color = PIECES[type].color;
  context.save();
  context.globalAlpha = alpha;
  if (ghost) {
    context.strokeStyle = color;
    context.lineWidth = 1.5;
    roundedRect(context, x + 4, y + 4, size - 8, size - 8, 4);
    context.stroke();
    context.restore();
    return;
  }

  context.shadowColor = color;
  context.shadowBlur = size * .2;
  const gradient = context.createLinearGradient(x, y, x + size, y + size);
  gradient.addColorStop(0, lighten(color, 28));
  gradient.addColorStop(.42, color);
  gradient.addColorStop(1, darken(color, 24));
  context.fillStyle = gradient;
  roundedRect(context, x + 1.5, y + 1.5, size - 3, size - 3, Math.max(3, size * .12));
  context.fill();
  context.shadowBlur = 0;
  context.strokeStyle = "rgba(255,255,255,.32)";
  context.lineWidth = 1;
  roundedRect(context, x + 3, y + 3, size - 6, size - 6, Math.max(2, size * .09));
  context.stroke();
  context.fillStyle = "rgba(255,255,255,.18)";
  roundedRect(context, x + 5, y + 5, size - 10, Math.max(2, size * .12), 2);
  context.fill();
  context.restore();
}

function drawBoard() {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#080b18";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(130, 151, 215, .07)";
  ctx.lineWidth = 1;
  for (let x = 1; x < COLS; x++) {
    ctx.beginPath(); ctx.moveTo(x * CELL + .5, 0); ctx.lineTo(x * CELL + .5, canvas.height); ctx.stroke();
  }
  for (let y = 1; y < ROWS; y++) {
    ctx.beginPath(); ctx.moveTo(0, y * CELL + .5); ctx.lineTo(canvas.width, y * CELL + .5); ctx.stroke();
  }

  const offsetX = shake ? (Math.random() - .5) * shake : 0;
  const offsetY = shake ? (Math.random() - .5) * shake * .45 : 0;
  ctx.translate(offsetX, offsetY);

  board.forEach((row, y) => row.forEach((type, x) => {
    if (type) drawBlock(ctx, x * CELL, y * CELL, CELL, type);
  }));

  if (active && state !== "over") {
    drawMatrix(active.matrix, active.x, ghostY(), active.type, .38, true);
    drawMatrix(active.matrix, active.x, active.y, active.type);
  }

  flashRows.forEach(row => {
    ctx.fillStyle = `rgba(255,255,255,${Math.min(.8, row.life / 180)})`;
    ctx.fillRect(0, row.y * CELL, canvas.width, CELL);
  });

  drawParticles();
  ctx.restore();
}

function drawMatrix(matrix, px, py, type, alpha = 1, ghost = false) {
  matrix.forEach((row, y) => row.forEach((value, x) => {
    if (value && py + y >= 0) drawBlock(ctx, (px + x) * CELL, (py + y) * CELL, CELL, type, alpha, ghost);
  }));
}

function drawNext() {
  nextCtx.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
  queue.slice(0, 3).forEach((type, index) => {
    const matrix = PIECES[type].matrix;
    const occupied = occupiedBounds(matrix);
    const size = index === 0 ? 24 : 19;
    const sectionY = index === 0 ? 9 : 84 + (index - 1) * 66;
    const xOffset = (nextCanvas.width - occupied.width * size) / 2 - occupied.minX * size;
    const yOffset = sectionY + (index === 0 ? (58 - occupied.height * size) / 2 : (48 - occupied.height * size) / 2) - occupied.minY * size;
    matrix.forEach((row, y) => row.forEach((value, x) => {
      if (value) drawBlock(nextCtx, xOffset + x * size, yOffset + y * size, size, type, index === 0 ? 1 : .68);
    }));
    if (index === 0) {
      nextCtx.strokeStyle = "rgba(137,159,255,.12)";
      nextCtx.beginPath(); nextCtx.moveTo(14, 79.5); nextCtx.lineTo(nextCanvas.width - 14, 79.5); nextCtx.stroke();
    }
  });
}

function drawHold() {
  holdCtx.clearRect(0, 0, holdCanvas.width, holdCanvas.height);
  if (heldType === null) return;
  const matrix = PIECES[heldType].matrix;
  const bounds = occupiedBounds(matrix);
  const size = 24;
  const xOffset = (holdCanvas.width - bounds.width * size) / 2 - bounds.minX * size;
  const yOffset = (holdCanvas.height - bounds.height * size) / 2 - bounds.minY * size;
  matrix.forEach((row, y) => row.forEach((value, x) => {
    if (value) drawBlock(holdCtx, xOffset + x * size, yOffset + y * size, size, heldType);
  }));
}

function occupiedBounds(matrix) {
  const cells = [];
  matrix.forEach((row, y) => row.forEach((value, x) => { if (value) cells.push({ x, y }); }));
  const xs = cells.map(cell => cell.x);
  const ys = cells.map(cell => cell.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return { minX, minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

function createLineParticles(rowY, row) {
  row.forEach((type, x) => {
    for (let i = 0; i < 3; i++) {
      particles.push({
        x: x * CELL + CELL / 2,
        y: rowY * CELL + CELL / 2,
        vx: (Math.random() - .5) * 4.5,
        vy: (Math.random() - .75) * 4,
        life: 420 + Math.random() * 220,
        maxLife: 640,
        color: PIECES[type].color
      });
    }
  });
}

function updateEffects(delta) {
  shake = Math.max(0, shake - delta * .025);
  flashRows.forEach(row => row.life -= delta);
  flashRows = flashRows.filter(row => row.life > 0);
  particles.forEach(p => {
    p.x += p.vx * delta / 16;
    p.y += p.vy * delta / 16;
    p.vy += .12 * delta / 16;
    p.life -= delta;
  });
  particles = particles.filter(p => p.life > 0);
}

function drawParticles() {
  particles.forEach(p => {
    ctx.globalAlpha = Math.max(0, p.life / p.maxLife);
    ctx.fillStyle = p.color;
    ctx.fillRect(p.x - 2, p.y - 2, 4, 4);
  });
  ctx.globalAlpha = 1;
}

function lighten(hex, amount) { return adjustColor(hex, amount); }
function darken(hex, amount) { return adjustColor(hex, -amount); }
function adjustColor(hex, amount) {
  const value = parseInt(hex.slice(1), 16);
  const r = Math.max(0, Math.min(255, (value >> 16) + amount));
  const g = Math.max(0, Math.min(255, ((value >> 8) & 255) + amount));
  const b = Math.max(0, Math.min(255, (value & 255) + amount));
  return `rgb(${r},${g},${b})`;
}

function sound(kind) {
  if (muted) return;
  try {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === "suspended") audioContext.resume();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const settings = {
      move: [150, .018, "square"], rotate: [240, .035, "square"], lock: [105, .025, "triangle"],
      drop: [80, .06, "square"], clear: [520, .11, "sine"], tetris: [760, .18, "sine"],
      start: [360, .1, "sine"], over: [70, .28, "sawtooth"]
    }[kind];
    const now = audioContext.currentTime;
    oscillator.type = settings[2];
    oscillator.frequency.setValueAtTime(settings[0], now);
    oscillator.frequency.exponentialRampToValueAtTime(Math.max(35, settings[0] * (kind === "over" ? .45 : 1.35)), now + settings[1]);
    gain.gain.setValueAtTime(.035, now);
    gain.gain.exponentialRampToValueAtTime(.001, now + settings[1]);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + settings[1]);
  } catch (_) {
    muted = true;
    soundButton.classList.add("muted");
  }
}

function advanceGame(delta, time) {
  updateEffects(delta);

  if (state === "playing" && active) {
    if (aiEnabled) {
      updateAI(delta);
    } else {
      dropElapsed += delta;
      if (dropElapsed >= dropInterval()) softDrop(false);
    }

    const grounded = collides(active.x, active.y + 1, active.matrix);
    if (grounded) lockStarted = true;
    // Keep the clock running after a floor kick, so the reset cap cannot be bypassed.
    if (lockStarted) {
      lockElapsed += delta;
      if (grounded && lockElapsed >= 450) mergePiece();
    }
  }

  if (state === "over" && aiEnabled && aiRestartAt && time >= aiRestartAt) resetGame();

  if (state !== "ready" && Date.now() - lastAutosaveAt >= 500) saveGame();
}

function updateSimulationClock(time = performance.now()) {
  if (!lastTime) lastTime = time;
  const elapsed = Math.max(0, time - lastTime);
  lastTime = time;

  // Background tabs may receive timer events only once every few seconds.
  // Preserve real elapsed time and replay it in stable 50 ms simulation steps.
  pendingSimulationMs += Math.min(elapsed, 60 * 60 * 1000);
  let steps = 0;
  while (pendingSimulationMs > 0 && steps < 1200) {
    const delta = Math.min(50, pendingSimulationMs);
    advanceGame(delta, time);
    pendingSimulationMs = Math.max(0, pendingSimulationMs - delta);
    steps++;
  }
}

function gameLoop(time = 0) {
  if (!document.hidden) updateSimulationClock(time);

  drawBoard();
  requestAnimationFrame(gameLoop);
}

function startBackgroundTicker() {
  const tick = () => {
    if (document.hidden) updateSimulationClock(performance.now());
  };

  if (typeof Worker === "undefined" || typeof Blob === "undefined") {
    if (typeof setInterval === "function") setInterval(tick, 50);
    return;
  }

  try {
    const workerUrl = URL.createObjectURL(new Blob([
      "setInterval(() => postMessage(Date.now()), 50);"
    ], { type: "text/javascript" }));
    backgroundTicker = new Worker(workerUrl);
    URL.revokeObjectURL(workerUrl);
    backgroundTicker.onmessage = tick;
  } catch (_) {
    if (typeof setInterval === "function") setInterval(tick, 50);
  }
}

function handleAction(action) {
  if (aiEnabled && action !== "pause") return;
  const actions = { left: () => move(-1), right: () => move(1), down: () => softDrop(true), rotate, drop: hardDrop, hold: holdPiece, pause: togglePause };
  actions[action]?.();
  saveGame();
}

document.addEventListener("keydown", event => {
  const keyMap = { ArrowLeft: "left", ArrowRight: "right", ArrowDown: "down", ArrowUp: "rotate", x: "rotate", X: "rotate", c: "hold", C: "hold", " ": "drop", p: "pause", P: "pause", Escape: "pause" };
  const action = keyMap[event.key];
  if (!action) return;
  event.preventDefault();
  if (event.repeat && ["rotate", "drop", "hold", "pause"].includes(action)) return;
  handleAction(action);
});

document.querySelectorAll("[data-action]").forEach(button => {
  button.addEventListener("pointerdown", event => {
    event.preventDefault();
    handleAction(button.dataset.action);
  });
});

startButton.addEventListener("click", () => state === "paused" ? togglePause() : resetGame());
aiButton.addEventListener("click", () => setAI(!aiEnabled));
soundButton.addEventListener("click", () => {
  muted = !muted;
  soundButton.classList.toggle("muted", muted);
  soundButton.textContent = muted ? "×" : "♪";
});

document.addEventListener("visibilitychange", () => {
  updateSimulationClock(performance.now());
  if (aiEnabled && state === "playing") aiStatus.textContent = document.hidden ? "后台运行" : "正在运行";
  saveGame();
});

window.addEventListener("pagehide", saveGame);

const restoredGame = restoreGame();
if (!restoredGame) {
  setSeed(generateSeed());
  fillQueue();
}
drawNext();
drawHold();
drawBoard();
requestAnimationFrame(gameLoop);
startBackgroundTicker();
if (!restoredGame && new URLSearchParams(location.search).get("ai") === "1") setAI(true);
