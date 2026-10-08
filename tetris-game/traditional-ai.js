"use strict";

// Mirrors training/traditional/{v1,v2}.py. Input is strictly a public observation.
const TRADITIONAL_V2_WEIGHTS = Object.freeze({
  cleared: 8.2, aggregate_height: -0.47, holes: -4.2,
  covered_holes: -0.16, bumpiness: -0.28, max_height: -0.18,
  wells: -0.08, row_transitions: -0.14,
  column_transitions: -0.22, top_risk: -0.9
});
const TRADITIONAL_V2_BEAM_WIDTH = 8;
const TRADITIONAL_V2_LOOKAHEAD = 0.58;
const TRADITIONAL_V2_DEATH_PENALTY = 1000000;

function traditionalPlacements(observation) {
  if (!observation.currentPiece || observation.phase !== "playing") return [];
  const result = searchPlacementBranch(observation, 0, null);
  if (observation.holdAvailable && (observation.hold || observation.next[0])) {
    result.push(...searchPlacementBranch(observation, 1, null));
  }
  return result.sort((a, b) => a.actionId - b.actionId);
}

function traditionalFeatures(board) {
  const heights = [];
  let holes = 0, covered = 0;
  for (let x = 0; x < 10; x++) {
    let first = 20, above = 0;
    for (let y = 0; y < 20; y++) {
      if (board[y][x]) { first = Math.min(first, y); above++; }
      else if (first !== 20) { holes++; covered += above; }
    }
    heights.push(20 - first);
  }
  let wells = 0;
  for (let x = 0; x < 10; x++) {
    wells += Math.max(0, Math.min(x ? heights[x - 1] : 20,
      x < 9 ? heights[x + 1] : 20) - heights[x]);
  }
  let rowTransitions = 0;
  for (const row of board) {
    let previous = true;
    for (const cell of row) {
      const occupied = Boolean(cell);
      rowTransitions += Number(previous !== occupied);
      previous = occupied;
    }
    rowTransitions += Number(!previous);
  }
  let columnTransitions = 0;
  for (let x = 0; x < 10; x++) {
    let previous = false;
    for (let y = 0; y < 20; y++) {
      const occupied = Boolean(board[y][x]);
      columnTransitions += Number(previous !== occupied);
      previous = occupied;
    }
    columnTransitions += Number(!previous);
  }
  const maximum = Math.max(...heights);
  return { aggregate_height: heights.reduce((a, b) => a + b, 0), holes,
    covered_holes: covered,
    bumpiness: heights.slice(1).reduce((sum, h, i) => sum + Math.abs(h - heights[i]), 0),
    max_height: maximum, wells, row_transitions: rowTransitions,
    column_transitions: columnTransitions, top_risk: Math.max(0, maximum - 13) ** 2 };
}

function traditionalBoardScore(board, cleared, weights) {
  const values = traditionalFeatures(board);
  values.cleared = cleared;
  return Object.keys(values).reduce((sum, name) => sum + weights[name] * values[name], 0);
}

function traditionalDropDistance(observation, placement) {
  const piece = placement.hold
    ? makePiece(observation.hold || observation.next[0]) : observation.currentPiece;
  let x = piece.x, y = piece.y;
  let rotation = ROTATION_STATES.indexOf(piece.rotation);
  for (const step of placement.path) {
    if (step === "Left") x--;
    else if (step === "Right") x++;
    else if (step === "Down") y++;
    else if (step === "RotateCW") {
      const result = tryRotatePiece({ type: piece.type,
        matrix: pieceMatrix(piece.type, ROTATION_STATES[rotation]),
        x, y, rotation: ROTATION_STATES[rotation] }, observation.board);
      if (!result) throw new Error("Legal path has failed rotation");
      x = result.x; y = result.y;
      rotation = (rotation + 1) % 4;
    }
  }
  let distance = 0;
  const matrix = pieceMatrix(piece.type, ROTATION_STATES[rotation]);
  while (!collidesOnBoard(observation.board, x, y + 1, matrix)) { y++; distance++; }
  return distance;
}

function traditionalSimulate(observation, placement) {
  const visible = [...observation.next];
  let held = observation.hold;
  const current = observation.currentPiece.type;
  let type = current;
  if (placement.hold) {
    if (!observation.holdAvailable) throw new Error("Hold is unavailable");
    type = held || visible.shift();
    held = current;
  }
  const board = observation.board.map(row => [...row]);
  let score = observation.score + traditionalDropDistance(observation, placement) * 2;
  let lines = observation.lines, level = observation.level;
  if (!placement.occupiedCells.some(cell => cell.y >= 0)) {
    return [{ board, currentPiece: null, hold: held, holdAvailable: false,
      next: visible, score, lines, level, phase: "over", gameOver: true }, 0];
  }
  for (const cell of placement.occupiedCells) if (cell.y >= 0) board[cell.y][cell.x] = type;
  const remaining = board.filter(row => !row.every(Boolean));
  const cleared = 20 - remaining.length;
  const result = Array.from({ length: cleared }, () => Array(10).fill(null)).concat(remaining);
  if (cleared) {
    lines += cleared;
    score += [0, 100, 300, 500, 800][cleared] * level;
    level = Math.floor(lines / 10) + 1;
  }
  let following = null, phase = "unknown";
  if (visible.length) {
    following = makePiece(visible.shift());
    phase = collidesOnBoard(result, following.x, following.y, following.matrix) ? "over" : "playing";
    if (phase === "over") following = null;
  }
  return [{ board: result, currentPiece: following, hold: held,
    holdAvailable: phase === "playing", next: visible,
    score, lines, level, phase, gameOver: phase === "over" }, cleared];
}

function traditionalStateKey(state) {
  return JSON.stringify([state.board.map(row => row.map(Boolean)),
    state.currentPiece?.type || null, state.hold, state.next, state.phase]);
}

function traditionalChooseV2(observation, mode = "hold", placements = null) {
  if (!["hold", "beam"].includes(mode)) throw new Error("Unknown V2 mode");
  const root = placements || traditionalPlacements(observation);
  if (!root.length) throw new Error("No legal placement");
  const unique = new Map();
  for (const placement of root) {
    const [state, cleared] = traditionalSimulate(observation, placement);
    const value = traditionalBoardScore(state.board, cleared, TRADITIONAL_V2_WEIGHTS)
      - (state.gameOver ? TRADITIONAL_V2_DEATH_PENALTY : 0);
    const key = traditionalStateKey(state);
    const old = unique.get(key);
    if (!old || value > old.value || (value === old.value && placement.actionId < old.action)) {
      unique.set(key, { value, action: placement.actionId, state });
    }
  }
  const ranked = [...unique.values()].sort((a, b) => b.value - a.value || a.action - b.action);
  if (mode === "hold") return ranked[0].action;
  let bestScore = -Infinity, bestAction = null;
  const cache = new Map();
  for (const item of ranked.slice(0, TRADITIONAL_V2_BEAM_WIDTH)) {
    let future = 0;
    if (item.state.phase === "playing") {
      const key = traditionalStateKey(item.state);
      if (!cache.has(key)) cache.set(key, traditionalPlacements(item.state));
      const second = cache.get(key);
      if (second.length) {
        let secondBest = -Infinity;
        for (const placement of second) {
          const [state, cleared] = traditionalSimulate(item.state, placement);
          const score = traditionalBoardScore(state.board, cleared, TRADITIONAL_V2_WEIGHTS)
            - (state.gameOver ? TRADITIONAL_V2_DEATH_PENALTY : 0);
          secondBest = Math.max(secondBest, score);
        }
        future = TRADITIONAL_V2_LOOKAHEAD * secondBest;
      } else future = -TRADITIONAL_V2_DEATH_PENALTY;
    }
    const score = item.value + future;
    if (score > bestScore || (score === bestScore && item.action < bestAction)) {
      bestScore = score; bestAction = item.action;
    }
  }
  return bestAction;
}

function traditionalV1BoardValue(board, cleared) {
  const f = traditionalFeatures(board);
  return cleared * 8.2 - f.aggregate_height * 0.47 - f.holes * 4.2
    - f.covered_holes * 0.16 - f.bumpiness * 0.28 - f.max_height * 0.18 - f.wells * 0.08;
}

function traditionalV1Simulate(board, placement) {
  if (placement.occupiedCells.some(cell => cell.y < 0)) return null;
  const result = board.map(row => [...row]);
  for (const cell of placement.occupiedCells) result[cell.y][cell.x] = "T";
  const remaining = result.filter(row => !row.every(Boolean));
  const cleared = 20 - remaining.length;
  return [Array.from({ length: cleared }, () => Array(10).fill(null)).concat(remaining), cleared];
}

function traditionalV1Random(state) {
  const nextState = (state + 0x6d2b79f5) >>> 0;
  let value = nextState;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return { state: nextState, value: ((value ^ (value >>> 14)) >>> 0) / 4294967296 };
}

function traditionalChooseV1(observation, rngState, placements = null) {
  const normal = (placements || traditionalPlacements(observation)).filter(p => !p.hold);
  if (!normal.length) throw new Error("No legal non-Hold placement");
  let best = normal[0].actionId, bestValue = -Infinity;
  for (const placement of normal) {
    const simulation = traditionalV1Simulate(observation.board, placement);
    if (!simulation) continue;
    const [board, cleared] = simulation;
    let value = traditionalV1BoardValue(board, cleared);
    const following = { board, currentPiece: makePiece(observation.next[0]), hold: null,
      holdAvailable: false, next: [], phase: "playing" };
    const nextValues = [];
    for (const candidate of traditionalPlacements(following)) {
      const result = traditionalV1Simulate(board, candidate);
      if (result) nextValues.push(traditionalV1BoardValue(...result));
    }
    if (nextValues.length) value += 0.58 * Math.max(...nextValues);
    const random = traditionalV1Random(rngState);
    rngState = random.state;
    value += random.value * 0.002;
    if (value > bestValue) { bestValue = value; best = placement.actionId; }
  }
  return { actionId: best, rngState };
}
