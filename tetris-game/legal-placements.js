"use strict";

const ACTION_COUNT = 1840;
const PLACEMENT_MIN_Y = -3;
const PLACEMENT_MAX_Y = 19;
// BFS expansion order is part of the canonical path definition.
const PLACEMENT_STEPS = ["Left", "Right", "RotateCW", "Down"];

function encodeAction(hold, rotation, x, y) {
  if (typeof hold === "object" && hold !== null) ({ hold, rotation, x, y } = hold);
  if (![hold, rotation, x, y].every(Number.isInteger)
    || hold < 0 || hold > 1 || rotation < 0 || rotation > 3
    || x < 0 || x >= COLS || y < PLACEMENT_MIN_Y || y > PLACEMENT_MAX_Y) {
    throw new RangeError("Action coordinates are outside the 1840-action space");
  }
  return (((hold * 4 + rotation) * COLS + x) * 23 + y + 3);
}

function decodeAction(action) {
  if (!Number.isInteger(action) || action < 0 || action >= ACTION_COUNT) {
    throw new RangeError("Action ID must be an integer from 0 to 1839");
  }
  const y = action % 23 - 3;
  let rest = Math.floor(action / 23);
  const x = rest % COLS;
  rest = Math.floor(rest / COLS);
  return { hold: Math.floor(rest / 4), rotation: rest % 4, x, y };
}

function placementPath(node, hold, finalStep) {
  const steps = [];
  for (let current = node; current.parent; current = current.parent) steps.push(current.step);
  steps.reverse();
  if (hold) steps.unshift("Hold");
  if (finalStep) steps.push(finalStep);
  return steps;
}

function searchPlacementBranch(observation, hold, stats) {
  const testBoard = observation.board;
  const piece = hold
    ? makePiece(observation.hold ?? observation.next[0])
    : observation.currentPiece;
  if (!piece || collidesOnBoard(testBoard, piece.x, piece.y, piece.matrix)) return [];

  const type = piece.type;
  const matrices = ROTATION_STATES.map(rotation => pieceMatrix(type, rotation));
  const shapes = matrices.map(matrix => {
    const cells = [];
    matrix.forEach((row, dy) => row.forEach((value, dx) => {
      if (value) cells.push([dx, dy]);
    }));
    return { cells, minX: Math.min(...cells.map(cell => cell[0])),
      minY: Math.min(...cells.map(cell => cell[1])) };
  });
  const first = { x: piece.x, y: piece.y,
    rotation: ROTATION_STATES.indexOf(piece.rotation), parent: null, step: null };
  const frontier = [first];
  const visited = new Set([stateKey(first.x, first.y, first.rotation)]);
  const byCells = new Map();

  function stateKey(x, y, rotation) {
    // The matrix origin remains near the 10x20 board; these ranges never overlap.
    return ((y + 8) * 32 + x + 8) * 4 + rotation;
  }
  function collidesAt(x, y, rotation) {
    for (const [dx, dy] of shapes[rotation].cells) {
      const column = x + dx;
      const row = y + dy;
      if (column < 0 || column >= COLS || row >= ROWS || (row >= 0 && testBoard[row][column])) return true;
    }
    return false;
  }
  function addPlacement(node) {
    const shape = shapes[node.rotation];
    const x = node.x + shape.minX;
    const y = node.y + shape.minY;
    if (x < 0 || x >= COLS || y < PLACEMENT_MIN_Y || y > PLACEMENT_MAX_Y) return;
    const cells = shape.cells.map(([dx, dy]) => ({ x: node.x + dx, y: node.y + dy }))
      .sort((a, b) => a.y - b.y || a.x - b.x);
    const actionId = encodeAction(hold, node.rotation, x, y);
    const cellKey = cells.map(cell => cell.y * COLS + cell.x).join(",");
    const previous = byCells.get(cellKey);
    if (previous && previous.actionId <= actionId) return;
    byCells.set(cellKey, { actionId, hold, rotation: node.rotation, x, y, occupiedCells: cells,
      path: placementPath(node, hold, "HardDrop") });
  }
  function enqueue(x, y, rotation, parent, step) {
    const top = y + shapes[rotation].minY;
    if (top < PLACEMENT_MIN_Y - 3 || top > PLACEMENT_MAX_Y) return;
    const key = stateKey(x, y, rotation);
    if (visited.has(key)) return;
    visited.add(key);
    frontier.push({ x, y, rotation, parent, step });
  }

  for (let head = 0; head < frontier.length; head++) {
    const node = frontier[head];
    if (collidesAt(node.x, node.y + 1, node.rotation)) addPlacement(node);

    for (const step of PLACEMENT_STEPS) {
      let x = node.x;
      let y = node.y;
      let rotation = node.rotation;
      if (step === "Left" || step === "Right") {
        x += step === "Left" ? -1 : 1;
        if (collidesAt(x, y, rotation)) continue;
      } else if (step === "RotateCW") {
        const result = tryRotatePiece({ type, matrix: matrices[rotation], x, y,
          rotation: ROTATION_STATES[rotation] }, testBoard);
        if (!result) continue;
        x = result.x;
        y = result.y;
        rotation = (rotation + 1) % 4;
      } else {
        y++;
        if (collidesAt(x, y, rotation)) continue;
      }
      enqueue(x, y, rotation, node, step);
    }
  }
  if (stats) stats.states += visited.size;
  return [...byCells.values()].sort((a, b) => a.actionId - b.actionId);
}

function getLegalPlacements(stats = null) {
  if (stats) stats.states = 0;
  const observation = getPublicObservation();
  if (!observation.currentPiece || observation.phase !== "playing") return [];
  const placements = searchPlacementBranch(observation, 0, stats);
  if (observation.holdAvailable && (observation.hold || observation.next[0])) {
    placements.push(...searchPlacementBranch(observation, 1, stats));
  }
  return placements.sort((a, b) => a.actionId - b.actionId);
}

function getActionMask(stats = null) {
  const mask = Array(ACTION_COUNT).fill(false);
  for (const placement of getLegalPlacements(stats)) mask[placement.actionId] = true;
  return mask;
}
