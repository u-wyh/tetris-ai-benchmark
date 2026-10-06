"use strict";

const ACTION_COUNT = 1840;
const PLACEMENT_TICK_MS = 50;
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

function occupiedCells(piece) {
  const cells = [];
  piece.matrix.forEach((row, dy) => row.forEach((value, dx) => {
    if (value) cells.push({ x: piece.x + dx, y: piece.y + dy });
  }));
  return cells.sort((a, b) => a.y - b.y || a.x - b.x);
}

function placementPath(node, hold, finalStep) {
  const steps = [];
  for (let current = node; current.parent; current = current.parent) steps.push(current.step);
  steps.reverse();
  if (hold) steps.unshift("Hold");
  if (finalStep) steps.push(finalStep);
  return steps;
}

function searchPlacementBranch(observation, hold) {
  const testBoard = observation.board;
  const piece = hold
    ? makePiece(observation.hold ?? observation.next[0])
    : observation.currentPiece;
  if (!piece || collidesOnBoard(testBoard, piece.x, piece.y, piece.matrix)) return [];

  const type = piece.type;
  const matrices = Object.fromEntries(ROTATION_STATES.map(rotation => [rotation, pieceMatrix(type, rotation)]));
  const first = {
    x: piece.x, y: piece.y, rotation: piece.rotation,
    lockElapsed: hold ? 0 : observation.lock.elapsedMs,
    lockResetCount: hold ? 0 : observation.lock.resetCount,
    lockStarted: hold ? false : observation.lock.started,
    parent: null, step: null, depth: 0
  };
  const frontier = [first];
  const visited = new Set();
  const byCells = new Map();

  function matrixOf(node) { return matrices[node.rotation]; }
  function isGrounded(node) {
    return collidesOnBoard(testBoard, node.x, node.y + 1, matrixOf(node));
  }
  function keyOf(node, grounded) {
    return `${node.x},${node.y},${node.rotation},${grounded ? 1 : 0},${node.lockStarted ? 1 : 0},${node.lockResetCount},${node.lockElapsed}`;
  }
  function addPlacement(node, finalStep = null) {
    const cells = occupiedCells({ x: node.x, y: node.y, matrix: matrixOf(node) });
    const x = Math.min(...cells.map(cell => cell.x));
    const y = Math.min(...cells.map(cell => cell.y));
    if (x < 0 || x >= COLS || y < PLACEMENT_MIN_Y || y > PLACEMENT_MAX_Y) return;
    const rotation = ROTATION_STATES.indexOf(node.rotation);
    const actionId = encodeAction(hold, rotation, x, y);
    const cellKey = cells.map(cell => `${cell.x}:${cell.y}`).join("|");
    const pathLength = node.depth + hold + (finalStep ? 1 : 0);
    const previous = byCells.get(cellKey);
    if (previous && (previous.actionId < actionId
      || (previous.actionId === actionId && previous.path.length <= pathLength))) return;
    byCells.set(cellKey, { actionId, hold, rotation, x, y, occupiedCells: cells,
      path: placementPath(node, hold, finalStep) });
  }
  function enqueue(node) {
    const top = occupiedCells({ x: node.x, y: node.y, matrix: matrixOf(node) })[0].y;
    if (top < PLACEMENT_MIN_Y - 3 || top > PLACEMENT_MAX_Y) return;
    const grounded = isGrounded(node);
    if (grounded && node.lockStarted && node.lockElapsed >= LOCK_DELAY_MS) {
      addPlacement(node);
      return;
    }
    const key = keyOf(node, grounded);
    if (visited.has(key)) return;
    visited.add(key);
    frontier.push(node);
  }

  visited.add(keyOf(first, isGrounded(first)));
  for (let head = 0; head < frontier.length; head++) {
    const node = frontier[head];
    const matrix = matrixOf(node);
    const grounded = isGrounded(node);
    if (grounded && node.lockStarted && node.lockElapsed >= LOCK_DELAY_MS) {
      addPlacement(node);
      continue;
    }

    let dropY = node.y;
    while (!collidesOnBoard(testBoard, node.x, dropY + 1, matrix)) dropY++;
    addPlacement({ ...node, y: dropY }, "HardDrop");

    for (const step of PLACEMENT_STEPS) {
      let x = node.x;
      let y = node.y;
      let rotation = node.rotation;
      if (step === "Left" || step === "Right") {
        x += step === "Left" ? -1 : 1;
        if (collidesOnBoard(testBoard, x, y, matrix)) continue;
      } else if (step === "RotateCW") {
        const result = tryRotatePiece({ type, matrix, x, y, rotation }, testBoard);
        if (!result) continue;
        ({ x, y, rotation } = result);
      } else {
        y++;
        if (collidesOnBoard(testBoard, x, y, matrix)) continue;
      }

      let lockElapsed = node.lockElapsed;
      let lockResetCount = node.lockResetCount;
      let lockStarted = node.lockStarted;
      if (grounded && step !== "Down" && lockResetCount < MAX_LOCK_RESETS) {
        lockElapsed = 0;
        lockResetCount++;
        lockStarted = true;
      }
      const successor = { x, y, rotation, lockElapsed, lockResetCount, lockStarted,
        parent: node, step, depth: node.depth + 1 };
      if (isGrounded(successor)) successor.lockStarted = true;
      if (successor.lockStarted) {
        successor.lockElapsed = Math.min(LOCK_DELAY_MS, successor.lockElapsed + PLACEMENT_TICK_MS);
      }
      enqueue(successor);
    }
  }
  return [...byCells.values()].sort((a, b) => a.actionId - b.actionId);
}

function getLegalPlacements() {
  const observation = getPublicObservation();
  if (!observation.currentPiece || observation.phase !== "playing") return [];
  const placements = searchPlacementBranch(observation, 0);
  if (observation.holdAvailable && (observation.hold || observation.next[0])) {
    placements.push(...searchPlacementBranch(observation, 1));
  }
  return placements.sort((a, b) => a.actionId - b.actionId);
}

function getActionMask() {
  const mask = Array(ACTION_COUNT).fill(false);
  for (const placement of getLegalPlacements()) mask[placement.actionId] = true;
  return mask;
}
