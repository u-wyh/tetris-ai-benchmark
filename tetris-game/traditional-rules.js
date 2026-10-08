"use strict";
// Geometry-only worker counterpart of game.js; no queue, Bag or RNG access.
const COLS = 10;
const ROWS = 20;
const PIECES = {
  I: { matrix: [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]] },
  J: { matrix: [[1,0,0],[1,1,1],[0,0,0]] },
  L: { matrix: [[0,0,1],[1,1,1],[0,0,0]] },
  O: { matrix: [[1,1],[1,1]] },
  S: { matrix: [[0,1,1],[1,1,0],[0,0,0]] },
  T: { matrix: [[0,1,0],[1,1,1],[0,0,0]] },
  Z: { matrix: [[1,1,0],[0,1,1],[0,0,0]] }
};
const ROTATION_STATES = ["0", "R", "2", "L"];
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
function pieceMatrix(type, rotation) {
  let matrix = PIECES[type].matrix.map(row => [...row]);
  for (let i = 0; i < ROTATION_STATES.indexOf(rotation); i++) {
    matrix = matrix[0].map((_, x) => matrix.map(row => row[x]).reverse());
  }
  return matrix;
}
function makePiece(type) {
  const matrix = pieceMatrix(type, "0");
  return { type, matrix, rotation: "0", x: Math.floor((COLS - matrix[0].length) / 2),
    y: matrix.length === 4 ? -1 : 0 };
}
function collidesOnBoard(board, x, y, matrix) {
  for (let dy = 0; dy < matrix.length; dy++) for (let dx = 0; dx < matrix[dy].length; dx++) {
    if (!matrix[dy][dx]) continue;
    const nx = x + dx, ny = y + dy;
    if (nx < 0 || nx >= COLS || ny >= ROWS || (ny >= 0 && board[ny][nx])) return true;
  }
  return false;
}
function tryRotatePiece(piece, board) {
  const from = piece.rotation;
  const to = ROTATION_STATES[(ROTATION_STATES.indexOf(from) + 1) % 4];
  if (piece.type === "O") return { ...piece, rotation: to };
  const matrix = pieceMatrix(piece.type, to);
  const kicks = (piece.type === "I" ? SRS_KICKS_I : SRS_KICKS_JLSTZ)[`${from}>${to}`];
  for (const [dx, dyUp] of kicks) {
    const x = piece.x + dx, y = piece.y - dyUp;
    if (!collidesOnBoard(board, x, y, matrix)) return { ...piece, matrix, rotation: to, x, y };
  }
  return null;
}
function getPublicObservation() { throw new Error("Worker requires explicit public observation"); }
