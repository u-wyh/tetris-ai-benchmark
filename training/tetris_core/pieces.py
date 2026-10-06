"""Benchmark v1.0 piece shapes and clockwise SRS kicks, copied from game.js."""

COLS, ROWS = 10, 20
TYPES = ("I", "J", "L", "O", "S", "T", "Z")
ROTATIONS = ("0", "R", "2", "L")
BASE = {
    "I": ((0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 0, 0), (0, 0, 0, 0)),
    "J": ((1, 0, 0), (1, 1, 1), (0, 0, 0)),
    "L": ((0, 0, 1), (1, 1, 1), (0, 0, 0)),
    "O": ((1, 1), (1, 1)),
    "S": ((0, 1, 1), (1, 1, 0), (0, 0, 0)),
    "T": ((0, 1, 0), (1, 1, 1), (0, 0, 0)),
    "Z": ((1, 1, 0), (0, 1, 1), (0, 0, 0)),
}
JLSTZ_KICKS = (
    ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
)
I_KICKS = (
    ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
)


def rotate_matrix(matrix):
    return tuple(tuple(row[i] for row in matrix[::-1]) for i in range(len(matrix[0])))


MATRICES = {}
CELLS = {}
for _type, _base in BASE.items():
    _matrix = _base
    for _rotation in range(4):
        MATRICES[_type, _rotation] = _matrix
        CELLS[_type, _rotation] = tuple(
            (x, y) for y, row in enumerate(_matrix) for x, value in enumerate(row) if value
        )
        _matrix = rotate_matrix(_matrix)


def make_piece(piece_type):
    matrix = MATRICES[piece_type, 0]
    return {"type": piece_type, "rotation": "0", "x": (COLS - len(matrix[0])) // 2,
            "y": -1 if len(matrix) == 4 else 0, "matrix": [list(row) for row in matrix]}


def collides(board, piece_type, rotation, x, y):
    for dx, dy in CELLS[piece_type, rotation]:
        column, row = x + dx, y + dy
        if column < 0 or column >= COLS or row >= ROWS or (row >= 0 and board[row][column]):
            return True
    return False


def rotate_cw(board, piece_type, rotation, x, y):
    target = (rotation + 1) % 4
    if piece_type == "O":
        return x, y, target
    kicks = I_KICKS if piece_type == "I" else JLSTZ_KICKS
    for dx, dy_up in kicks[rotation]:
        nx, ny = x + dx, y - dy_up
        if not collides(board, piece_type, target, nx, ny):
            return nx, ny, target
    return None
