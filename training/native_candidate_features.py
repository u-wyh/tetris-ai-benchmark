"""Explicitly loaded, hash-pinned C prototype; not a production training backend."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from training.candidate_features import FEATURE_NAMES, FEATURE_SCALE, PIECES
from training.evaluation.fast_candidate_features import candidate_features as reference
from training.tetris_core import ACTION_COUNT

ABI = 'tetris-features-v1:u16[20]:i64[n,13]:i32[16]:u8[1840,16]:u8[1840]'
FLAGS = ['-std=c11', '-O3', '-fPIC', '-shared', '-Wall', '-Wextra', '-Werror']


class NativeCandidateFeatures:
    """Hash and ABI check at explicit load; immutable artifact path required.

    The manifest records compiler identity for provenance. Future training
    integration must pin its full identity in checkpoint metadata as well.
    """
    def __init__(self, manifest_path, *, expected_manifest_sha256):
        manifest_path = Path(manifest_path).resolve()
        payload = manifest_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
            raise ValueError('Native manifest identity mismatch')
        manifest = json.loads(payload)
        binary = Path(manifest['binary']).resolve()
        source = Path(__file__).with_suffix('.c')
        if (manifest['abi'] != ABI or manifest['flags'] != FLAGS or
                binary.parent != manifest_path.parent or not binary.is_file() or
                hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binary_sha256'] or
                hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256'] or
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['wrapper_sha256']):
            raise ValueError('Native feature runtime identity mismatch')
        self.identity = manifest
        self._lib = ctypes.CDLL(str(binary))
        self._lib.tetris_features_abi.argtypes = []
        self._lib.tetris_features_abi.restype = ctypes.c_char_p
        if self._lib.tetris_features_abi().decode('ascii') != ABI:
            raise ValueError('Native feature ABI mismatch')
        self._call = self._lib.tetris_candidate_features
        self._call.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint16, shape=(20,), flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),
            ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, shape=(16,), flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.uint8, shape=(1840,16), flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.bool_, shape=(1840,), flags='C_CONTIGUOUS')]
        self._call.restype = ctypes.c_int
        self._scales = np.ascontiguousarray(FEATURE_SCALE, dtype=np.int32)
        if ACTION_COUNT != 1840 or len(FEATURE_NAMES) != 16:
            raise ValueError('Unexpected action or feature dimensions')

    def __call__(self, public, placements):
        if not isinstance(placements, (list, tuple)) or not placements:
            return reference(public, placements)
        # Unusual inputs retain the reference's complete semantics. Normal
        # placements pack into a single contiguous i64 matrix, including cells.
        try:
            board = public['board']
            if len(board) != 20 or any(len(row) != 10 for row in board):
                return reference(public, placements)
            packed = []
            for p in placements:
                cells = p['occupiedCells']
                if (len(cells) != 4 or isinstance(p['actionId'], (bool, np.bool_)) or
                        not isinstance(p['actionId'], (int, np.integer))):
                    return reference(public, placements)
                a,b,c,d = cells
                packed.append((p['actionId'], p['hold'], p['rotation'], p['x'], p['y'],
                    a['x'], a['y'], b['x'], b['y'], c['x'], c['y'], d['x'], d['y']))
            packed = np.asarray(packed)
            if packed.dtype != np.dtype(np.int64):
                return reference(public, placements)
            packed = np.ascontiguousarray(packed)
            rows = np.asarray([sum(1 << x for x, cell in enumerate(row) if cell is not None)
                               for row in board], dtype=np.uint16)
            current = PIECES.index(public['currentPiece']['type'])
            held = PIECES.index(public['hold'] or public['next'][0])
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            return reference(public, placements)
        matrix = np.zeros((ACTION_COUNT, len(FEATURE_NAMES)), dtype=np.uint8)
        mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        status = self._call(rows, packed, len(packed), current, held, self._scales, matrix, mask)
        if status:
            return reference(public, placements)
        return matrix, mask
