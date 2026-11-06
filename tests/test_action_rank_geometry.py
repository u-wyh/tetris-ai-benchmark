"""Bounding-box decoding matches formal occupied cells and block conservation."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_action_rank_geometry import occupied_cells,compare_row
from training.tetris_core import TetrisCore
from training.tetris_core.pieces import make_piece
from training.tetris_core.actions import encode_action
from training.evaluation.fast_candidate_features import candidate_features


@pytest.mark.parametrize('kind','IJLOSTZ')
@pytest.mark.parametrize('pattern',['empty','partial_lock','single_clear','holes'])
@pytest.mark.parametrize('hold',[None,'I','O'])
def test_geometry_matches_formal_occupied_cells_for_pieces_holds_and_edges(kind,pattern,hold):
    core=TetrisCore(42);core.current=make_piece(kind);core.hold=hold
    if pattern=='partial_lock':
        core.current.update(x=0,y=-1);core.board[1][0]='Z'
    elif pattern=='single_clear':core.board[19]=['Z' if x not in (3,4) else None for x in range(10)]
    elif pattern=='holes':
        for y in range(10,20):
            core.board[y]=['Z' if (x+3*y)%7<2 else None for x in range(10)]
    placements=core.get_legal_placements();features,_=candidate_features(core.get_public_observation(),placements)
    count=sum(c is not None for row in core.board for c in row)
    for p in placements:
        f=features[p['actionId']]
        assert occupied_cells(f,p['actionId'])==tuple(sorted((c['x'],c['y']) for c in p['occupiedCells']))
        assert compare_row(f,f,p['actionId'],p['actionId'],count)['same_full_cells']


@pytest.mark.parametrize('kind',[0,2,3,4,6])
def test_normalized_rotational_footprints_and_negative_y(kind):
    # I/L/O/S/Z shape examples include matrices whose occupied bounds move.
    for rotation in range(4):
        f=np.zeros(16,dtype=np.uint8);f[:5]=[kind,0,rotation,3,2]
        cells=occupied_cells(f,encode_action(0,rotation,3,-1))
        assert min(x for x,y in cells)==3 and min(y for x,y in cells)==-1
    if kind==3:
        signatures=[]
        for rotation in range(4):
            f=np.zeros(16,dtype=np.uint8);f[:5]=[3,0,rotation,3,2]
            signatures.append(occupied_cells(f,encode_action(0,rotation,3,-1)))
        assert len(set(signatures))==1


def test_action_coordinate_mismatch_rejected():
    f=np.zeros(16,dtype=np.uint8);f[:5]=[3,0,0,3,3]
    with pytest.raises(ValueError,match='coordinates'):occupied_cells(f,encode_action(0,0,4,0))


def test_independent_board_count_mismatch_rejected():
    core=TetrisCore(42);p=core.get_legal_placements()[0]
    features,_=candidate_features(core.get_public_observation(),[p]);f=features[p['actionId']]
    with pytest.raises(ValueError,match='conservation'):compare_row(f,f,p['actionId'],p['actionId'],1)
