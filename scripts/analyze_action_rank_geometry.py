"""Describe saved rank flips by exact occupied cells; never runs a policy or env."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

import numpy as np

from collect_critic_calibration import array_manifest
from training.candidate_features import FEATURE_NAMES, FEATURE_SCALE, PIECES
from training.tetris_core.actions import decode_action
from training.tetris_core.pieces import CELLS

INDEX={name:index for index,name in enumerate(FEATURE_NAMES)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def occupied_cells(features, action):
    features=np.asarray(features)
    if (features.shape!=(16,) or features.dtype!=np.uint8 or
            np.any(features>FEATURE_SCALE)):
        raise ValueError('Invalid saved feature vector')
    decoded=decode_action(int(action))
    if (decoded!={name:int(features[INDEX[name]]) for name in ['hold','rotation','x']} |
            {'y':int(features[INDEX['y_plus_3']])-3}):
        raise ValueError('Saved action and feature coordinates disagree')
    shape=CELLS[PIECES[int(features[0])],decoded['rotation']]
    min_x=min(x for x,y in shape);min_y=min(y for x,y in shape)
    # Encoded x/y are the occupied bounding box, NOT the matrix origin.
    cells=tuple(sorted((decoded['x']+x-min_x,decoded['y']+y-min_y) for x,y in shape))
    if len(set(cells))!=4 or any(not 0<=x<10 or y>=20 for x,y in cells):
        raise ValueError('Saved occupied cells outside legal board bounds')
    return cells


def compare_row(reference,candidate,reference_action,candidate_action,board_count):
    left,right=occupied_cells(reference,reference_action),occupied_cells(candidate,candidate_action)
    lv,rv=tuple(c for c in left if c[1]>=0),tuple(c for c in right if c[1]>=0)
    reference,candidate=np.asarray(reference,dtype=np.int16),np.asarray(candidate,dtype=np.int16)
    clear=INDEX['cleared_lines'];holes=INDEX['holes_after'];aggregate=INDEX['aggregate_height_after']
    # aggregate_height = holes + occupied blocks. A legal nonoverlapping lock
    # adds its visible cells, and each cleared row removes exactly ten blocks.
    base_left=int(reference[aggregate]-reference[holes]+10*reference[clear]-len(lv))
    base_right=int(candidate[aggregate]-candidate[holes]+10*candidate[clear]-len(rv))
    if base_left!=board_count or base_right!=board_count:
        raise ValueError('Saved features violate independent board occupancy conservation')
    delta=candidate-reference
    assert int(delta[aggregate])==int(delta[holes])-10*int(delta[clear])+len(rv)-len(lv)
    return dict(same_full_cells=(left==right),same_visible_cells=(lv==rv),
        same_piece=bool(reference[0]==candidate[0]),same_hold=bool(reference[1]==candidate[1]),
        reference_visible_count=len(lv),candidate_visible_count=len(rv),
        delta_cleared=int(delta[clear]),delta_holes=int(delta[holes]),
        delta_aggregate=int(delta[aggregate]),independent_board_count=board_count)


def clearing_summary(rows,selected):
    values=rows['delta_cleared'][selected]
    if not len(values):return dict(samples=0)
    return dict(samples=len(values),fewer=int((values<0).sum()),same=int((values==0).sum()),
        more=int((values>0).sum()),net_cleared_delta=int(values.sum()),
        delta_histogram={str(int(v)):int((values==v).sum()) for v in np.unique(values)})


def run(folder):
    from resource_guard import now,write_json
    from training.evaluation.evaluate_resumable import durable_json
    if (folder/'geometry_summary.json').exists():raise FileExistsError('Preserve completed geometry analysis')
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for name,digest in design['frozen_files'].items():
            if sha(name)!=digest:raise ValueError('Frozen geometry input changed: '+name)
    unchanged();source=Path(design['rank_dir']);dataset=Path(design['dataset_dir']);ledger=[]
    for index in range(1,5):
        chunk=source/f'rows_{index:04d}';manifest=json.loads((chunk/'manifest.json').read_text())
        if (manifest['identity']['chunk']!=index or manifest['identity']['design_sha256']!=sha(source/'design.json') or
                manifest['file_sha256']!=sha(chunk/'rows.npz')):
            raise ValueError('Saved rank row identity differs')
        with np.load(chunk/'rows.npz',allow_pickle=False) as data:rows={n:data[n] for n in data.files}
        if array_manifest(rows)!=manifest['arrays']:raise ValueError('Saved rank row arrays differ')
        np.testing.assert_array_equal(rows['worker'],np.tile(np.arange(8),512))
        np.testing.assert_array_equal(rows['global_step'],np.repeat(np.arange((index-1)*512,index*512),8))
        np.testing.assert_array_equal(rows['chunk'],np.full(4096,index))
        np.testing.assert_array_equal(rows['disagrees'],rows['reference_action']!=rows['candidate_action'])
        segment=dataset/f'chunk_{index:04d}';metadata=json.loads((segment/'dataset.json').read_text())
        if sha(segment/'rollout.npz')!=metadata['file_sha256']:raise ValueError('Saved rollout identity differs')
        with np.load(segment/'rollout.npz',allow_pickle=False) as data:
            observed={name:data[name] for name in ['obs_state','info_episode_seed']}
        if array_manifest(observed)!={n:metadata['arrays'][n] for n in observed}:
            raise ValueError('Saved board observation identity differs')
        states=observed['obs_state'].reshape(4096,237)
        if not np.isin(states[:,:200],[0,1]).all():raise ValueError('Board occupancy encoding differs')
        np.testing.assert_array_equal(rows['episode_seed'],observed['info_episode_seed'].reshape(-1))
        counts=states[:,:200].sum(axis=1).astype(np.int16)
        for row in np.flatnonzero(rows['disagrees']):
            result=compare_row(rows['reference_features'][row],rows['candidate_features'][row],
                rows['reference_action'][row],rows['candidate_action'][row],int(counts[row]))
            result.update(chunk=index,worker=int(rows['worker'][row]),global_step=int(rows['global_step'][row]),
                reference_action=int(rows['reference_action'][row]),candidate_action=int(rows['candidate_action'][row]))
            ledger.append(result)
    if len(ledger)!=design['expected_flips']:raise ValueError('Changed rank sample count')
    arrays={name:np.asarray([r[name] for r in ledger]) for name in ledger[0]}
    all_rows=np.ones(len(ledger),dtype=bool)
    same_geometry_hold_piece=arrays['same_visible_cells'] & arrays['same_hold'] & arrays['same_piece']
    result=dict(at=now(),status='passed',flips=len(ledger),
        same_visible_cells=int(arrays['same_visible_cells'].sum()),same_full_cells=int(arrays['same_full_cells'].sum()),
        same_hold=int(arrays['same_hold'].sum()),same_piece=int(arrays['same_piece'].sum()),
        same_visible_cells_hold_piece=int(same_geometry_hold_piece.sum()),
        joint_visible_hold_piece_counts={f'visible={v},hold={h},piece={p}':int(((arrays['same_visible_cells']==v)&
            (arrays['same_hold']==h)&(arrays['same_piece']==p)).sum())
            for v in [False,True] for h in [False,True] for p in [False,True]},
        clearing=dict(all=clearing_summary(arrays,all_rows),
            same_visible_hold_piece=clearing_summary(arrays,same_geometry_hold_piece),
            changed_visible_or_hold_or_piece=clearing_summary(arrays,~same_geometry_hold_piece)),
        delta_aggregate_sum=int(arrays['delta_aggregate'].sum()),delta_holes_sum=int(arrays['delta_holes'].sum()),
        visible_count_delta_sum=int((arrays['candidate_visible_count']-arrays['reference_visible_count']).sum()),
        all_independent_board_conservation_exact=True,environment_steps=0,model_forward_calls=0,optimizer_steps=0,
        interpretation='Post-rank descriptive analysis. Same cells/piece/hold does not prove path, drop-score, queue or trajectory equivalence; no quality or causal claim.')
    unchanged()
    target=folder/'flipped_geometry.npz'
    if target.exists():raise FileExistsError('Preserve prior geometry output')
    temporary=folder/'flipped_geometry.tmp'
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush()
        import os
        os.fsync(stream.fileno())
    with np.load(temporary,allow_pickle=False) as data:
        assert array_manifest({n:data[n] for n in data.files})==array_manifest(arrays)
    temporary.rename(target)
    durable_json(folder/'rows_manifest.json',dict(design_sha256=sha(folder/'design.json'),
        file_sha256=sha(target),arrays=array_manifest(arrays)))
    write_json(folder/'geometry_summary.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
