"""Inspect error concentration using an existing verified rollout; never sample."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from collect_critic_calibration import array_manifest, residual_metrics
from resource_guard import now, write_json


def analyze(folder):
    target = folder/'tail_analysis.json'
    if target.exists():
        raise FileExistsError('Preserve existing analysis')
    metadata = json.loads((folder/'dataset.json').read_text())
    archive = folder/'rollout.npz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata['file_sha256']:
        raise ValueError('Dataset file hash differs')
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if array_manifest(arrays) != metadata['arrays']:
        raise ValueError('Dataset arrays differ')
    residual = arrays['returns'].astype(np.float64)-arrays['values'].astype(np.float64)
    squared = residual**2
    total = float(squared.sum())
    before_terminal = np.zeros_like(residual, dtype=bool)
    terminals = np.argwhere(arrays['step_terminated'])
    # Fixed exploratory window: endpoint and up to 31 prior transitions in same episode.
    for step, worker in terminals:
        start = max(0, step-31)
        starts = np.flatnonzero(arrays['episode_starts'][:step+1, worker])
        if starts.size:
            start = max(start, int(starts[-1]))
        before_terminal[start:step+1, worker] = True
    def describe(mask):
        return dict(residual_metrics(arrays['values'][mask], arrays['returns'][mask]),
                    squared_error_share=float(squared[mask].sum())/total if total else None)
    n_top = max(1, int(np.ceil(residual.size*.01)))
    order = np.argsort(squared, axis=None, kind='stable')[-n_top:][::-1]
    top = []
    for index in order:
        step, worker = np.unravel_index(index, squared.shape)
        top.append(dict(step=int(step), worker=int(worker), residual=float(residual[step, worker]),
            new_holes=int(arrays['info_new_holes'][step, worker]),
            terminated=bool(arrays['step_terminated'][step, worker]),
            within_32_steps_before_terminal=bool(before_terminal[step, worker])))
    worker_shares = squared.sum(axis=0)/total if total else np.zeros(squared.shape[1])
    assert np.isclose(worker_shares.sum(), 1) if total else True
    result = dict(at=now(), status='passed', dataset_sha256=metadata['file_sha256'],
        worker_squared_error_shares=worker_shares.tolist(),
        terminal_events=terminals.tolist(),
        within_32_steps_before_terminal=describe(before_terminal),
        outside_terminal_window=describe(~before_terminal),
        new_holes_step=describe(arrays['info_new_holes'] > 0),
        top_ceil_one_percent=dict(samples=n_top, squared_error_share=float(squared.flat[order].sum())/total
            if total else None, transitions=top),
        scope='Exploratory concentration of fixed GAE residuals, correlated observations from one window. '
              'Terminal window includes endpoint and never crosses an episode start. '
              'No causal inference, independent validation, new sample, training or parameter selection.')
    write_json(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir.resolve())
