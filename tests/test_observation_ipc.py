"""Both live wire protocols preserve dtype, rank, metadata and SB3 stacking."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_observation_ipc import ReplayPipes, stack_observations
from verify_feature_backend_benchmark import identical


def test_real_forkserver_responses_preserve_metadata_and_stacked_arrays():
    frames = [[({'state': np.array([rank, index], dtype=np.float32),
                'candidate': np.full((2, 3), rank+index, dtype=np.uint8)},
               index/10, False, {'pieces': index, 'new_holes': rank}, {})
               for index in range(2)] for rank in range(2)]
    pipes = ReplayPipes(frames)
    processes = list(pipes.processes)
    try:
        for index in range(2):
            full, cached = pipes.request('full', index), pipes.request('cached', index)
            assert identical(full, cached) and identical(full, [worker[index] for worker in frames])
            stacked = stack_observations(full)
            for name in stacked:
                np.testing.assert_array_equal(stacked[name], np.stack([worker[index][0][name] for worker in frames]))
            assert stacked['candidate'].dtype == np.uint8 and stacked['state'].dtype == np.float32
    finally:
        pipes.close()
    assert all(not process.is_alive() for process in processes)
