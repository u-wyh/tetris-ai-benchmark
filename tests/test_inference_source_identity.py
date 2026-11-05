import sys
import subprocess

import pytest

from training.evaluation.feature_backend import backend_name, inference_source_identity


def put(root,name,source):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)


def fixture(root):
    files={
        'training/__init__.py':'',
        'training/evaluation/evaluate.py':
            'from training.env import TetrisEnv\ndef later():\n from .optional import value\n',
        'training/evaluation/optional.py':'value=1\n',
        'training/evaluation/evaluate_resumable.py':'from training.evaluation import evaluate\n',
        'training/candidate_policy.py':'from training.candidate_features import FEATURE_SCALE\n',
        'training/candidate_features.py':'FEATURE_SCALE=1\n',
        'training/env/__init__.py':'from .tetris_env import TetrisEnv\n',
        'training/env/tetris_env.py':'from ..tetris_core import TetrisCore\n',
        'training/tetris_core/__init__.py':'from .core import TetrisCore\n',
        'training/tetris_core/core.py':'TetrisCore=None\n',
    }
    for name,source in files.items():put(root,name,source)
    return files


def test_relative_function_and_dynamic_policy_imports_are_all_tracked(tmp_path):
    files=fixture(tmp_path);identity=inference_source_identity(True,tmp_path)
    assert set(identity)==set(files)
    assert all(len(sha)==64 for sha in identity.values())


@pytest.mark.parametrize('name',[
    'training/candidate_policy.py','training/candidate_features.py',
    'training/evaluation/optional.py','training/env/tetris_env.py','training/tetris_core/core.py'])
def test_any_inference_dependency_edit_invalidates_identity(tmp_path,name):
    fixture(tmp_path);before=inference_source_identity(True,tmp_path)
    with (tmp_path/name).open('a') as stream:stream.write('\n# modified\n')
    assert inference_source_identity(True,tmp_path)!=before


def test_new_package_initializer_is_detected_and_unrelated_trainer_is_excluded(tmp_path):
    fixture(tmp_path);before=inference_source_identity(True,tmp_path)
    put(tmp_path,'training/train_vector_transaction.py','changed_train_loop=True\n')
    put(tmp_path,'training/initialization.py','changed_train_initialization=True\n')
    assert inference_source_identity(True,tmp_path)==before
    put(tmp_path,'training/evaluation/__init__.py','initialization_can_affect_inference=True\n')
    assert inference_source_identity(True,tmp_path)!=before


def test_real_closure_includes_all_current_candidate_inference_implementations():
    names=set(inference_source_identity(True))
    expected={'training/candidate_policy.py','training/candidate_features.py',
              'training/env/tetris_env.py','training/env/fast_candidate_env.py',
              'training/env/bitboard_candidate_env.py','training/evaluation/fast_candidate_features.py',
              'training/tetris_core/legal_placements.py','training/tetris_core/pieces.py',
              'training/tetris_core/rng.py','training/tetris_core/core.py',
              'training/evaluation/evaluate_resumable.py','training/evaluation/feature_backend.py'}
    assert expected <= names
    assert 'training/train_vector_transaction.py' not in names
    assert 'training/initialization.py' not in names


def test_identity_collection_does_not_initialize_numerical_libraries():
    subprocess.run([sys.executable,'-c',
        'import sys; from training.evaluation.feature_backend import inference_source_identity; '
        'inference_source_identity(True); assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
        check=True,capture_output=True,text=True,timeout=10)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError,match='Unsupported'):backend_name('unrecognized')
