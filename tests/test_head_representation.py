import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import freeze_head_representation as cache


def fixture(folder):
    mask=np.zeros(1840,dtype=bool);mask[:2]=True
    digest=hashlib.sha256(mask.tobytes()).hexdigest()
    states=[]
    for i in range(640):
        state=dict(reference_action=0,input_digest=dict(mask=digest,observation='public'))
        if i<8:state.update(kind='pair',pair=i//2,seed=[992000,992004,992006,992008][i//2],
                            split='train',role='base' if i%2==0 else 'alternative',good_action=1)
        else:
            lane=(i-8)%8
            state.update(kind='normal',lane=lane,chunk=(i-8)//8,seed=100000+lane,
                         split='train' if lane<4 else 'validation')
        states.append(state)
    arrays=dict(h=np.zeros((1280,64),np.float32),raw_score=np.zeros(1280,np.float32),
        actions=np.tile(np.array([0,1],np.int64),640),offsets=np.arange(0,1282,2,dtype=np.int64),
        original_weight=np.zeros(64,np.float32),original_bias=np.zeros(1,np.float32))
    meta=dict(version=cache.CACHE_VERSION,status='passed',states=states,source_model_sha256=cache.MODEL_SHA,
        source_model_counts=[30003200,36625],source_policy_adam_counts_rng_unchanged=True,
        eligible_for_single_joint_lp=True,work=dict(gpu_forward_calls=80))
    return arrays,meta


def save(folder,arrays,metadata):
    np.savez(folder/'features.npz',**arrays)
    metadata['arrays_sha256']=cache.sha(folder/'features.npz')
    (folder/'cache_metadata.json').write_text(json.dumps(metadata))
    return cache.sha(folder/'cache_metadata.json'),metadata['arrays_sha256']


def test_exact_public_cache_can_be_loaded_without_any_model_or_environment(tmp_path):
    arrays,metadata=fixture(tmp_path);hashes=save(tmp_path,arrays,metadata)
    loaded,info=cache.load_cache(tmp_path,*hashes)
    assert np.array_equal(loaded['offsets'],arrays['offsets']) and len(info['states'])==640


@pytest.mark.parametrize('change',['hidden_rng','dtype','nonfinite','offset','duplicate_action','mask','wrong_reference','heldout_seed','heldout_split','pair_seed','preferred_masked','source_model'])
def test_resealed_schema_or_episode_leakage_is_rejected(tmp_path,change):
    arrays,meta=fixture(tmp_path)
    if change=='hidden_rng':arrays['hidden_rng']=np.zeros(10)
    elif change=='dtype':arrays['h']=arrays['h'].astype(np.float64)
    elif change=='nonfinite':arrays['h'][0,0]=np.nan
    elif change=='offset':arrays['offsets'][1]=0
    elif change=='duplicate_action':arrays['actions'][1]=0
    elif change=='mask':meta['states'][0]['input_digest']['mask']='wrong'
    elif change=='wrong_reference':meta['states'][0]['reference_action']=5
    elif change=='heldout_seed':meta['states'][12]['seed']=100000
    elif change=='heldout_split':meta['states'][12]['split']='train'
    elif change=='pair_seed':meta['states'][0]['seed']=1234
    elif change=='preferred_masked':meta['states'][0]['good_action']=8
    else:meta['source_model_sha256']='changed'
    hashes=save(tmp_path,arrays,meta)
    with pytest.raises(ValueError):cache.load_cache(tmp_path,*hashes)


def test_external_seal_is_checked_before_opening_npz(tmp_path,monkeypatch):
    arrays,meta=fixture(tmp_path);hashes=save(tmp_path,arrays,meta)
    (tmp_path/'features.npz').write_bytes(b'changed')
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Unsealed archive opened'))
    with pytest.raises(ValueError,match='External'):cache.load_cache(tmp_path,*hashes)
