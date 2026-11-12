"""One fixed, CUDA-only 64-parameter preference fit and full-policy endpoint audit."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from freeze_head_representation import MODEL_SHA, load_cache, sha, unchanged, public_view, validate_arrays
from paired_branch_capture import selected_cases, specs
from paired_branch_guard import require_authorization, process_args, argument
from restore_rank_contexts import observation_digest
from verify_feature_backend_benchmark import identical

FIXED = dict(steps=256, learning_rate=.01, betas=[.9, .999], eps=1e-8,
             pair_temperature=1., teacher_kl_coefficient=1., delta_l2_coefficient=.01,
             dtype='float64', batch_size=8, normal_chunk_schedule='step_mod79')


def batch_indices(metadata, step):
    pairs = metadata.get('training_pair_indices', [0, 2, 4, 6])
    if pairs not in ([0, 2, 4, 6], [0, 2, 4, 6, 640, 642, 644, 646]):
        raise ValueError('Unregistered training coverage schedule')
    quartet = (step % 2)*4 if len(pairs) == 8 else 0
    indices = pairs[quartet:quartet+4]+[8+(step % 79)*8+lane for lane in range(4)]
    for row, i in enumerate(indices):
        state = metadata['states'][i]
        if (state['split'] != 'train' or
                (row >= 4 and (state['kind'] != 'normal' or state['seed'] not in range(100000, 100004))) or
                (row < 4 and (state['kind'] != 'pair' or state['seed'] in (992011, 992016, 992027)))):
            raise ValueError('Validation entered the training minibatch')
    return indices


def pack(arrays, metadata, indices, device):
    import torch
    lengths = [int(arrays['offsets'][i+1]-arrays['offsets'][i]) for i in indices]
    h = np.zeros((len(indices), max(lengths), 64), np.float64)
    raw = np.zeros(h.shape[:2], np.float64);mask = np.zeros(h.shape[:2], bool)
    good = [];base = []
    for row, i in enumerate(indices):
        left, right = arrays['offsets'][i:i+2];n = lengths[row]
        h[row, :n] = arrays['h'][left:right];raw[row, :n] = arrays['raw_score'][left:right];mask[row, :n] = True
        actions = arrays['actions'][left:right];state = metadata['states'][i]
        if row < 4:
            good.append(int(np.flatnonzero(actions == state['good_action'])[0]))
            base.append(int(np.flatnonzero(actions == state['reference_action'])[0]))
    return tuple(torch.as_tensor(x, device=device) for x in (h, raw, mask, np.array(good), np.array(base)))


def loss_terms(delta, h, raw, mask, good, base):
    import torch
    from torch.nn import functional as F
    scores = raw+h@delta
    row = torch.arange(4, device=delta.device)
    preference = F.softplus(-(scores[row, good]-scores[row, base])).mean()
    teacher_log = F.log_softmax(raw[4:].masked_fill(~mask[4:], -1e30), dim=-1)
    proposed_log = F.log_softmax(scores[4:].masked_fill(~mask[4:], -1e30), dim=-1)
    kl = (teacher_log.exp()*(teacher_log-proposed_log)).sum(dim=-1).mean()
    regularization = .01*delta.square().sum()
    return preference+kl+regularization, preference, kl, regularization


def episode_mean(values, seeds):
    return float(np.mean([np.mean([value for value, s in zip(values, seeds) if s == seed])
                          for seed in sorted(set(seeds))]))


def apply_delta(policy, delta):
    import torch
    candidate = copy.deepcopy(policy)
    with torch.no_grad():
        head = candidate.score_head[-1]
        head.weight.copy_((head.weight.double()+torch.as_tensor(delta, device=head.weight.device,
                                                              dtype=torch.float64).reshape_as(head.weight)).to(head.weight.dtype))
    candidate.requires_grad_(False)
    candidate.set_training_mode(False)
    return candidate


def live_guard(folder, design):
    from guarded_process import heartbeat_ok
    require_authorization(design, folder)
    guard = folder/'guard';launch = json.loads((guard/'launch.json').read_text())
    if ((guard/'result.json').exists() or launch['profile'] != json.loads((ROOT/'configs/training_resources.json').read_text()) or
            launch['command'] != [str(ROOT/'.venv/bin/python'), str(Path(__file__).resolve()), '--run-dir', str(folder)] or
            not 0 < launch['seconds'] <= 120):
        raise RuntimeError('Matching live original-profile head-probe guardian required')
    group = next(row[3:] for row in Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
    state = dict(row.split('=', 1) for row in subprocess.check_output(['systemctl', '--user', 'show',
        launch['unit']+'.service', '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'ActiveState'], text=True, timeout=5).splitlines())
    kernel = Path('/sys/fs/cgroup')/group.lstrip('/')
    quota, period = (kernel/'cpu.max').read_text().split()
    if (state['ActiveState'] != 'active' or state['ControlGroup'] != group or quota == 'max' or
            not 0 < int(quota)/int(period) <= 2.5 or (kernel/'memory.max').read_text().strip() == 'max' or
            not 0 < int((kernel/'memory.max').read_text()) <= 6*1024**3 or (kernel/'memory.swap.max').read_text().strip() != '0' or
            set(os.sched_getaffinity(0)) != set(launch['physical_cpus']) or len(launch['physical_cpus']) != 6):
        raise RuntimeError('Head-probe kernel CPU/memory/swap/affinity limits differ')
    watchdog = process_args(int(state['MainPID']));guardian_pid = int(argument(watchdog, '--guardian-pid'))
    guardian = process_args(guardian_pid)
    if (str(ROOT/'scripts/guarded_process.py') not in watchdog or
            Path(argument(watchdog, '--heartbeat')).resolve() != guard/'heartbeat.json' or
            str(ROOT/'scripts/resource_guard.py') not in guardian or
            Path(argument(guardian, '--output')).resolve() != guard or
            not heartbeat_ok(guard/'heartbeat.json', guardian_pid, 10)):
        raise RuntimeError('Head-probe watchdog/thermal heartbeat missing')
    return dict(unit=launch['unit'], cpu_limit=float(int(quota)/int(period)), memory_limit=int((kernel/'memory.max').read_text()),
                guardian_pid=guardian_pid, watchdog_pid=int(state['MainPID']))


def fit(arrays, metadata, receipt, check, checkpoint):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    delta = torch.nn.Parameter(torch.zeros(64, device='cuda', dtype=torch.float64))
    optimizer = torch.optim.Adam([delta], lr=.01, betas=(.9, .999), eps=1e-8, foreach=False)
    if optimizer.state:
        raise ValueError('Fresh head optimizer already has inherited moments')
    receipt['head_optimizer_initial_state_empty'] = True
    history = []
    for step in range(256):
        check()
        batch = pack(arrays, metadata, batch_indices(metadata, step), 'cuda')
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(delta, *batch)
        if not all(torch.isfinite(value).item() for value in terms):
            raise ValueError('Nonfinite head objective')
        receipt['head_training_forwards'] += 1
        terms[0].backward()
        if not torch.isfinite(delta.grad).all().item():
            raise ValueError('Nonfinite head gradient')
        optimizer.step()
        if not torch.isfinite(delta).all().item():
            raise ValueError('Nonfinite updated head')
        receipt['head_optimizer_steps'] += 1
        if (step+1) % 64 == 0:
            history.append(dict(step=step+1, objective_before_update=[float(t.detach().cpu()) for t in terms]))
            checkpoint(delta, optimizer, step+1)
    state = optimizer.state[delta]
    if int(state['step'].item()) != 256 or len(optimizer.state) != 1:
        raise ValueError('Head optimizer actual step or parameter scope differs')
    return delta.detach().cpu().numpy(), history


def current_cases(design, preferences, public, states, training_block=0):
    cases_by_split = {}
    for split in ('train', 'validation'):
        evidence = json.loads(Path(design[split+'_window']).read_text())
        block = training_block if split == 'train' else 0
        cases = selected_cases(preferences, split, block)
        probe = json.loads(Path(design[split+'_probe']).read_text())
        if (evidence['status'] != 'passed' or evidence['label_counts']['beneficial'] != 4 or
                evidence['model']['sha256'] != MODEL_SHA or evidence['model']['counts'] != [30003200, 36625] or
                evidence.get('case_split', 'train') != split or evidence['per_branch_model_identity_explicit'] is not True or
                evidence.get('training_block', 0) != block or
                evidence['optimizer_steps'] != 0 or [c['case'] for c in evidence['comparisons']] != cases or
                probe['status'] != 'probe_passed' or probe['source_full_state_restored'] is not True or
                probe['branch_ids'] != [s['branch_id'] for s in specs(cases)]):
            raise ValueError('Current-GPU label/probe/case readiness differs')
        for pair, case in enumerate(cases):
            index = case['state_id'];view = public_case(public, index)
            if (any(case[k] != states[index][k] for k in states[index]) or
                    observation_digest(**view) != probe['public_digests'][pair*2] or
                    probe['public_digests'][pair*2] != probe['public_digests'][pair*2+1]):
                raise ValueError('Current matched-state public observations differ')
        cases_by_split[split] = cases
    if set(c['seed'] for c in cases_by_split['train']) & set(c['seed'] for c in cases_by_split['validation']):
        raise ValueError('Whole-episode preference leakage')
    return cases_by_split


def public_case(public, index):
    return dict(observation=dict(state=public['observation_state'][index], candidate=public['observation_candidate'][index]),
                mask=public['action_mask'][index])


def observe(policy, views, receipt, role):
    import torch
    if len(views) != 8 or receipt['full_policy_forward_attempts'] >= receipt.get('full_policy_forward_budget', 82):
        raise ValueError('Finite CUDA8 full-policy budget exceeded')
    observations = {k: torch.as_tensor(np.stack([v['observation'][k] for v in views]), device='cuda')
                    for k in views[0]['observation']}
    masks = np.stack([v['mask'] for v in views]);captured = []
    def hook(module, inputs, output):
        captured.append((inputs[0].detach().cpu().numpy().copy(), output.detach().cpu().numpy().reshape(-1).copy()))
    handle = policy.score_head[-1].register_forward_hook(hook)
    receipt['full_policy_forward_attempts'] += 1
    try:
        with torch.inference_mode():
            distribution = policy.get_distribution(observations, action_masks=masks)
            modes = distribution.distribution.probs.argmax(dim=1).cpu().numpy()
        receipt[role+'_policy_forwards'] += 1
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Expected a single final-head call')
    h, raw = captured[0];rows = [];offset = 0
    for index, view in enumerate(views):
        actions = np.flatnonzero(view['mask']).astype(np.int64);stop = offset+len(actions)
        rows.append(dict(actions=actions, h=h[offset:stop], raw_score=raw[offset:stop],
                         mode=int(modes[index]), input_digest=observation_digest(**view)))
        offset = stop
    if offset != len(raw) or h.shape != (offset, 64) or not np.isfinite(h).all() or not np.isfinite(raw).all():
        raise ValueError('Full-policy exported legal features differ')
    return rows


def cached_row(arrays, metadata, index):
    left, right = arrays['offsets'][index:index+2]
    return dict(actions=arrays['actions'][left:right], h=arrays['h'][left:right],
                raw_score=arrays['raw_score'][left:right], mode=metadata['states'][index]['reference_action'],
                input_digest=metadata['states'][index]['input_digest'])


def reconstruction(row, weight, bias):
    error = float(np.max(np.abs(row['h'].astype(np.float64)@weight.astype(np.float64)+float(bias[0])-row['raw_score'])))
    if error > 1e-4:
        raise ValueError('Head reconstruction exceeds fixed1e-4 tolerance')
    return error


def same_features(old, new):
    if (not np.array_equal(old['h'], new['h']) or not np.array_equal(old['actions'], new['actions']) or
            old['input_digest'] != new['input_digest']):
        raise ValueError('Frozen non-head representation or public input changed')


def pair_report(cases, before, after):
    rows = []
    for pair, case in enumerate(cases):
        original, candidate = before[pair*2], after[pair*2]
        same_features(original, candidate)
        good = int(np.flatnonzero(original['actions'] == case['action'])[0])
        base = int(np.flatnonzero(original['actions'] == case['baseline_action'])[0])
        old_margin = float(original['raw_score'][good])-float(original['raw_score'][base])
        margin = float(candidate['raw_score'][good])-float(candidate['raw_score'][base])
        rows.append(dict(case=case, original_margin=old_margin, candidate_margin=margin,
            original_softplus=float(np.logaddexp(0., -old_margin)), candidate_softplus=float(np.logaddexp(0., -margin)),
            actual_gpu_mode=candidate['mode'], good_is_actual_gpu_top1=candidate['mode'] == case['action'],
            pair_ahead=margin > 0, preferred_raw_rank=1+int(np.sum(candidate['raw_score'] > candidate['raw_score'][good]))))
    seeds = [c['seed'] for c in cases]
    return dict(cases=rows, episodes=len(set(seeds)), good_global_top1=sum(r['good_is_actual_gpu_top1'] for r in rows),
                episode_equal_original_loss=episode_mean([r['original_softplus'] for r in rows], seeds),
                episode_equal_candidate_loss=episode_mean([r['candidate_softplus'] for r in rows], seeds))


def raw_teacher_kl(before, after):
    before, after = np.asarray(before, np.float64), np.asarray(after, np.float64)
    before = before-float(np.max(before));after = after-float(np.max(after))
    logq = before-np.log(np.exp(before).sum());logp = after-np.log(np.exp(after).sum())
    return float(np.sum(np.exp(logq)*(logq-logp)))


def append_training_block(arrays, metadata, cases, rows):
    """Append eight sealed views in memory, retaining the complete old640 prefix."""
    if (len(metadata['states']) != 640 or len(cases) != 4 or len(rows) != 8 or
            [c['seed'] for c in cases] != [992010, 992012, 992014, 992015]):
        raise ValueError('Fixed block1 whole-episode coverage differs')
    appended = copy.deepcopy(metadata)
    for pair, case in enumerate(cases):
        left, right = rows[pair*2:pair*2+2];same_features(left, right)
        if (not np.array_equal(left['raw_score'], right['raw_score']) or
                left['mode'] != case['baseline_action'] or right['mode'] != case['baseline_action'] or
                case['action'] not in left['actions']):
            raise ValueError('New source duplicate, legal label or original GPU mode differs')
        for role, row in zip(('base', 'alternative'), (left, right)):
            appended['states'].append(dict(kind='pair', pair=4+pair, role=role, seed=case['seed'],
                split='train', reference_action=case['baseline_action'], good_action=case['action'],
                input_digest=copy.deepcopy(row['input_digest'])))
    output = {k: v.copy() for k, v in arrays.items()}
    for key in ('h', 'raw_score', 'actions'):
        output[key] = np.concatenate([arrays[key]]+[r[key] for r in rows])
    output['offsets'] = np.r_[arrays['offsets'], arrays['offsets'][-1]+np.cumsum([len(r['actions']) for r in rows])].astype(np.int64)
    appended['training_pair_indices'] = [0, 2, 4, 6, 640, 642, 644, 646]
    validate_arrays(output, appended)
    if set(c['seed'] for c in cases) & {s['seed'] for s in metadata['states'] if s['split'] == 'validation'}:
        raise ValueError('New preference episode overlaps validation')
    return output, appended


def reused_validation(design, cases, public, weight, bias):
    """Verify external seals before reading old validation arrays; no new source forward."""
    reuse = design['reuse_validation']
    for key in ('readiness', 'arrays'):
        if sha(reuse[key]) != reuse[key+'_sha256']:
            raise ValueError('External validation seal differs')
    ready = json.loads(Path(reuse['readiness']).read_text())
    if (ready['status'] != 'passed' or ready['source_model_sha256'] != MODEL_SHA or
            ready['source_ppo_counts'] != [30003200, 36625] or ready['validation_cases'] != cases or
            ready['source_validation_modes_verified'] is not True or ready['validation_episodes'] != 3 or
            ready['validation_features_sha256'] != reuse['arrays_sha256'] or
            ready['inherited_eligible_for_training_flags_unchanged'] is not True):
        raise ValueError('Reused validation provenance differs')
    states = [dict(reference_action=c['baseline_action'], input_digest=observation_digest(**public_case(public, c['state_id'])))
              for c in cases for _ in range(2)]
    with np.load(reuse['arrays'], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_arrays(arrays, dict(states=states))
    if not np.array_equal(weight, arrays['original_weight']) or not np.array_equal(bias, arrays['original_bias']):
        raise ValueError('Reused validation source head differs')
    rows = [cached_row(arrays, dict(states=states), i) for i in range(8)]
    for i in range(0, 8, 2):
        same_features(rows[i], rows[i+1])
        if not np.array_equal(rows[i]['raw_score'], rows[i+1]['raw_score']):
            raise ValueError('Reused validation duplicate scores differ')
    for row in rows:
        reconstruction(row, weight, bias)
    return rows


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from audit_head_direction import load_public
    from restore_own_death_contexts import load_verified_chunk
    from training.train_transaction import global_rng_state
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from training.evaluation.evaluate_resumable import durable_json
    from resource_guard import now
    design = json.loads((folder/'probe_design.json').read_text())
    unchanged(design['frozen_files']);proof = live_guard(folder, design)
    blocks = design.get('training_blocks', 1)
    if type(blocks) is not int or blocks not in (1, 2):
        raise ValueError('Only the registered4/8episode protocols are supported')
    if design['fixed'] != FIXED or not torch.cuda.is_available():
        raise RuntimeError('Fixed head protocol or available CUDA required')
    if (folder/'execution.json').exists():
        raise FileExistsError('Single fit already attempted; inspect actual checkpoint, never restart blindly')
    receipt = dict(at=now(), pid=os.getpid(), status='started', full_policy_forward_attempts=0,
        source_policy_forwards=0, candidate_policy_forwards=0, head_training_forwards=0,
        head_optimizer_steps=0, ppo_optimizer_steps=0, environment_steps=0, model_loads=0, guard_proof=proof)
    receipt['full_policy_forward_budget'] = 82 if blocks == 1 else 83
    durable_json(folder/'execution.json', receipt)
    try:
        if (training_source_identity() != design['training_sources'] or
                inference_source_identity(resumable=True) != design['inference_sources'] or
                evaluation_runtime_identity(False) != design['evaluation_runtime'] or sha(design['model']) != MODEL_SHA):
            raise ValueError('Source numerical runtime/model changed')
        arrays, metadata = load_cache(Path(design['cache']), design['cache_metadata_sha256'], design['cache_arrays_sha256'])
        public, states, _, _ = load_public(Path(design['public_folder']), design['frozen_files'])
        preferences = json.loads((Path(design['public_folder'])/'preferences.json').read_text())
        cases = current_cases(design, preferences, public, states)
        extension = []
        if blocks == 2:
            extended_design = dict(design, train_window=design['extension_window'], train_probe=design['extension_probe'])
            extension = current_cases(extended_design, preferences, public, states, training_block=1)['train']
        for i, case in enumerate(cases['train']):
            state = metadata['states'][i*2]
            if (state['seed'], state['reference_action'], state['good_action']) != (case['seed'], case['baseline_action'], case['action']):
                raise ValueError('Training cache differs from current-GPU labels')
            if observation_digest(**public_case(public, case['state_id'])) != state['input_digest']:
                raise ValueError('Training public input differs from sealed features')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source = MaskablePPO.load(design['model'], device='cuda', env=None);receipt['model_loads'] = 1
        if source.device.type != 'cuda' or source.get_env() is not None or [source.num_timesteps, source._n_updates] != [30003200, 36625]:
            raise ValueError('Original CUDA source counts/environment differ')
        source.policy.set_training_mode(False)
        before = copy.deepcopy((source.policy.state_dict(), source.policy.optimizer.state_dict(), global_rng_state(source), source._n_updates))
        weight = source.policy.score_head[-1].weight.detach().cpu().numpy().reshape(-1).copy()
        bias = source.policy.score_head[-1].bias.detach().cpu().numpy().copy()
        if not np.array_equal(weight, arrays['original_weight']) or not np.array_equal(bias, arrays['original_bias']):
            raise ValueError('Frozen cache/source head differs')
        val_views = [public_case(public, c['state_id']) for c in cases['validation'] for _ in range(2)]
        if blocks == 2:
            val_before = reused_validation(design, cases['validation'], public, weight, bias)
            new_views = [public_case(public, c['state_id']) for c in extension for _ in range(2)]
            require_authorization(design, folder)
            new_rows = observe(source.policy, new_views, receipt, 'source')
            for row in new_rows:reconstruction(row, weight, bias)
            # Persist only the newly computed eight rows; never overwrite the old caches.
            with (folder/'extension_features.npz').open('xb') as stream:
                np.savez_compressed(stream, h=np.concatenate([r['h'] for r in new_rows]),
                    raw_score=np.concatenate([r['raw_score'] for r in new_rows]),
                    actions=np.concatenate([r['actions'] for r in new_rows]),
                    offsets=np.r_[0, np.cumsum([len(r['actions']) for r in new_rows])].astype(np.int64),
                    original_weight=weight, original_bias=bias)
                stream.flush();os.fsync(stream.fileno())
            arrays, metadata = append_training_block(arrays, metadata, extension, new_rows)
            cases['train'] += extension
        else:
            val_before = observe(source.policy, val_views, receipt, 'source')
        max_error = 0.
        for pair, case in enumerate(cases['validation']):
            left, right = val_before[pair*2:pair*2+2];same_features(left, right)
            if left['mode'] != case['baseline_action'] or right['mode'] != case['baseline_action']:
                raise ValueError('Current validation source GPU mode differs')
        for row in val_before:
            max_error = max(max_error, reconstruction(row, weight, bias))
        if blocks == 1:
            with (folder/'validation_features.npz').open('xb') as stream:
                np.savez_compressed(stream, h=np.concatenate([r['h'] for r in val_before]),
                    raw_score=np.concatenate([r['raw_score'] for r in val_before]), actions=np.concatenate([r['actions'] for r in val_before]),
                    offsets=np.r_[0, np.cumsum([len(r['actions']) for r in val_before])].astype(np.int64),
                    original_weight=weight, original_bias=bias)
                stream.flush();os.fsync(stream.fileno())
        readiness = dict(at=now(), status='passed', allowed_scope='Exactly one fixed256-step head probe; validation never fitted',
            source_model_sha256=MODEL_SHA, source_ppo_counts=[30003200, 36625], train_cases=cases['train'], validation_cases=cases['validation'],
            train_episodes=4*blocks, validation_episodes=3, source_validation_modes_verified=True,
            validation_features_sha256=sha(design['reuse_validation']['arrays'] if blocks == 2 else folder/'validation_features.npz'),
            source_reconstruction_max=max_error,
            inherited_eligible_for_training_flags_unchanged=False,
            scope='Prior artifacts retain eligible_for_training=false. This explicit bounded-probe readiness does not grant PPO resume or promotion.')
        # Record the unchanged legacy flags themselves, not an inverted approval claim.
        readiness['inherited_eligible_for_training_flags_unchanged'] = all(
            json.loads(Path(design[split+'_window']).read_text())['eligible_for_training'] is False for split in ('train', 'validation'))
        if blocks == 2:
            readiness['inherited_eligible_for_training_flags_unchanged'] &= json.loads(Path(design['extension_window']).read_text())['eligible_for_training'] is False
            readiness['extension_features_sha256'] = sha(folder/'extension_features.npz')
            readiness['training_pair_indices'] = metadata['training_pair_indices']
            readiness['validation_source_forward_reused'] = True
            readiness['source_reconstruction_max'] = max(max_error, *(reconstruction(r, weight, bias) for r in new_rows))
        if not readiness['inherited_eligible_for_training_flags_unchanged'] or metadata['eligible_for_training'] is not False:
            raise ValueError('Prior evidence eligibility scope unexpectedly changed')
        durable_json(folder/'readiness.json', readiness)
        def check():
            require_authorization(design, folder)
        def checkpoint(delta, optimizer, step):
            tmp = folder/'head_fit_checkpoint.pt.tmp'
            torch.save(dict(delta=delta.detach().cpu(), optimizer=optimizer.state_dict(), head_optimizer_steps=step,
                source_model_sha256=MODEL_SHA, source_ppo_counts=[30003200, 36625], probe_design_sha256=sha(folder/'probe_design.json'),
                eligible_for_ppo_resume=False), tmp)
            with tmp.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(tmp, folder/'head_fit_checkpoint.pt')
            fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            durable_json(folder/'progress.json', dict(at=now(), head_optimizer_steps=step,
                checkpoint_sha256=sha(folder/'head_fit_checkpoint.pt'), source_ppo_counts=[30003200, 36625]))
            live_guard(folder, design)
        delta, history = fit(arrays, metadata, receipt, check, checkpoint)
        candidate = apply_delta(source.policy, delta)
        changed = [name for name, value in candidate.state_dict().items() if not torch.equal(value, before[0][name])]
        if changed != ['score_head.2.weight']:
            raise ValueError('Candidate changed parameters outside the64-value final weight')
        candidate_weight = candidate.score_head[-1].weight.detach().cpu().numpy().reshape(-1).copy()
        train_views = [public_case(public, c['state_id']) for c in cases['train'] for _ in range(2)]
        train_after = []
        for start in range(0, len(train_views), 8):
            check();train_after += observe(candidate, train_views[start:start+8], receipt, 'candidate')
        check();val_after = observe(candidate, val_views, receipt, 'candidate')
        train_before = [cached_row(arrays, metadata, i) for i in list(range(8))+(list(range(640, 648)) if blocks == 2 else [])]
        train_report = pair_report(cases['train'], train_before, train_after)
        val_report = pair_report(cases['validation'], val_before, val_after)
        for old, new in zip(train_before+val_before, train_after+val_after):
            same_features(old, new);max_error = max(max_error, reconstruction(new, candidate_weight, bias))
        normal = {'train': [], 'validation': []};discrepancies = [];previous_sha = None
        directory = Path(design['normal_dir'])
        for chunk in range(79):
            check();path = directory/f'chunk_{chunk:04d}'
            snapshot, _, manifest = load_verified_chunk(path, design['frozen_files'][str(path/'manifest.json')], design['normal_identity'])
            if manifest['number'] != chunk or manifest['previous_manifest_sha256'] != previous_sha or snapshot['round'] != chunk*256:
                raise ValueError('Normal complete public source chain differs')
            views = [public_view(worker['state']) for worker in snapshot['workers']]
            output = observe(candidate, views, receipt, 'candidate')
            for lane, new in enumerate(output):
                index = 8+chunk*8+lane;old = cached_row(arrays, metadata, index);same_features(old, new)
                max_error = max(max_error, reconstruction(new, candidate_weight, bias))
                split = metadata['states'][index]['split']
                normal[split].append(dict(seed=100000+lane, teacher_kl=raw_teacher_kl(old['raw_score'], new['raw_score']),
                                          mode_changed=new['mode'] != old['mode']))
                if new['mode'] != old['mode']:
                    discrepancies.append(dict(state=index, chunk=chunk, seed=100000+lane, split=split,
                                              source_mode=old['mode'], candidate_actual_gpu_mode=new['mode']))
            previous_sha = design['frozen_files'][str(path/'manifest.json')]
        normal_report = {split: dict(states=len(rows), episodes=len({r['seed'] for r in rows}),
            mode_changes=sum(r['mode_changed'] for r in rows),
            episode_equal_teacher_kl=episode_mean([r['teacher_kl'] for r in rows], [r['seed'] for r in rows]))
            for split, rows in normal.items()}
        gate = (train_report['good_global_top1'] >= 1 and val_report['good_global_top1'] >= 1 and
                val_report['episode_equal_candidate_loss'] < val_report['episode_equal_original_loss'])
        if gate:
            evaluation_model = copy.copy(source);evaluation_model.policy = candidate
            evaluation_model.save(folder/'evaluation_candidate.zip')
            durable_json(folder/'evaluation_candidate_identity.json', dict(at=now(), evaluation_only=True,
                eligible_for_ppo_resume=False, source_model_sha256=MODEL_SHA, source_ppo_counts=[30003200, 36625],
                head_optimizer_steps=256, model_sha256=sha(folder/'evaluation_candidate.zip'), quality_gate_passed=False))
        with (folder/'coefficients.npz').open('xb') as stream:
            np.savez_compressed(stream, delta=delta, source_weight=weight, candidate_weight=candidate_weight, bias=bias)
            stream.flush();os.fsync(stream.fileno())
        check();unchanged(design['frozen_files'])
        if not identical(before, (source.policy.state_dict(), source.policy.optimizer.state_dict(), global_rng_state(source), source._n_updates)):
            raise ValueError('Original source policy/Adam/counts/RNG changed')
        if (receipt['head_optimizer_steps'] != 256 or receipt['source_policy_forwards'] != 1 or
                receipt['candidate_policy_forwards'] != 80+blocks or
                receipt['full_policy_forward_attempts'] != receipt['full_policy_forward_budget']):
            raise ValueError('Actual finite work counts differ')
        result = dict(at=now(), status='completed', eligible_for_quality_evaluation=bool(gate), quality_gate_passed=False,
            eligible_for_ppo_resume=False, source_state_unchanged=True, changed_candidate_parameters=changed,
            delta_l1=float(np.abs(delta).sum()), delta_l2=float(np.linalg.norm(delta)), source_ppo_counts=[30003200, 36625],
            max_score_reconstruction_error=max_error, training_history=history, train=train_report, validation=val_report,
            normal=normal_report, all_normal_mode_changes=discrepancies, coefficients_sha256=sha(folder/'coefficients.npz'),
            checkpoint_sha256=sha(folder/'head_fit_checkpoint.pt'), work={k: receipt[k] for k in
                ('source_policy_forwards', 'candidate_policy_forwards', 'head_training_forwards', 'head_optimizer_steps',
                 'ppo_optimizer_steps', 'environment_steps', 'model_loads')},
            scope='One fixed head fit, final endpoint only; training/correlated validation separated. Actual GPU greedy modes. Cached-logit teacher KL is a defined numerical target, not bitwise original GPU probabilities. No whole-game quality or promotion claim.')
        durable_json(folder/'probe_result.json', result)
        receipt.update(at=now(), status='completed', eligible_for_quality_evaluation=bool(gate), original_state_unchanged=True)
    except BaseException as exc:
        receipt.update(at=now(), status='failed', error=repr(exc), partial_work_requires_audit=True)
        raise
    finally:
        durable_json(folder/'execution.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
