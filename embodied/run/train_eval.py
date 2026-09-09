import collections
import json
import shutil
from pathlib import Path
from functools import partial as bind

import elements
import embodied
import numpy as np


def _load_topk_manifest(path):
  if not path.exists():
    return []
  try:
    data = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(f'Could not read Top-K manifest {path}: {exc}') from exc
  entries = data.get('checkpoints', [])
  if not isinstance(entries, list):
    raise RuntimeError(
        f'Invalid Top-K manifest {path}: checkpoints must be a list')
  return entries


def _rank_topk(entries, limit):
  """Rank by success descending and, for ties, earlier step first."""
  return sorted(
      entries, key=lambda x: (-float(x['success']), int(x['step'])))[:limit]


def _write_topk_manifest(path, entries, threshold, limit):
  payload = {
      'metric': 'eval_epstats/log/success',
      'threshold': float(threshold),
      'limit': int(limit),
      'tie_break': 'earlier_step',
      'checkpoints': entries,
  }
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  temporary.replace(path)


def train_eval(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args):

  agent = make_agent()
  replay_train = make_replay_train()
  replay_eval = make_replay_eval()
  logger = make_logger()

  if args.eval_every_steps <= 0:
    raise ValueError(
        f'eval_every_steps must be positive, got {args.eval_every_steps}')
  if args.topk_checkpoints < 1:
    raise ValueError(
        f'topk_checkpoints must be at least 1, got {args.topk_checkpoints}')
  if not 0.0 <= args.topk_success_threshold <= 1.0:
    raise ValueError(
        'topk_success_threshold must be between 0 and 1, got '
        f'{args.topk_success_threshold}')

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  train_episodes = collections.defaultdict(elements.Agg)
  train_epstats = elements.Agg()
  eval_episodes = collections.defaultdict(elements.Agg)
  eval_epstats = elements.Agg()
  eval_map_records = []
  eval_maps_path = Path(str(logdir)) / 'eval_maps.jsonl'
  topk_dir = Path(str(logdir)) / 'best_checkpoints'
  topk_dir.mkdir(parents=True, exist_ok=True)
  topk_manifest_path = topk_dir / 'manifest.json'
  topk_entries = _load_topk_manifest(topk_manifest_path)
  topk_entries = [
      entry for entry in topk_entries
      if (topk_dir / entry['filename']).exists()]
  topk_entries = _rank_topk(topk_entries, args.topk_checkpoints)
  _write_topk_manifest(
      topk_manifest_path, topk_entries,
      args.topk_success_threshold, args.topk_checkpoints)
  eval_cycle = 0
  if eval_maps_path.exists():
    with eval_maps_path.open(encoding='utf-8') as handle:
      previous_cycles = [
          int(json.loads(line)['eval_cycle']) for line in handle if line.strip()]
    eval_cycle = max(previous_cycles, default=-1) + 1
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = elements.when.Clock(args.log_every)
  should_report = elements.when.Every(args.eval_every_steps)
  should_save = elements.when.Clock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker, mode):
    episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
    epstats = dict(train=train_epstats, eval=eval_epstats)[mode]
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    recording = bool(tran.get('log/video_recorded', 0.0) > 0.5)
    for key, value in tran.items():
      if key == 'log/video_recorded':
        continue  # control flag, not a metric
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0 and recording:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        # Convention: `log/<agg>/<name>` reduces to a single `log/<name>`
        # metric with the requested aggregator; anything else keeps the
        # legacy avg/max/sum triple for backward compatibility.
        parts = key.split('/')
        if len(parts) == 3 and parts[1] in ('avg', 'max', 'sum', 'min', 'last'):
          episode.add(f'log/{parts[2]}', value, agg=parts[1])
        else:
          episode.add(key + '/avg', value, agg='avg')
          episode.add(key + '/max', value, agg='max')
          episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      length = result.pop('length')
      ep_prefix = 'episode' if mode == 'train' else 'eval_episode'
      logger.add({'score': score, 'length': length}, prefix=ep_prefix)
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      # Arrival time: only recorded for successful episodes, so epstats
      # averages it over successes only (mean ignores absent keys).
      if result.get('log/success', 0.0) > 0.5:
        result['log/time_to_goal'] = np.float32(length)
      if mode == 'eval':
        eval_map_records.append({
            'worker': int(worker),
            'map_seed': int(result.get('log/eval_map_seed', -1)),
            'map_slot': int(result.get('log/eval_map_slot', -1)),
            'maps_per_env': int(result.get('log/eval_maps_per_env', -1)),
            'success': float(result.get('log/success', 0.0)),
            'collision': float(result.get('log/collision', 0.0)),
            'crash': float(result.get('log/crash', 0.0)),
            'timeout': float(result.get('log/timeout', 0.0)),
            'episode_length': int(length),
            'final_distance': float(
                result.get('log/final_distance', np.nan)),
            'min_distance': float(result.get('log/min_distance', np.nan)),
            'min_lidar_dist': float(
                result.get('log/min_lidar_dist', np.nan)),
        })
      epstats.add(result)

  def write_and_validate_eval_maps(records, cycle, checkpoint_step, quota):
    expected = args.eval_envs * quota
    if len(records) != expected:
      raise RuntimeError(
          f'Eval coverage error: expected {expected} counted episodes, '
          f'got {len(records)}')
    worker_counts = collections.Counter(x['worker'] for x in records)
    bad_workers = {
        worker: worker_counts.get(worker, 0)
        for worker in range(args.eval_envs)
        if worker_counts.get(worker, 0) != quota}
    if bad_workers:
      raise RuntimeError(
          f'Eval coverage error: per-worker counts must equal {quota}, '
          f'got {bad_workers}')
    configured_counts = {x['maps_per_env'] for x in records}
    if configured_counts != {quota}:
      raise RuntimeError(
          f'Eval configuration error: eval_maps_per_env must equal the '
          f'per-worker quota {quota}, got {sorted(configured_counts)}')
    seeds = [x['map_seed'] for x in records]
    if any(seed < 0 for seed in seeds):
      raise RuntimeError('Eval coverage error: missing eval_map_seed metadata')
    duplicates = sorted(
        seed for seed, count in collections.Counter(seeds).items() if count != 1)
    if duplicates or len(set(seeds)) != expected:
      raise RuntimeError(
          f'Eval coverage error: expected {expected} unique maps; '
          f'duplicate seeds={duplicates}')
    slots_by_worker = collections.defaultdict(set)
    for record in records:
      slots_by_worker[record['worker']].add(record['map_slot'])
    bad_slots = {
        worker: sorted(slots)
        for worker, slots in slots_by_worker.items() if len(slots) != quota}
    if bad_slots:
      raise RuntimeError(
          f'Eval coverage error: workers did not cover {quota} unique slots: '
          f'{bad_slots}')
    with eval_maps_path.open('a', encoding='utf-8') as handle:
      for record in sorted(records, key=lambda x: x['map_seed']):
        row = {
            'eval_cycle': int(cycle),
            'checkpoint_step': int(checkpoint_step),
            **record,
        }
        handle.write(json.dumps(row, allow_nan=True) + '\n')

  def maybe_save_topk(eval_stats, cycle, checkpoint_step):
    success = float(eval_stats['log/success'])
    if success < args.topk_success_threshold:
      return False
    step_value = int(checkpoint_step)
    if any(int(entry['step']) == step_value for entry in topk_entries):
      return False
    filename = f'step_{step_value:010d}_success_{success:.4f}.ckpt'
    candidate = {
        'step': step_value,
        'success': success,
        'eval_cycle': int(cycle),
        'filename': filename,
    }
    ranked = _rank_topk(
        [*topk_entries, candidate], args.topk_checkpoints)
    if candidate not in ranked:
      return False

    # Best checkpoints are directly loadable agent-only checkpoints. The rolling
    # full checkpoint below retains replay and step state for recovery. Using the
    # low-level API also avoids version-specific constructor arguments and an
    # extra timestamp generation below filename.
    elements.checkpoint.save(
        str(topk_dir / filename), {'agent': agent.save})

    retained = {entry['filename'] for entry in ranked}
    for entry in topk_entries:
      if entry['filename'] not in retained:
        old_path = topk_dir / entry['filename']
        if old_path.is_dir():
          shutil.rmtree(old_path)
        elif old_path.exists():
          old_path.unlink()
    topk_entries[:] = ranked
    _write_topk_manifest(
        topk_manifest_path, topk_entries,
        args.topk_success_threshold, args.topk_checkpoints)
    print(
        f'Saved Top-K agent checkpoint at step {step_value}: '
        f'success={success:.4f}')
    return True

  fns = [bind(make_env_train, i) for i in range(args.envs)]
  driver_train = embodied.Driver(fns, parallel=(not args.debug))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(lambda tran, _: policy_fps.step())
  driver_train.on_step(replay_train.add)
  driver_train.on_step(bind(logfn, mode='train'))

  fns = [bind(make_env_eval, i) for i in range(args.eval_envs)]
  driver_eval = embodied.Driver(fns, parallel=(not args.debug))
  driver_eval.on_step(replay_eval.add)
  driver_eval.on_step(bind(logfn, mode='eval'))
  driver_eval.on_step(lambda tran, _: policy_fps.step())

  stream_train = iter(agent.stream(make_stream(replay_train, 'train')))
  stream_report = iter(agent.stream(make_stream(replay_train, 'report')))
  stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)
  carry_eval = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay_train) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay_train.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver_train.on_step(trainfn)

  def reportfn(carry, stream):
    agg = elements.Agg()
    for _ in range(args.report_batches):
      batch = next(stream)
      carry, mets = agent.report(carry, batch)
      agg.add(mets)
    return carry, agg.result()

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay_train = replay_train
  cp.replay_eval = replay_eval
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  train_policy = lambda *args: agent.policy(*args, mode='train')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')
  driver_train.reset(agent.init_policy)
  while step < args.steps:

    if should_report(step):
      print('Evaluation')
      if args.eval_eps % args.eval_envs:
        raise ValueError(
            f'eval_eps ({args.eval_eps}) must be divisible by eval_envs '
            f'({args.eval_envs}) for equal per-worker map coverage')
      quota = args.eval_eps // args.eval_envs
      eval_map_records.clear()
      driver_eval.reset(agent.init_policy)
      driver_eval(eval_policy, episodes_per_env=quota)
      write_and_validate_eval_maps(
          eval_map_records, eval_cycle, step, quota)
      eval_stats = eval_epstats.result()
      maybe_save_topk(eval_stats, eval_cycle, step)
      eval_cycle += 1
      logger.add(eval_stats, prefix='eval_epstats')
      if len(replay_train):
        carry_report, mets = reportfn(carry_report, stream_report)
        logger.add(mets, prefix='report')
      if len(replay_eval):
        carry_eval, mets = reportfn(carry_eval, stream_eval)
        logger.add(mets, prefix='eval')

    driver_train(train_policy, steps=10)

    if should_log(step):
      logger.add(agg.result())
      logger.add(train_epstats.result(), prefix='epstats')
      logger.add(replay_train.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
