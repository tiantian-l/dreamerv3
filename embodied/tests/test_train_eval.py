import json

from embodied.run.train_eval import (
    _load_topk_manifest,
    _rank_topk,
    _write_topk_manifest,
)


def test_rank_topk_uses_success_then_earlier_step():
  entries = [
      {'step': 300, 'success': 0.90, 'filename': 'c'},
      {'step': 200, 'success': 0.90, 'filename': 'b'},
      {'step': 100, 'success': 0.86, 'filename': 'a'},
      {'step': 400, 'success': 0.95, 'filename': 'd'},
  ]
  ranked = _rank_topk(entries, 3)
  assert [entry['filename'] for entry in ranked] == ['d', 'b', 'c']


def test_topk_manifest_roundtrip(tmp_path):
  path = tmp_path / 'manifest.json'
  entries = [
      {
          'step': 100_000,
          'success': 0.859375,
          'eval_cycle': 1,
          'filename': 'step_0000100000_success_0.8594.ckpt',
      },
  ]
  _write_topk_manifest(path, entries, threshold=0.85, limit=5)
  assert _load_topk_manifest(path) == entries
  payload = json.loads(path.read_text())
  assert payload['threshold'] == 0.85
  assert payload['limit'] == 5
  assert payload['tie_break'] == 'earlier_step'
