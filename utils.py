"""Training logger + checkpointing."""

import os
import json
import time
from datetime import datetime

import torch


class TrainingLogger:
    """JSONL logger. One line per log event. No hidden state."""

    def __init__(self, log_dir: str, experiment_name: str = 'run'):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'{experiment_name}_{stamp}.jsonl')
        self.start_time = time.time()
        self._write({'event': 'start', 'ts': self.start_time})

    def _write(self, payload: dict):
        payload.setdefault('t_elapsed', time.time() - self.start_time)
        with open(self.path, 'a') as f:
            f.write(json.dumps(payload) + '\n')

    def log_step(self, epoch: int, batch: int, **fields):
        self._write({'event': 'step', 'epoch': epoch, 'batch': batch, **fields})

    def log_event(self, name: str, **fields):
        self._write({'event': name, **fields})


def save_checkpoint(model, path: str, **extra):
    state = {
        'model_state_dict': model.state_dict(),
        **extra,
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)
