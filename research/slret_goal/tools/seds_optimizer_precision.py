"""Numerical baseline control: FP32 BertAdam moments, native parameter dtypes."""
import torch


def ensure_fp32_moments(optimizer):
    """Initialize moments before first update, never repair already-rounded history."""
    initialized = 0
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is None or parameter.dtype != torch.float16:
                continue
            state = optimizer.state[parameter]
            if not state:
                state.update(step=0, next_m=torch.zeros_like(parameter, dtype=torch.float32),
                             next_v=torch.zeros_like(parameter, dtype=torch.float32))
                initialized += 1
            elif state['next_m'].dtype != torch.float32 or state['next_v'].dtype != torch.float32:
                raise ValueError('Cannot convert already-rounded optimizer history')
    return initialized
