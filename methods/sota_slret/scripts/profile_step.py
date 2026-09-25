"""Profile one upstream SEDS training step (forward+backward) on a real batch."""
import sys
import time

sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from upstream import load_upstream, seds_argv  # noqa: E402

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
up, args = load_upstream(seds_argv("ph", "/home/haipd/SLR/artifacts/sota_slret_agent/splits/ph_val_s0",
                                   ["--do_train", "--batch_size", str(BS), "--num_thread_reader", "8",
                                    "--output_dir", "/home/haipd/SLR/runs/sota_slret/smoke/profile"]))
import torch  # noqa: E402

args = up.set_seed_logger(args)
device, n_gpu = up.init_device(args, 0)
from modules.tokenization_clip import SimpleTokenizer  # noqa: E402

import os
from fast_seds import FastCLIP4Clip
model = FastCLIP4Clip.from_pretrained(args.cross_model, cache_dir=os.path.join(str(up.PYTORCH_PRETRAINED_BERT_CACHE), "distributed"), distributed=True, state_dict=None, task_config=args).to(device)
model.train()
dl, _, _ = up.DATALOADER_DICT["ph_pose"]["train"](args, SimpleTokenizer())
it = iter(dl)
batch = {k: v.to(device) for k, v in next(it).items()}


def step():
    s = batch
    loss = model(s['pairs_text'], s['pairs_segment'], s['pairs_mask'], {'pose': s['right_pose']}, {'pose': s['left_pose']},
                 {'pose': s['body_pose'], 'clips_start': s['body_clips_start'], 'mask': s['body_mask'], 'rgb': s['RGB_feature']},
                 s['pairs_text_aug'], s['pairs_mask_aug'])[0]
    loss.backward()
    model.zero_grad()


print("pose shape", batch['right_pose'].shape, batch['body_pose'].shape, "rgb", batch['RGB_feature'].shape)
step(); torch.cuda.synchronize()
t = time.time(); step(); torch.cuda.synchronize(); print("step s", time.time() - t)
print("peak GB", torch.cuda.max_memory_allocated() / 1e9)
t = time.time(); _ = [next(it) for _ in range(3)]; print("dataloader s/batch", (time.time() - t) / 3)
import sys as _s
if len(_s.argv)>2: raise SystemExit
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    step(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
