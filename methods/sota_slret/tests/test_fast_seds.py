"""FastCLIP4Clip must compute the same function as upstream CLIP4Clip (forward + grads).
Run: python methods/sota_slret/tests/test_fast_seds.py  (needs GPU + PHOENIX features)."""
import os
import sys
import time

sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from upstream import load_upstream, seds_argv  # noqa: E402

up, args = load_upstream(seds_argv("ph", "/home/haipd/SLR/artifacts/sota_slret_agent/splits/ph_val_s0",
                                   ["--do_train", "--batch_size", "24", "--num_thread_reader", "4",
                                    "--output_dir", "/home/haipd/SLR/runs/sota_slret/smoke/test_fast"]))
import torch  # noqa: E402
from fast_seds import FastCLIP4Clip  # noqa: E402
from modules.modeling import CLIP4Clip  # noqa: E402
from modules.tokenization_clip import SimpleTokenizer  # noqa: E402

args = up.set_seed_logger(args)
device, _ = up.init_device(args, 0)
cache = os.path.join(str(up.PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
sd = torch.load("ckpts/ph_best_model.bin", map_location="cpu")
args.init_sign_model = None
ref = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache, distributed=True, state_dict=dict(sd), task_config=args).to(device)
fast = FastCLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache, distributed=True, state_dict=dict(sd), task_config=args).to(device)
dl, _, _ = up.DATALOADER_DICT["ph_pose"]["train"](args, SimpleTokenizer())
b = {k: v.to(device) for k, v in next(iter(dl)).items()}


def run(m, train):
    m.train(train)
    m.zero_grad()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    body = {'pose': b['body_pose'], 'clips_start': b['body_clips_start'], 'mask': b['body_mask'], 'rgb': b['RGB_feature']}
    if train:
        out = m(b['pairs_text'], b['pairs_segment'], b['pairs_mask'], {'pose': b['right_pose']}, {'pose': b['left_pose']},
                body, b['pairs_text_aug'], b['pairs_mask_aug'])
        out[0].backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        return out[0].item(), grads
    with torch.no_grad():
        mask, vp, vr = m.get_visual_output({'pose': b['right_pose']}, {'pose': b['left_pose']}, dict(body))
    return vp, vr


vp0, vr0 = run(ref, False)
vp1, vr1 = run(fast, False)
print("eval pose max|diff|", (vp0 - vp1).abs().max().item(), "rgb", (vr0 - vr1).abs().max().item())
assert torch.equal(vp0, vp1) and torch.equal(vr0, vr1)
torch.cuda.synchronize(); t = time.time()
l0, g0 = run(ref, True)
torch.cuda.synchronize(); t1 = time.time()
l1, g1 = run(fast, True)
torch.cuda.synchronize(); t2 = time.time()
print("train loss", l0, l1, "time ref %.2fs fast %.2fs" % (t1 - t, t2 - t1))
assert abs(l0 - l1) < 1e-4 * max(1, abs(l0))
assert set(g0) == set(g1)
worst = max(((g0[n] - g1[n]).abs().max().item() / (g0[n].abs().max().item() + 1e-12), n) for n in g0)
print("worst relative grad diff", worst)
assert worst[0] < 1e-3
print("PASS")
