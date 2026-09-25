"""gradcache_step must reproduce DDP semantics: encoder grads = (1/n) * full-batch grad,
loss-phase params (logit_scale) = full grad; tested with dropout off + BN eval so the
per-rank BN difference vanishes. Also checks RNG-restored recompute is bit-identical."""
import os, sys
sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from upstream import load_upstream, seds_argv
up, args = load_upstream(seds_argv("ph", "/home/haipd/SLR/artifacts/sota_slret_agent/splits/ph_val_s0",
    ["--do_train", "--batch_size", "24", "--num_thread_reader", "4", "--output_dir", "/home/haipd/SLR/runs/sota_slret/smoke/test_gc"]))
import torch
from torch import nn
from fast_seds import FastCLIP4Clip
from gradcache import gradcache_step, encode_chunk, frozen_bn_stats, REP_KEYS
from modules.tokenization_clip import SimpleTokenizer
args = up.set_seed_logger(args); device, _ = up.init_device(args, 0)
cache = os.path.join(str(up.PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
m = FastCLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache, distributed=False, state_dict=None, task_config=args).to(device)
dl, _, _ = up.DATALOADER_DICT["ph_pose"]["train"](args, SimpleTokenizer())
b = {k: v.to(device) for k, v in next(iter(dl)).items()}

# 1) RNG-restored recompute in full train mode is bit identical
m.train(); st = (torch.get_rng_state(), torch.cuda.get_rng_state())
with torch.no_grad(), frozen_bn_stats(m):
    o1 = encode_chunk(m, {k: v[:8] for k, v in b.items()})
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])
    o2 = encode_chunk(m, {k: v[:8] for k, v in b.items()})
assert all(torch.equal(o1[k], o2[k]) for k in REP_KEYS), "recompute mismatch"
print("recompute identical: OK")

# 2) deterministic mode: dropout off, BN eval
m.train()
for mod in m.modules():
    if isinstance(mod, nn.Dropout): mod.p = 0.0
    if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.eval()
with torch.no_grad():
    fullo = encode_chunk(m, b)
    ch = [encode_chunk(m, {k: v[i:i+8] for k, v in b.items()}) for i in (0, 8, 16)]
for k in REP_KEYS:
    cat = torch.cat([c[k] for c in ch]); d = (cat - fullo[k]).abs().max().item(); sc = fullo[k].abs().max().item()
    print("chunk-vs-full", k, fullo[k].dtype, "max|diff| %.2e (scale %.2e)" % (d, sc))
m.zero_grad(set_to_none=True)
body = {'pose': b['body_pose'], 'clips_start': b['body_clips_start'], 'mask': b['body_mask'], 'rgb': b['RGB_feature']}
out = m(b['pairs_text'], b['pairs_segment'], b['pairs_mask'], {'pose': b['right_pose']}, {'pose': b['left_pose']}, body, b['pairs_text_aug'], b['pairs_mask_aug'])
out[0].backward()
ref = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
m.zero_grad(set_to_none=True)
logs = gradcache_step(m, b, chunk=8)
n = 3
print("loss ref %.6f gc %.6f" % (out[0].item(), logs["loss"]))
assert abs(out[0].item() - logs["loss"]) < 1e-3 * abs(out[0].item())
got = {nm: p.grad.clone() for nm, p in m.named_parameters() if p.grad is not None}
assert set(got) == set(ref), set(got) ^ set(ref)
num = den = 0.0; cos_min = (1.0, "")
for nm in ref:
    expect = (ref[nm] if "logit_scale" in nm else ref[nm] / n).float().flatten()
    g = got[nm].float().flatten()
    num += (g - expect).pow(2).sum().item(); den += expect.pow(2).sum().item()
    if expect.norm() > 1e-3 * (den ** 0.5 + 1e-12) and expect.numel() > 1:
        c = torch.nn.functional.cosine_similarity(g, expect, dim=0).item()
        cos_min = min(cos_min, (c, nm))
rel = (num / den) ** 0.5
print("global relative grad err %.2e ; min per-param cosine %s" % (rel, cos_min))
assert rel < 1e-2 and cos_min[0] > 0.99
print("PASS")
