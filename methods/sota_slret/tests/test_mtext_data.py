"""mtext data path: native captions + HF tokenizer through the untouched upstream loaders (CPU)."""
import sys
import numpy as np
sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from upstream import load_upstream, seds_argv
up, args = load_upstream(seds_argv("ph", "/home/haipd/SLR/artifacts/sota_slret_agent/splits/ph_val_s0",
                                   ["--do_train", "--batch_size", "8", "--num_thread_reader", "0", "--max_words", "48",
                                    "--output_dir", "/tmp/claude-1007/-home-haipd-SLR/b3c87030-4695-430a-8f5a-b899c4a8d757/scratchpad/mt"]))
args.world_size, args.rank, args.n_gpu = 1, 0, 1
import methods_registry
from modules.tokenization_clip import SimpleTokenizer
m = methods_registry.get("mtext", {"lang": "native"})
tok = m._tok()
dl, _ = up.DATALOADER_DICT["ph_pose"]["test"](args, SimpleTokenizer()); dl = m.wrap_eval_loader(dl, args)
ds = dl.dataset
k0 = list(ds.sentences_dict)[0]
item = ds[0]
ids = item["text"]["pairs_text"][0].tolist()
dec = tok.hf.decode([i for i in ids if i != 0])
print("eval text:", ds.sentences_dict[k0]); print("decoded :", dec)
assert ds.sentences_dict[k0] == ds.captions[k0]["ori_text"]
assert ids[0] == tok.hf.cls_token_id and tok.hf.sep_token_id in ids
lens = [len(tok.tokenize(t)) + 2 for t in ds.sentences_dict.values()]
print("val native token length p50/p90/p99/max", np.percentile(lens, [50, 90, 99]), max(lens), "frac>48:", np.mean(np.array(lens) > 48))
tdl, _, _ = up.DATALOADER_DICT["ph_pose"]["train"](args, SimpleTokenizer()); tdl = m.wrap_train_loader(tdl, args)
t = tdl.dataset
sid, txt = t.sentences_dict[0]
assert txt == t.captions[sid]["ori_text"], (txt, t.captions[sid])
it = t[0]; print("train aug ids:", tok.hf.decode([i for i in it["text"]["pairs_text_aug"][0].tolist() if i != 0]))
en = methods_registry.get("mtext", {"lang": "english"})
dl2, _ = up.DATALOADER_DICT["ph_pose"]["test"](args, SimpleTokenizer()); dl2 = en.wrap_eval_loader(dl2, args)
assert dl2.dataset.sentences_dict[k0] == dl2.dataset.captions[k0]["text"]
print("PASS")
