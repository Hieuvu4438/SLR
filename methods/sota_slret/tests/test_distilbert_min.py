"""Compare DistilBertMin to HF DistilBertModel (run with base env python: torch 2.11 + transformers 5.5)."""
import glob, sys
import torch
sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from distilbert_min import DistilBertMin
from transformers import AutoModel, AutoTokenizer
p = glob.glob("/home/haipd/.cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32-multilingual-v1/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(p)
b = tok(["liebe zuschauer guten abend", "und nun die wettervorhersage für morgen donnerstag den zwölften august", "你们好！"],
        padding="max_length", max_length=48, truncation=True, return_tensors="pt")
ref = AutoModel.from_pretrained(p).eval()
mine = DistilBertMin.from_dir(p).eval()
with torch.no_grad():
    r = ref(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    m = mine(b["input_ids"], b["attention_mask"])
valid = b["attention_mask"].bool()
d = (r - m).abs()[valid].max().item()
print("max |diff| on valid tokens:", d)
assert d < 1e-4
print("PASS")
