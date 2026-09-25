"""Non-baseline methods.

H1 `cmr` — Cross-modal Matching Reranker: a light cross-encoder over (text tokens, pose tokens,
RGB tokens), trained jointly with the SEDS bi-encoder on in-batch hard negatives and used at
inference to re-rank each query's bi-encoder top-K. Tensor contracts:
  text tokens  seq       [B, Lt=32, 512]   valid mask text_mask   [B, 32]  (1 = valid)
  pose tokens  vis_pose  [B, Lv=65, 512]   video_mask (upstream)  [B, 65]  (0 = valid)
  rgb tokens   vis_rgb   [B, 65, 512]
  bi-encoder scores  S   [N_video, N_text]
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from methods_registry import Method, register


class SignMatchHead(nn.Module):
    """Text tokens attend (cross-attn) to typed pose/RGB clip tokens; masked-mean → match logit."""

    def __init__(self, d=512, layers=2, heads=8, ff=2048, dropout=0.1, memory="pose_rgb"):
        super().__init__()
        self.memory = memory
        self.ln_t = nn.LayerNorm(d)
        self.ln_v = nn.LayerNorm(d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def build_memory(self, vp, vr, vf, vmask_valid):
        if self.memory == "pose_rgb":
            mem = torch.cat([vp + self.type_emb[0], vr + self.type_emb[1]], 1)
            valid = torch.cat([vmask_valid, vmask_valid], 1)
        elif self.memory == "fusion":
            mem, valid = vf, vmask_valid
        elif self.memory == "rgb":
            mem, valid = vr, vmask_valid
        else:
            raise ValueError(self.memory)
        return mem, valid

    def forward(self, t, t_valid, mem, mem_valid):
        assert t.dim() == 3 and mem.dim() == 3 and t.shape[0] == mem.shape[0]
        assert t_valid.shape == t.shape[:2] and mem_valid.shape == mem.shape[:2]
        x = self.ln_t(t.float())
        m = self.ln_v(mem.float())
        for layer in self.layers:
            x = layer(x, m, tgt_key_padding_mask=~t_valid, memory_key_padding_mask=~mem_valid)
        w = t_valid.float().unsqueeze(-1)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.out(self.ln_out(pooled)).squeeze(-1)


def _sample_neg(weights, gen):
    """weights [N, M] >= 0 with zeros on forbidden entries; one sample per row."""
    weights = weights + 1e-12 * (weights.sum(1, keepdim=True) == 0)
    return torch.multinomial(weights, 1, generator=gen).squeeze(1)


@register
class CMR(Method):
    name = "cmr"
    defaults = {"itm_weight": 1.0, "neg": "hard", "detach": False, "memory": "pose_rgb", "layers": 2,
                "rerank_k": 16, "rerank_w": 1.0, "kmax": 32}

    def setup(self, model, args):
        dev = next(model.parameters()).device
        model.match_head = SignMatchHead(layers=self.cfg["layers"], memory=self.cfg["memory"]).to(dev)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(args.seed) + 12345)
        self.last_eval_artifacts = None

    def extra_loss(self, model, reps, batch):
        S = model._last_fusion_sim  # [B_video, B_text], detached, eval-style dual mix
        B = S.shape[0]
        ids = batch["pairs_text"].view(B, -1)
        same = (ids[:, None, :] == ids[None, :, :]).all(-1)  # identical captions are not negatives
        neg = self.cfg["neg"]
        if neg == "hard":
            w_v = F.softmax(S.float(), dim=0).T      # [text, video]: prob of each video for text j
            w_t = F.softmax(S.float(), dim=1)        # [video, text]
        elif neg == "random":
            w_v = torch.ones_like(S).float().T
            w_t = torch.ones_like(S).float()
        elif neg in ("visual", "textual"):
            if neg == "visual":  # videos similar to the positive *video* (sign-space confusability)
                e = F.normalize(reps["vis_fusion"].detach().float()[:, 1:].mean(1), dim=-1)
            else:                # videos whose *captions* are similar (semantic hardness)
                tm = reps["text_mask"].float().unsqueeze(-1)
                e = F.normalize((reps["seq"].detach().float() * tm).sum(1) / tm.sum(1), dim=-1)
            G = e @ e.T / 0.05
            w_v = F.softmax(G, dim=1)   # for text j (positive video j): videos similar to j
            w_t = F.softmax(G, dim=1)   # for video i: texts of items similar to i
        else:
            raise ValueError(neg)
        w_v = w_v.masked_fill(same, 0.0)
        w_t = w_t.masked_fill(same, 0.0)
        nv = _sample_neg(w_v, self.gen)  # negative video for each text
        nt = _sample_neg(w_t, self.gen)  # negative text for each video
        ar = torch.arange(B, device=S.device)
        vid_idx = torch.cat([ar, nv, ar])
        txt_idx = torch.cat([ar, ar, nt])
        labels = torch.cat([torch.ones(B), torch.zeros(2 * B)]).to(S.device)
        get = (lambda x: x.detach()) if self.cfg["detach"] else (lambda x: x)
        vvalid = reps["video_mask"] == 0
        mem, mvalid = model.match_head.build_memory(get(reps["vis_pose"]), get(reps["vis_rgb"]), get(reps["vis_fusion"]), vvalid)
        logits = model.match_head(get(reps["seq"])[txt_idx], reps["text_mask"][txt_idx] > 0, mem[vid_idx], mvalid[vid_idx])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        acc = ((logits > 0).float() == labels).float().mean()
        return self.cfg["itm_weight"] * loss, {"itm": loss, "itm_acc": acc}

    @torch.no_grad()
    def _pair_logits(self, model, vtok, vvalid, ttok, tvalid, pairs, bs=512):
        out = []
        for i in range(0, len(pairs), bs):
            p = pairs[i:i + bs]
            out.append(model.match_head(ttok[p[:, 1]], tvalid[p[:, 1]], vtok[p[:, 0]], vvalid[p[:, 0]]).float())
        return torch.cat(out) if out else torch.zeros(0, device=vtok.device)

    @torch.no_grad()
    def final_scores(self, sims, model=None, vids=None, texts=None):
        S = sims["fusion"]
        model.eval()
        vp = torch.cat([v[1] for v in vids]); vr = torch.cat([v[2] for v in vids]); vm = torch.cat([v[0] for v in vids])
        vf = torch.cat([model.fusion(v[1], v[2], v[0]) for v in vids]) if self.cfg["memory"] == "fusion" else None
        vtok, vvalid = model.match_head.build_memory(vp, vr, vf, vm == 0)
        ttok = torch.cat([t[1] for t in texts]); tvalid = torch.cat([t[0] for t in texts]) > 0
        kmax = min(self.cfg["kmax"], S.shape[0], S.shape[1])
        St = torch.from_numpy(S).to(vtok.device)
        top_v = St.topk(kmax, dim=0).indices          # [kmax, N_text]  videos per text
        top_t = St.topk(kmax, dim=1).indices          # [N_video, kmax] texts per video
        nt, nv = S.shape[1], S.shape[0]
        p_t2v = torch.stack([top_v.T.reshape(-1), torch.arange(nt, device=St.device).repeat_interleave(kmax)], 1)
        p_v2t = torch.stack([torch.arange(nv, device=St.device).repeat_interleave(kmax), top_t.reshape(-1)], 1)
        l_t2v = self._pair_logits(model, vtok, vvalid, ttok, tvalid, p_t2v).view(nt, kmax)   # [text, k]
        l_v2t = self._pair_logits(model, vtok, vvalid, ttok, tvalid, p_v2t).view(nv, kmax)   # [video, k]
        self.last_eval_artifacts = {"top_v": top_v.T.cpu().numpy(), "l_t2v": l_t2v.cpu().numpy(),
                                    "top_t": top_t.cpu().numpy(), "l_v2t": l_v2t.cpu().numpy()}
        return rerank(S, self.last_eval_artifacts, self.cfg["rerank_k"], self.cfg["rerank_w"])


def rerank(S, art, k, w, big=1e6):
    """Direction-specific reranked score matrices. Top-k candidates (by bi-encoder score) are
    re-scored with S + w * itm_logit and placed above all non-top-k candidates."""
    S_t2v = S.copy()
    S_v2t = S.copy()
    if k <= 0:
        return S_t2v, S_v2t
    top_v, l_t2v = art["top_v"][:, :k], art["l_t2v"][:, :k]   # [N_text, k]
    for t in range(S.shape[1]):
        v = top_v[t]
        S_t2v[v, t] = big + S[v, t] + w * l_t2v[t]
    top_t, l_v2t = art["top_t"][:, :k], art["l_v2t"][:, :k]   # [N_video, k]
    for vi in range(S.shape[0]):
        t = top_t[vi]
        S_v2t[vi, t] = big + S[vi, t] + w * l_v2t[vi]
    return S_t2v, S_v2t


# ---------------------------------------------------------------------------------------------
# H7 `mtext` — multilingual CLIP-aligned text tower (token level) on native or English captions.
#   text tokens: HF encoder last_hidden_state [B, L, 768] -> per-token Dense 768->512 (the
#   sentence-transformers distillation head into CLIP ViT-B/32 space) -> [B, L, 512]; mask = ids != pad.
# Upstream loaders are reused unchanged via a tokenizer adapter + caption rewrite.
# ---------------------------------------------------------------------------------------------
import os as _os

from fast_seds import FastCLIP4Clip

MTEXT_REPO = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def _mtext_path():
    # resolve the cached snapshot directly (only the files we need were downloaded)
    import glob
    hub = _os.path.expanduser("~/.cache/huggingface/hub")
    snaps = glob.glob(_os.path.join(hub, "models--" + MTEXT_REPO.replace("/", "--"), "snapshots", "*"))
    snaps = [s for s in snaps if _os.path.exists(_os.path.join(s, "model.safetensors"))]
    assert len(snaps) == 1, snaps
    return snaps[0]


class HFTokAdapter:
    """Mimics the CLIP SimpleTokenizer interface used by upstream `_get_text`."""

    def __init__(self, hf):
        self.hf = hf
        assert hf.pad_token_id == 0, "upstream pads with id 0"
        self.special = {"<|startoftext|>": hf.cls_token_id, "<|endoftext|>": hf.sep_token_id}

    def tokenize(self, text):
        return self.hf.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.special[t] if t in self.special else self.hf.convert_tokens_to_ids(t) for t in tokens]


def _native(entry):
    e = entry[0] if isinstance(entry, list) else entry
    return e.get("ori_text", e["text"])  # How2Sign captions are already native English


def rewrite_captions(ds, lang, tok):
    if lang == "native":
        for k, v in list(ds.sentences_dict.items()):
            if isinstance(v, tuple):   # train loader: idx -> (sentence_id, text)
                ds.sentences_dict[k] = (v[0], _native(ds.captions[v[0]]))
            else:                      # eval loader: sentence_id -> text
                ds.sentences_dict[k] = _native(ds.captions[k])
    elif lang != "english":
        raise ValueError(lang)
    ds.tokenizer = tok


class MTextCLIP4Clip(FastCLIP4Clip):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super().__init__(cross_config, clip_state_dict, task_config)
        import torch as _t
        from distilbert_min import DistilBertMin  # == HF DistilBertModel (tests/test_distilbert_min.py)
        path = _mtext_path()
        # module name contains "clip." so the optimizer puts it in the CLIP lr group (1e-5)
        self.mclip = DistilBertMin.from_dir(path)
        dense = _t.load(_os.path.join(path, "2_Dense", "pytorch_model.bin"), map_location="cpu") \
            if _os.path.exists(_os.path.join(path, "2_Dense", "pytorch_model.bin")) else \
            __import__("safetensors.torch", fromlist=["load_file"]).load_file(_os.path.join(path, "2_Dense", "model.safetensors"))
        w = dense["linear.weight"]
        self.mclip_proj = nn.Linear(w.shape[1], w.shape[0], bias="linear.bias" in dense)
        self.mclip_proj.weight.data.copy_(w)
        if "linear.bias" in dense:
            self.mclip_proj.bias.data.copy_(dense["linear.bias"])

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False, get_hidden=True):
        ids = input_ids.view(-1, input_ids.shape[-1])
        mask = ids != 0
        h = self.mclip(ids, mask)
        hidden = self.mclip_proj(h.float())
        assert hidden.shape[:2] == ids.shape and hidden.shape[-1] == 512
        return mask.float(), hidden.float()


@register
class MText(Method):
    name = "mtext"
    defaults = {"lang": "native"}
    model_cls = MTextCLIP4Clip

    def _tok(self):
        from transformers import AutoTokenizer
        if not hasattr(self, "_hf_tok"):
            self._hf_tok = HFTokAdapter(AutoTokenizer.from_pretrained(_mtext_path()))
        return self._hf_tok

    def wrap_train_loader(self, dl, args):
        rewrite_captions(dl.dataset, self.cfg["lang"], self._tok())
        return dl

    def wrap_eval_loader(self, dl, args):
        rewrite_captions(dl.dataset, self.cfg["lang"], self._tok())
        return dl
