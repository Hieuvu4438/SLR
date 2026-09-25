"""Registry of training methods. `baseline` reproduces upstream SEDS exactly (under gradcache).
Each method may override model class, loss, auxiliary loss, loader wrapping and eval scoring.
Unknown config keys fail loudly."""
from fast_seds import FastCLIP4Clip
from gradcache import upstream_loss


class Method:
    name = "baseline"
    defaults = {}
    model_cls = FastCLIP4Clip
    eval_streams = ("fusion", "pose", "rgb")

    def __init__(self, cfg):
        unknown = set(cfg) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown config keys {sorted(unknown)}")
        self.cfg = dict(self.defaults, **cfg)

    def setup(self, model, args):
        pass

    def wrap_train_loader(self, dl, args):
        return dl

    def wrap_eval_loader(self, dl, args):
        return dl

    def on_epoch_start(self, model, epoch):
        pass

    def after_step(self, model, batch, step):
        pass

    def loss_fn(self, model, reps):
        return upstream_loss(model, reps)

    extra_loss = None

    def final_scores(self, sims, model=None, vids=None, texts=None):
        return sims["fusion"]


_REGISTRY = {"baseline": Method}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get(name, cfg):
    import method_impls  # noqa: F401  (registers non-baseline methods)
    return _REGISTRY[name](cfg)
