from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from elsc.config import dump_resolved, load_config
from elsc.data.cico_dataset import CiCoFeatureDataset
from elsc.data.tokenize import CiCoCollator
from elsc.resources import require_resources
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer
from elsc.utils import atomic_json_dump, sha256_file


def _resolve_checkpoint(run_dir: Path, value: str) -> Path:
    return (
        run_dir / "checkpoints" / ({"best_dev": "best_dev.pt", "last": "last.pt"}.get(value, value))
    )


@torch.no_grad()
def _fixture_score(model, batch, device: torch.device, dual_mix: float) -> torch.Tensor:
    model.eval()
    video, _ = model.encode_video(batch["h"].to(device), batch["valid"].to(device))
    text = model.encode_text(*(value.to(device) for value in batch["clean_text"]))
    i2t, t2i = model.bridge.score(video, text, objective=True)
    return model.bridge.mixed_score(i2t, t2i, dual_mix).cpu()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export inference-only CiCo core + ELSC adapter")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(run_dir / "resolved_config.yaml", stage="export")
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    device = torch.device(args.device)
    resources = config.get("resources", {})
    require_resources(
        output.parent,
        device,
        min_disk_gib=float(resources.get("min_free_disk_gib", 20)),
        min_gpu_gib=float(resources.get("min_free_gpu_gib_evaluation", 4)),
        operation="inference export",
    )
    model, _ = build_retriever_from_checkpoint(config, checkpoint, device=device)
    inference_state = {
        key: value for key, value in model.state_dict().items() if not key.startswith("local_head.")
    }
    export_path = output / "model.pt"
    torch.save(
        {
            "format": "elsc-inference-v1",
            "model": inference_state,
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "adapter_enabled": model.adapter_enabled,
        },
        export_path,
    )
    dump_resolved(config, output / "resolved_config.yaml")
    bpe_source = Path(config["upstream"]["cico_root"]) / "modules" / "bpe_simple_vocab_16e6.txt.gz"
    shutil.copy2(bpe_source, output / bpe_source.name)

    dataset = CiCoFeatureDataset(
        config["data"]["dev_manifest"],
        feature_len=int(config["data"]["feature_len"]),
        alpha=float(config["data"]["alpha"]),
        split="dev",
    )
    tokenizer = load_cico_tokenizer(config)
    loader = DataLoader(
        dataset,
        batch_size=min(4, len(dataset)),
        shuffle=False,
        collate_fn=CiCoCollator(tokenizer, int(config["data"]["max_words"]), augment=False),
    )
    fixture = next(iter(loader))
    before = _fixture_score(model, fixture, device, float(config["model"]["dual_mix"]))
    reloaded, _ = build_retriever_from_checkpoint(config, export_path, device=device)
    after = _fixture_score(reloaded, fixture, device, float(config["model"]["dual_mix"]))
    maximum_error = float((before - after).abs().max())
    if not torch.allclose(before, after, atol=1e-6, rtol=1e-5):
        raise RuntimeError(f"export score parity failed; max_abs_error={maximum_error}")
    report = {
        "schema_version": 1,
        "format": "elsc-inference-v1",
        "model_sha256": sha256_file(export_path),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "excluded_training_modules": ["local_head", "teacher", "lexical_bank", "support_cache"],
        "parity": {"status": "passed", "max_abs_error": maximum_error, "atol": 1e-6, "rtol": 1e-5},
    }
    atomic_json_dump(report, output / "export_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
