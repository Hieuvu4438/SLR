from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_protocol_artifact(path: str | Path | None) -> dict[str, object]:
    if path is None or not Path(path).is_file():
        return {"status": "official_artifact_missing", "metrics": None}
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"query_ids", "candidates", "substitution_policy", "seed", "candidate_count"}
    missing = required - set(value)
    if missing:
        return {"status": "invalid_protocol_artifact", "missing": sorted(missing), "metrics": None}
    return {
        "status": "protocol_artifact_ready",
        "query_count": len(value["query_ids"]),
        "metrics": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an official SAN fine-grained protocol artifact"
    )
    parser.add_argument("--protocol")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = inspect_protocol_artifact(args.protocol)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "protocol_artifact_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
