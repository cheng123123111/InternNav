import argparse
import json
from pathlib import Path


def make_key(row: dict) -> tuple:
    return (
        str(row.get("source_tar", "")),
        str(row.get("member_name", "")),
        int(row.get("label", -1)),
        str(row.get("sample_type", "")),
    )


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Merge stop manifests with deduplication.")
    parser.add_argument("--base", required=True, help="Existing base manifest.")
    parser.add_argument("--incoming", required=True, help="Incoming manifest to merge.")
    parser.add_argument("--output", required=True, help="Merged output manifest.")
    args = parser.parse_args()

    base_path = Path(args.base)
    incoming_path = Path(args.incoming)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = []
    seen = set()

    base_rows = load_rows(base_path)
    for row in base_rows:
        key = make_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)

    incoming_rows = load_rows(incoming_path)
    added = 0
    for row in incoming_rows:
        key = make_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        added += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "base": str(base_path),
                "incoming": str(incoming_path),
                "output": str(output_path),
                "base_rows": len(base_rows),
                "incoming_rows": len(incoming_rows),
                "merged_rows": len(merged),
                "added_rows": added,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
