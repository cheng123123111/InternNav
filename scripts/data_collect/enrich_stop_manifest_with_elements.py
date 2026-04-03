import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.data_collect.stop_alignment_utils import (
    extract_relation_hint,
    extract_stop_action_text,
    extract_stop_target_elements,
)


def main():
    parser = argparse.ArgumentParser(description='Add target element fields to an existing stop manifest.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-elements', type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    usable = 0
    relation_nonempty = 0
    with input_path.open('r', encoding='utf-8') as fin, output_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instruction = row.get('instruction') or row.get('episode_instruction') or ''
            target_elements = extract_stop_target_elements(instruction, max_elements=args.max_elements)
            relation_hint = extract_relation_hint((row.get('stop_object_phrase') or row.get('stop_phrase') or '').strip())
            stop_action_text = extract_stop_action_text(instruction)
            row['target_elements'] = target_elements
            row['target_element_count'] = len(target_elements)
            row['target_elements_text'] = ' ; '.join(target_elements)
            row['relation_hint'] = relation_hint
            row['stop_action_text'] = stop_action_text
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
            usable += int(len(target_elements) > 0)
            relation_nonempty += int(bool(relation_hint))

    print(json.dumps({
        'input': str(input_path),
        'output': str(output_path),
        'rows': count,
        'rows_with_target_elements': usable,
        'rows_with_relation_hint': relation_nonempty,
        'element_coverage': usable / max(count, 1),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
