#!/usr/bin/env python3

import argparse
import base64
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import requests


TURN_THRESH_DEG = 45.0
DIST_PER_SEG = 4.0
MIN_SEG_FRAMES = 24


def load_episode(raw_root: Path, scene_id: str, episode_index: int):
    ep_dir = raw_root / scene_id / "raw" / f"episode_{episode_index:06d}"
    meta = json.loads((ep_dir / "meta.json").read_text())
    frames = json.loads((ep_dir / "frames.json").read_text())
    return ep_dir, meta, frames


def yaw_from_pose(pose):
    pose = np.array(pose, dtype=np.float32)
    forward = -pose[:3, 2]
    return math.degrees(math.atan2(float(forward[0]), -float(forward[2])))


def path_distance(p0, p1):
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    return float(np.linalg.norm((p1 - p0)[[0, 2]]))


def choose_keyframes(frames):
    keyframes = [0]
    acc_dist = 0.0
    last_yaw = yaw_from_pose(frames[0]["pose.125cm_30deg"])
    for i in range(1, len(frames)):
        acc_dist += path_distance(
            np.array(frames[i - 1]["pose.125cm_30deg"])[:3, 3],
            np.array(frames[i]["pose.125cm_30deg"])[:3, 3],
        )
        cur_yaw = yaw_from_pose(frames[i]["pose.125cm_30deg"])
        yaw_diff = abs(((cur_yaw - last_yaw + 180) % 360) - 180)
        # Prefer splitting on major geometric changes; long straight segments
        # should stay merged unless they become very long.
        if yaw_diff >= TURN_THRESH_DEG or acc_dist >= DIST_PER_SEG:
            keyframes.append(i)
            acc_dist = 0.0
            last_yaw = cur_yaw
    if keyframes[-1] != len(frames) - 1:
        keyframes.append(len(frames) - 1)
    return merge_short_segments(keyframes, len(frames))


def merge_short_segments(keyframes, num_frames):
    if len(keyframes) <= 2:
        return keyframes
    merged = [keyframes[0]]
    for kf in keyframes[1:-1]:
        if kf - merged[-1] < MIN_SEG_FRAMES:
            continue
        merged.append(kf)
    if num_frames - 1 - merged[-1] < MIN_SEG_FRAMES and len(merged) > 1:
        return merged + [num_frames - 1]
    return merged + [num_frames - 1]


def build_segments(frames, keyframes):
    segments = []
    for s, e in zip(keyframes[:-1], keyframes[1:]):
        start_pose = np.array(frames[s]["pose.125cm_30deg"], dtype=np.float32)
        end_pose = np.array(frames[e]["pose.125cm_30deg"], dtype=np.float32)
        dist = path_distance(start_pose[:3, 3], end_pose[:3, 3])
        yaw_s = yaw_from_pose(start_pose)
        yaw_e = yaw_from_pose(end_pose)
        yaw_diff = ((yaw_e - yaw_s + 180) % 360) - 180
        action_hint = "go straight"
        if yaw_diff > TURN_THRESH_DEG:
            action_hint = "turn right and continue"
        elif yaw_diff < -TURN_THRESH_DEG:
            action_hint = "turn left and continue"
        segments.append(
            {
                "start_frame": s,
                "end_frame": e,
                "distance_m": round(dist, 2),
                "yaw_start_deg": round(yaw_s, 1),
                "yaw_end_deg": round(yaw_e, 1),
                "yaw_change_deg": round(yaw_diff, 1),
                "action_hint": action_hint,
            }
        )
    return segments


def encode_image(path: Path):
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def build_segment_contact(frames, ep_dir: Path, segment, out_path: Path):
    idxs = [segment["start_frame"], (segment["start_frame"] + segment["end_frame"]) // 2, segment["end_frame"]]
    imgs = []
    for idx in idxs:
        img = cv2.imread(frames[idx]["rgb_path"])
        if img is None:
            continue
        img = cv2.resize(img, (320, 240))
        cv2.putText(img, f"frame {idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        imgs.append(img)
    if not imgs:
        raise RuntimeError(f"Failed to load frames for segment {segment}")
    canvas = np.concatenate(imgs, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def call_openai(api_key, base_url, model, messages, temperature=0.2):
    url = base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_json_like(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def segment_prompt(segment):
    return (
        "You are labeling a short indoor navigation sub-clip.\n"
        "Write one concise navigation instruction in English for this sub-clip only.\n"
        "Requirements:\n"
        "- imperative mood\n"
        "- mention turning only if needed\n"
        "- if the action hint says 'turn right and continue' or 'turn left and continue', that turn direction is authoritative and must be preserved exactly\n"
        "- describe visible landmarks when possible\n"
        "- use left/right only when it is visually unambiguous from the clip; otherwise avoid side claims\n"
        "- mention color, material, or shape cues if they are visually salient\n"
        "- a sub-clip may describe progress without stopping; do not force a stop unless visually appropriate\n"
        "- keep it under 28 words\n"
        "- do not mention frame numbers\n"
        f"- geometric hint: distance {segment['distance_m']}m, heading change {segment['yaw_change_deg']}deg, action hint {segment['action_hint']}\n"
        "Return a JSON object with exactly these keys:\n"
        '{"sub_instruction": "...", "revised_sub_instruction": "..."}\n'
        "Where:\n"
        "- sub_instruction is plain, direct, and instructional\n"
        "- revised_sub_instruction is more fluent and descriptive, but still faithful\n"
        "- revised_sub_instruction should sound like high-quality VLN data: natural, vivid, but not poetic or exaggerated\n"
        "- do not invent left/right relations unless they are clearly supported by the visuals\n"
        "- never flip the turn direction implied by the geometric hint\n"
        "- avoid repeating the exact phrase 'straight ahead' if a landmark-based wording is better\n"
    )


def summarize_prompt(instructions):
    joined = "\n".join([f"{i+1}. {x}" for i, x in enumerate(instructions)])
    return (
        "You are writing a single long-horizon VLN instruction from short sub-instructions.\n"
        "Combine them into one fluent navigation instruction in English.\n"
        "Requirements:\n"
        "- preserve ordering\n"
        "- keep it natural and concise\n"
        "- use visible landmarks and left/right relations only when they are clearly supported by the sub-instructions\n"
        "- preserve explicit turn directions from the sub-instructions exactly; never swap left and right\n"
        "- mention the terminal stop condition clearly\n"
        "- aggressively merge consecutive straight-walk segments when they describe the same corridor/room progression\n"
        "- remove repeated landmarks, repeated floor-pattern mentions, and repeated forward-motion phrases unless they add new navigation value\n"
        "- keep only the key turns, the most informative landmarks, and the final stop target\n"
        "- vary wording naturally; avoid repeating the same verb in every clause\n"
        "- prefer grounded visual cues like furniture, doors, rugs, stairs, walls, colors, or textures\n"
        "- do not add new side relations that were not clearly established earlier\n"
        "- the final sentence MUST end with an explicit stop target or stopping condition\n"
        "- no numbering\n"
        "- no meta commentary\n"
        'Return a JSON object: {"sum_instruction": "..."}\n\n'
        f"Sub-instructions:\n{joined}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    raw_root = Path(args.raw_root)
    ep_dir, meta, frames = load_episode(raw_root, args.scene_id, args.episode_index)

    keyframes = choose_keyframes(frames)
    segments = build_segments(frames, keyframes)

    segment_instructions = []
    contact_paths = []
    for idx, seg in enumerate(segments):
        contact_path = ep_dir / "instruction_assets" / f"segment_{idx:02d}.jpg"
        build_segment_contact(frames, ep_dir, seg, contact_path)
        contact_paths.append(str(contact_path))
        image_b64 = encode_image(contact_path)
        messages = [
            {
                "role": "system",
                "content": "You generate short indoor navigation instructions from trajectory snippets.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": segment_prompt(seg)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]
        raw = call_openai(api_key, base_url, model, messages)
        parsed = parse_json_like(raw)
        segment_instructions.append(parsed)

    long_instruction = call_openai(
        api_key,
        base_url,
        model,
        [
            {
                "role": "system",
                "content": "You write fluent long-horizon indoor navigation instructions.",
            },
            {
                "role": "user",
                "content": summarize_prompt([x["revised_sub_instruction"] for x in segment_instructions]),
            },
        ],
    )
    long_instruction = parse_json_like(long_instruction)["sum_instruction"]

    result = {
        "scene_id": args.scene_id,
        "episode_index": args.episode_index,
        "original_instruction": meta.get("instruction", ""),
        "keyframes": keyframes,
        "segments": segments,
        "segment_contact_sheets": contact_paths,
        "fine_grained_instructions": [x["sub_instruction"] for x in segment_instructions],
        "revised_fine_grained_instructions": [x["revised_sub_instruction"] for x in segment_instructions],
        "long_instruction": long_instruction,
        "model": model,
        "base_url": base_url,
    }

    output = Path(args.output) if args.output else ep_dir / "generated_instructions.json"
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
