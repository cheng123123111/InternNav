#!/usr/bin/env python3

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import quaternion as qt

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
from habitat_sim.utils.settings import default_sim_settings, make_cfg


TURN_DEG = 15.0
FORWARD_STEP = 0.05
GOAL_TOL = 0.35
WAYPOINT_TOL = 0.20
GOAL_HORIZON = 16
HFOV_DEG = 90.0
MIN_GEODESIC = 2.0
MAX_GEODESIC = 6.0


def build_sim(scene_path: str, width: int = 640, height: int = 480):
    settings = default_sim_settings.copy()
    settings["scene"] = scene_path
    settings["width"] = width
    settings["height"] = height
    settings["sensor_height"] = 1.25
    settings["color_sensor"] = True
    settings["depth_sensor"] = True
    settings["enable_physics"] = False
    cfg = make_cfg(settings)

    # Add a 30 degree look-down RGB/depth pair to better match N1-style settings.
    agent_cfg = cfg.agents[0]
    rgb30 = habitat_sim.CameraSensorSpec()
    rgb30.uuid = "rgb_125cm_30deg"
    rgb30.sensor_type = habitat_sim.SensorType.COLOR
    rgb30.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb30.resolution = [height, width]
    rgb30.position = [0.0, 1.25, 0.0]
    rgb30.orientation = [math.radians(-30.0), 0.0, 0.0]

    depth30 = habitat_sim.CameraSensorSpec()
    depth30.uuid = "depth_125cm_30deg"
    depth30.sensor_type = habitat_sim.SensorType.DEPTH
    depth30.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth30.resolution = [height, width]
    depth30.position = [0.0, 1.25, 0.0]
    depth30.orientation = [math.radians(-30.0), 0.0, 0.0]

    agent_cfg.sensor_specifications.extend([rgb30, depth30])
    return habitat_sim.Simulator(cfg)


def random_start_goal(
    pathfinder,
    min_geodesic: float = MIN_GEODESIC,
    max_geodesic: float = MAX_GEODESIC,
    max_trials: int = 200,
):
    for _ in range(max_trials):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if (
            pathfinder.find_path(path)
            and path.geodesic_distance >= min_geodesic
            and path.geodesic_distance <= max_geodesic
        ):
            return start, goal, np.array(path.points)
    raise RuntimeError("Failed to sample a valid start-goal pair.")


def yaw_to_quat(yaw_rad: float):
    return quat_from_angle_axis(yaw_rad, np.array([0.0, 1.0, 0.0], dtype=np.float32))


def quat_to_yaw(q):
    forward = quat_rotate_vector(q, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    return math.atan2(forward[0], -forward[2])


def wrap_angle(rad: float):
    return (rad + math.pi) % (2 * math.pi) - math.pi


def pose_matrix_from_sensor_state(sensor_state):
    rot = habitat_sim.utils.common.quat_to_magnum(sensor_state.rotation).to_matrix()
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.array(rot)
    mat[:3, 3] = np.array(sensor_state.position, dtype=np.float32)
    return mat


def project_world_to_pixel(camera_matrix, projection_matrix, world_point, width, height):
    point = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float32)
    clip = projection_matrix @ (camera_matrix @ point)
    w = float(clip[3])
    if abs(w) <= 1e-6:
        return None
    ndc = clip[:3] / w
    u = int((float(ndc[0]) * 0.5 + 0.5) * width)
    v = int((1.0 - (float(ndc[1]) * 0.5 + 0.5)) * height)
    if not (0 <= u < width and 0 <= v < height):
        return None
    return [u, v]


def annotate_future_goals(frames, width, height):
    for i in range(len(frames)):
        best_delta = -1
        best_goal = [-1, -1]
        camera_matrix = np.array(frames[i]["camera_matrix.125cm_30deg"], dtype=np.float32)
        projection_matrix = np.array(frames[i]["projection_matrix.125cm_30deg"], dtype=np.float32)
        max_lookahead = min(GOAL_HORIZON, len(frames) - i - 1)
        for delta in range(max_lookahead, 0, -1):
            target_point = np.array(frames[i + delta]["sensor_position"], dtype=np.float32)
            px = project_world_to_pixel(camera_matrix, projection_matrix, target_point, width, height)
            if px is not None:
                best_delta = delta
                best_goal = px
                break
        frames[i]["relative_goal_frame_id.125cm_30deg"] = int(best_delta)
        frames[i]["goal.125cm_30deg"] = best_goal


def step_forward(state, distance=FORWARD_STEP):
    delta = quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -distance], dtype=np.float32))
    state.position = state.position + delta
    return state


def turn_left(state, deg=TURN_DEG):
    state.rotation = quat_from_angle_axis(math.radians(deg), np.array([0.0, 1.0, 0.0], dtype=np.float32)) * state.rotation
    return state


def turn_right(state, deg=TURN_DEG):
    state.rotation = quat_from_angle_axis(math.radians(-deg), np.array([0.0, 1.0, 0.0], dtype=np.float32)) * state.rotation
    return state


def simplify_path(points: np.ndarray):
    if len(points) <= 2:
        return points
    kept = [points[0]]
    for i in range(1, len(points) - 1):
        prev_dir = points[i] - kept[-1]
        next_dir = points[i + 1] - points[i]
        if np.linalg.norm(prev_dir[:2]) < 1e-4 or np.linalg.norm(next_dir[:2]) < 1e-4:
            continue
        prev_yaw = math.atan2(prev_dir[0], prev_dir[2])
        next_yaw = math.atan2(next_dir[0], next_dir[2])
        if abs(wrap_angle(next_yaw - prev_yaw)) > math.radians(20):
            kept.append(points[i])
    kept.append(points[-1])
    return np.array(kept)


def synthesize_instruction(path_points: np.ndarray):
    simple = simplify_path(path_points)
    if len(simple) < 2:
        return "Go to the target point."
    pieces = []
    for i in range(1, min(len(simple), 5)):
        delta = simple[i] - simple[i - 1]
        dist = np.linalg.norm(delta[[0, 2]])
        if dist < 0.5:
            continue
        step = "Walk forward"
        if i < len(simple) - 1:
            nxt = simple[i + 1] - simple[i]
            yaw1 = math.atan2(delta[0], delta[2])
            yaw2 = math.atan2(nxt[0], nxt[2])
            diff = wrap_angle(yaw2 - yaw1)
            if diff > math.radians(25):
                step += ", then turn left"
            elif diff < -math.radians(25):
                step += ", then turn right"
        pieces.append(step)
    if not pieces:
        return "Go to the target point."
    return ". ".join(pieces) + "."


def resample_path(points: np.ndarray, step: float):
    if len(points) < 2:
        return points
    out = [points[0]]
    for i in range(1, len(points)):
        s = np.array(points[i - 1], dtype=np.float32)
        e = np.array(points[i], dtype=np.float32)
        seg = e - s
        dist = float(np.linalg.norm(seg))
        if dist < 1e-6:
            continue
        direction = seg / dist
        n = max(1, int(dist / step))
        for k in range(1, n + 1):
            p = s + direction * min(k * step, dist)
            out.append(p)
    return np.stack(out)


def collect_episode(
    sim,
    scene_id: str,
    episode_index: int,
    output_dir: Path,
    seed: int,
    forward_step: float,
    min_geodesic: float,
    max_geodesic: float,
):
    random.seed(seed)
    np.random.seed(seed)

    start, goal, path_points = random_start_goal(
        sim.pathfinder,
        min_geodesic=min_geodesic,
        max_geodesic=max_geodesic,
    )
    goal = np.array(goal, dtype=np.float32)
    instruction = synthesize_instruction(path_points)

    scene_root = output_dir / scene_id
    raw_ep = scene_root / "raw" / f"episode_{episode_index:06d}"
    rgb_dir = raw_ep / "rgb_125cm_30deg"
    depth_dir = raw_ep / "depth_125cm_30deg"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    path_points = resample_path(path_points, step=forward_step)
    agent = sim.initialize_agent(0)
    frames = []
    max_steps = min(400, len(path_points))

    for step_idx in range(max_steps):
        state = habitat_sim.AgentState()
        state.position = np.array(path_points[step_idx], dtype=np.float32)
        nxt = path_points[min(step_idx + 1, len(path_points) - 1)]
        prv = path_points[max(step_idx - 1, 0)]
        direction = nxt - state.position if step_idx < len(path_points) - 1 else state.position - prv
        if np.linalg.norm(direction[[0, 2]]) < 1e-6:
            yaw = 0.0
        else:
            # Habitat's canonical forward direction is -Z at yaw=0.
            yaw = math.atan2(-direction[0], -direction[2])
        state.rotation = yaw_to_quat(yaw)
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        state = agent.get_state()
        sensor_state = state.sensor_states["rgb_125cm_30deg"]
        render_camera = agent._sensors["rgb_125cm_30deg"].render_camera
        pose = pose_matrix_from_sensor_state(sensor_state)
        rgb = cv2.cvtColor(obs["rgb_125cm_30deg"][..., :3], cv2.COLOR_RGB2BGR)
        depth = obs["depth_125cm_30deg"]

        rgb_path = rgb_dir / f"{episode_index:06d}_{step_idx:03d}.jpg"
        depth_path = depth_dir / f"{episode_index:06d}_{step_idx:03d}.png"
        cv2.imwrite(str(rgb_path), rgb)
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)

        action = 1
        rel_goal_frame = -1
        goal_px = [-1, -1]

        pos = np.array([state.position[0], state.position[1], state.position[2]], dtype=np.float32)
        if step_idx == len(path_points) - 1 or np.linalg.norm(pos[[0, 2]] - goal[[0, 2]]) <= GOAL_TOL:
            action = 0

        frames.append(
            {
                "frame_index": step_idx,
                "timestamp": step_idx / 4.0,
                "action": int(action),
                "pose.125cm_30deg": pose.tolist(),
                "goal.125cm_30deg": goal_px,
                "relative_goal_frame_id.125cm_30deg": int(rel_goal_frame),
                "sensor_position": np.array(sensor_state.position, dtype=np.float32).tolist(),
                "camera_matrix.125cm_30deg": np.array(render_camera.camera_matrix, dtype=np.float32).tolist(),
                "projection_matrix.125cm_30deg": np.array(render_camera.projection_matrix, dtype=np.float32).tolist(),
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
            }
        )

        if step_idx < len(path_points) - 2:
            next_dir = path_points[step_idx + 2] - path_points[step_idx + 1]
            next_yaw = math.atan2(next_dir[0], next_dir[2]) if np.linalg.norm(next_dir[[0, 2]]) > 1e-6 else yaw
            diff = wrap_angle(next_yaw - yaw)
            if diff > math.radians(15):
                frames[-1]["action"] = 2
            elif diff < -math.radians(15):
                frames[-1]["action"] = 3

        if action == 0:
            break

    annotate_future_goals(frames, rgb.shape[1], rgb.shape[0])

    for fr in frames:
        fr.pop("sensor_position", None)
        fr.pop("camera_matrix.125cm_30deg", None)
        fr.pop("projection_matrix.125cm_30deg", None)

    raw_meta = {
        "episode_index": episode_index,
        "scene_id": scene_id,
        "scene_path": sim.config.sim_cfg.scene_id,
        "instruction": instruction,
        "goal_point": np.array(goal).tolist(),
        "path_points": np.array(path_points).tolist(),
        "length": len(frames),
    }
    (raw_ep / "meta.json").write_text(json.dumps(raw_meta, indent=2))
    (raw_ep / "frames.json").write_text(json.dumps(frames, indent=2))
    return raw_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-step", type=float, default=FORWARD_STEP)
    parser.add_argument("--min-geodesic", type=float, default=MIN_GEODESIC)
    parser.add_argument("--max-geodesic", type=float, default=MAX_GEODESIC)
    args = parser.parse_args()

    sim = build_sim(args.scene)
    scene_id = Path(args.scene).stem
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    try:
        for ep_idx in range(args.episodes):
            summaries.append(
                collect_episode(
                    sim,
                    scene_id,
                    ep_idx,
                    out,
                    args.seed + ep_idx,
                    forward_step=args.forward_step,
                    min_geodesic=args.min_geodesic,
                    max_geodesic=args.max_geodesic,
                )
            )
    finally:
        sim.close()

    summary_path = out / scene_id / "raw" / "collection_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
