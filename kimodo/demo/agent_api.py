# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-safe control helpers for the Kimodo demo.

These functions expose the domain operations an orchestration process needs
without depending on GUI button callbacks or browser UI automation. A browser
client may still be connected as the WebGL render surface for Viser, but all
state changes here happen through Python handles owned by the demo session.
"""

from __future__ import annotations

import math
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kimodo.constraints import load_constraints_lst, save_constraints_lst
from kimodo.exports.bvh import save_motion_bvh
from kimodo.exports.motion_io import (
    amass_npz_to_bytes,
    g1_csv_to_bytes,
    load_motion_file,
    save_kimodo_npz,
)
from kimodo.model.registry import kimodo_short_key_for_skeleton_dataset, registry_skeleton_for_joint_count
from kimodo.skeleton import G1Skeleton34, SOMASkeleton30, SOMASkeleton77

from . import generation
from .config import get_model_info
from .state import ClientSession


CAMERA_PRESETS: dict[str, dict[str, list[float] | float]] = {
    "front": {
        "position": [0.0, 1.7, 6.0],
        "look_at": [0.0, 0.9, 0.0],
        "up_direction": [0.0, 1.0, 0.0],
        "fov_degrees": 45.0,
    },
    "back": {
        "position": [0.0, 1.7, -6.0],
        "look_at": [0.0, 0.9, 0.0],
        "up_direction": [0.0, 1.0, 0.0],
        "fov_degrees": 45.0,
    },
    "side": {
        "position": [6.0, 1.7, 0.0],
        "look_at": [0.0, 0.9, 0.0],
        "up_direction": [0.0, 1.0, 0.0],
        "fov_degrees": 45.0,
    },
    "left": {
        "position": [-6.0, 1.7, 0.0],
        "look_at": [0.0, 0.9, 0.0],
        "up_direction": [0.0, 1.0, 0.0],
        "fov_degrees": 45.0,
    },
    "iso": {
        "position": [3.8, 2.4, 6.0],
        "look_at": [0.0, 0.9, 0.0],
        "up_direction": [0.0, 1.0, 0.0],
        "fov_degrees": 45.0,
    },
    "top": {
        "position": [0.0, 8.0, 0.01],
        "look_at": [0.0, 0.0, 0.0],
        "up_direction": [0.0, 0.0, -1.0],
        "fov_degrees": 45.0,
    },
}

WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
WORLD_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float64)
WORLD_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_REVIEW_FRAME_CAP_SECONDS = 60
DEFAULT_REVIEW_FPS = 30
DEFAULT_REVIEW_FRAME_CAP = DEFAULT_REVIEW_FRAME_CAP_SECONDS * DEFAULT_REVIEW_FPS


def _as_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(x) for x in value]


def _rounded_list(value: Any, *, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(value, dtype=np.float64).tolist()]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_vector(value: Any, fallback: Any) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8 or not np.isfinite(norm):
        vec = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
    return vec / norm


def _camera_axes(view_dir: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera right, up, and forward axes for a camera looking at center."""

    camera_forward = -_normalize_vector(view_dir, WORLD_FORWARD)
    right = np.cross(camera_forward, up_hint)
    if float(np.linalg.norm(right)) <= 1e-8:
        right = np.cross(camera_forward, WORLD_RIGHT)
    right = _normalize_vector(right, WORLD_RIGHT)
    camera_up = _normalize_vector(np.cross(right, camera_forward), WORLD_UP)
    return right, camera_up, camera_forward


def _selected_frame_indices(frame_indices: list[int] | None, frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Cannot fit cameras for an empty motion.")
    if frame_indices is None:
        return list(range(frame_count))
    selected = [int(idx) for idx in frame_indices if 0 <= int(idx) < frame_count]
    if not selected:
        raise ValueError("No valid frame indices selected for camera fitting.")
    return selected


def _static_fit_view_axes(view: str, forward: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return view direction from look-at center to camera and camera up hint."""

    if view == "top":
        return WORLD_UP, -forward

    elevation = 0.16
    if view == "front":
        horizontal = forward
        view_dir = horizontal + WORLD_UP * elevation
    elif view == "back":
        horizontal = -forward
        view_dir = horizontal + WORLD_UP * elevation
    elif view == "side":
        horizontal = right
        view_dir = horizontal + WORLD_UP * elevation
    elif view == "left":
        horizontal = -right
        view_dir = horizontal + WORLD_UP * elevation
    elif view == "iso":
        view_dir = forward + right * 0.65 + WORLD_UP * 0.35
    else:
        raise KeyError(f"Unknown static-fit camera view '{view}'. Expected one of: {sorted(CAMERA_PRESETS)}")
    return _normalize_vector(view_dir, forward), WORLD_UP


def _fit_camera_to_points(
    *,
    view: str,
    points: np.ndarray,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    width: int,
    height: int,
    margin: float,
    fov_degrees: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    view_dir, up_hint = _static_fit_view_axes(view, forward, right)
    camera_right, camera_up, _camera_forward = _camera_axes(view_dir, up_hint)
    rel = points - center
    tan_y = math.tan(math.radians(float(fov_degrees)) / 2.0)
    tan_x = tan_y * (float(width) / float(height))
    depth_offsets = rel @ view_dir
    required_x = depth_offsets + np.abs(rel @ camera_right) / tan_x
    required_y = depth_offsets + np.abs(rel @ camera_up) / tan_y
    required_distance = float(np.max(np.maximum(required_x, required_y)))
    extent = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    distance = max(required_distance * margin, extent * 0.65, 1.5)
    position = center + view_dir * distance
    far = max(1000.0, distance + extent * 4.0 + 10.0)
    camera = {
        "position": _rounded_list(position),
        "look_at": _rounded_list(center),
        "up_direction": _rounded_list(camera_up),
        "fov_degrees": float(fov_degrees),
        "near": 0.01,
        "far": float(round(far, 6)),
    }
    metadata = {
        "distance": float(round(distance, 6)),
        "view_direction": _rounded_list(view_dir),
        "camera_right": _rounded_list(camera_right),
        "camera_up": _rounded_list(camera_up),
    }
    return camera, metadata


def plan_static_fit_cameras(
    *,
    joints_pos: Any,
    root_index: int = 0,
    frame_indices: list[int] | None = None,
    views: list[str] | None = None,
    width: int = 1280,
    height: int = 720,
    camera_margin: float = 1.2,
    camera_orientation: str = "trajectory",
    camera_fov_degrees: float = 45.0,
    camera_min_displacement_m: float = 0.35,
) -> dict[str, Any]:
    """Plan deterministic static cameras that fit all selected motion joints."""

    selected_views = ["iso", "front", "side"] if views is None else views
    if camera_margin <= 0:
        raise ValueError("camera_margin must be positive.")
    if width <= 0 or height <= 0:
        raise ValueError("Camera fitting requires a positive render resolution.")

    joints = _as_numpy(joints_pos).astype(np.float64)
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = joints[0]
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints_pos with shape [T, J, 3], got {joints.shape}.")
    selected_indices = _selected_frame_indices(frame_indices, int(joints.shape[0]))
    selected_joints = joints[selected_indices]
    points = selected_joints.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        raise ValueError("No finite joint positions available for camera fitting.")

    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    center = (bounds_min + bounds_max) / 2.0
    root_idx = max(0, min(int(root_index), int(joints.shape[1]) - 1))
    root_positions = selected_joints[:, root_idx, :]
    root_positions = root_positions[np.isfinite(root_positions).all(axis=1)]
    root_start = root_positions[0] if root_positions.size else center
    root_end = root_positions[-1] if root_positions.size else center
    displacement = np.array([root_end[0] - root_start[0], 0.0, root_end[2] - root_start[2]], dtype=np.float64)
    displacement_m = float(np.linalg.norm(displacement))

    if camera_orientation == "trajectory" and displacement_m >= float(camera_min_displacement_m):
        forward = _normalize_vector(displacement, WORLD_FORWARD)
        orientation_source = "trajectory"
    elif camera_orientation in {"trajectory", "world"}:
        forward = WORLD_FORWARD.copy()
        orientation_source = "world" if camera_orientation == "world" else "world_fallback"
    else:
        raise ValueError("camera_orientation must be 'trajectory' or 'world'.")
    right = _normalize_vector(np.cross(WORLD_UP, forward), WORLD_RIGHT)

    cameras: dict[str, Any] = {}
    for view in selected_views:
        if view == "current":
            continue
        camera, metadata = _fit_camera_to_points(
            view=view,
            points=points,
            center=center,
            forward=forward,
            right=right,
            width=int(width),
            height=int(height),
            margin=float(camera_margin),
            fov_degrees=float(camera_fov_degrees),
        )
        cameras[view] = {"camera": camera, **metadata}

    return {
        "mode": "static_fit",
        "orientation": camera_orientation,
        "orientation_source": orientation_source,
        "margin": float(camera_margin),
        "fov_degrees": float(camera_fov_degrees),
        "min_displacement_m": float(camera_min_displacement_m),
        "resolution": {"width": int(width), "height": int(height)},
        "frames": {
            "count": len(selected_indices),
            "first": int(selected_indices[0]),
            "last": int(selected_indices[-1]),
        },
        "bounds": {
            "min": _rounded_list(bounds_min),
            "max": _rounded_list(bounds_max),
            "center": _rounded_list(center),
        },
        "root": {
            "start": _rounded_list(root_start),
            "end": _rounded_list(root_end),
            "displacement_m": float(round(displacement_m, 6)),
            "forward": _rounded_list(forward),
            "right": _rounded_list(right),
        },
        "views": cameras,
    }


def _coerce_client_id(client_or_id: Any | None) -> int | None:
    if client_or_id is None:
        return None
    if hasattr(client_or_id, "client_id"):
        return int(client_or_id.client_id)
    return int(client_or_id)


def get_session(demo: Any, client_or_id: Any | None = None) -> ClientSession:
    """Return an active session, defaulting to the first connected client."""

    client_id = _coerce_client_id(client_or_id)
    if client_id is None:
        if not demo.client_sessions:
            raise RuntimeError("No active Kimodo client session is connected.")
        client_id = next(iter(demo.client_sessions.keys()))
    if client_id not in demo.client_sessions:
        raise RuntimeError(f"Kimodo client session {client_id} is not active.")
    return demo.client_sessions[client_id]


def wait_for_client(
    demo: Any,
    *,
    client_id: int | None = None,
    timeout_s: float = 60.0,
    poll_s: float = 0.1,
) -> dict[str, Any]:
    """Wait until a browser render client has created a Kimodo session."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if client_id is None and demo.client_sessions:
            session_id = next(iter(demo.client_sessions.keys()))
            return get_session_state(demo, session_id)
        if client_id is not None and client_id in demo.client_sessions:
            return get_session_state(demo, client_id)
        time.sleep(poll_s)
    raise TimeoutError(f"No Kimodo client connected within {timeout_s:.1f}s.")


def get_primary_motion(session: ClientSession):
    if not session.motions:
        raise RuntimeError("Session has no loaded motion.")
    return list(session.motions.values())[0]


def motion_to_numpy_dict(motion) -> dict[str, np.ndarray]:
    """Convert the primary Kimodo in-memory motion to export arrays."""

    joints_pos = motion.joints_pos.detach().cpu().numpy()
    joints_rot = motion.joints_rot.detach().cpu().numpy()
    joints_local_rot = motion.joints_local_rot.detach().cpu().numpy()

    if joints_pos.ndim != 3:
        raise ValueError(f"Expected unbatched joints_pos with shape [T, J, 3], got {joints_pos.shape}")
    if joints_rot.ndim != 4:
        raise ValueError(f"Expected unbatched joints_rot with shape [T, J, 3, 3], got {joints_rot.shape}")
    if joints_local_rot.ndim != 4:
        raise ValueError(f"Expected unbatched joints_local_rot with shape [T, J, 3, 3], got {joints_local_rot.shape}")

    motion_data = {
        "posed_joints": joints_pos,
        "global_rot_mats": joints_rot,
        "local_rot_mats": joints_local_rot,
        "root_positions": joints_pos[:, motion.skeleton.root_idx, :],
    }
    if motion.foot_contacts is not None:
        foot_contacts = motion.foot_contacts.detach().cpu().numpy()
        if foot_contacts.ndim != 2:
            raise ValueError(f"Expected unbatched foot_contacts with shape [T, C], got {foot_contacts.shape}")
        motion_data["foot_contacts"] = foot_contacts
    return motion_data


def _coerce_output_path(output_path: str | Path, *, ext: str) -> Path:
    path = Path(output_path)
    known_exts = {".npz", ".bvh", ".csv", ".mp4", ".png"}
    if path.suffix == "":
        return path.with_suffix(ext)
    if path.suffix.lower() in known_exts:
        return path.with_suffix(ext)
    return path


def export_session_motion(
    demo: Any,
    *,
    client_id: int | None = None,
    output_path: str | Path,
    fmt: str = "NPZ",
    standard_tpose: bool = False,
) -> dict[str, Any]:
    """Export the active motion from a live Kimodo session."""

    session = get_session(demo, client_id)
    motion = get_primary_motion(session)
    fmt = fmt.upper()

    if fmt == "BVH":
        path = _coerce_output_path(output_path, ext=".bvh")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_motion_bvh(
            path,
            motion.joints_local_rot,
            motion.joints_pos[:, session.skeleton.root_idx, :],
            skeleton=session.skeleton,
            fps=float(session.model_fps),
            standard_tpose=standard_tpose,
        )
    elif fmt == "CSV":
        path = _coerce_output_path(output_path, ext=".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(g1_csv_to_bytes(motion_to_numpy_dict(motion), session.skeleton, demo.device))
    elif fmt == "AMASS NPZ":
        path = _coerce_output_path(output_path, ext=".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(amass_npz_to_bytes(motion_to_numpy_dict(motion), session.skeleton, float(session.model_fps)))
    else:
        path = _coerce_output_path(output_path, ext=".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kimodo_npz(str(path), motion_to_numpy_dict(motion))

    return {
        "path": str(path),
        "format": fmt,
        "frame_count": int(motion.joints_pos.shape[0]),
        "fps": float(session.model_fps),
    }


def switch_session_model(
    demo: Any,
    *,
    client_id: int | None = None,
    model_name: str,
    preserve_prompts: bool = True,
) -> dict[str, Any]:
    """Switch a live session to another loaded Kimodo model without UI callbacks."""

    session = get_session(demo, client_id)
    if model_name == session.model_name:
        return get_session_state(demo, session.client.client_id)

    client = session.client
    old_model_fps = float(session.model_fps or 0.0)
    old_prompts = []
    if preserve_prompts:
        old_prompts = [(prompt.text, prompt.start_frame, prompt.end_frame) for prompt in client.timeline._prompts.values()]

    model_bundle = demo.load_model(model_name)
    session.playing = False
    demo.clear_motions(client.client_id)
    with session.timeline_data["keyframe_update_lock"]:
        for constraint in list(session.constraints.values()):
            constraint.clear()
        session.constraints = demo.build_constraint_tracks(client, model_bundle.skeleton)
        session.timeline_data["keyframes"] = {}
        session.timeline_data["intervals"] = {}
        client.timeline.clear_keyframes()
        client.timeline.clear_intervals()

    session.model_name = model_name
    session.model_fps = model_bundle.model_fps
    session.skeleton = model_bundle.skeleton
    session.motion_rep = model_bundle.motion_rep
    session.max_frame_idx = int(session.cur_duration * session.model_fps - 1)
    session.frame_idx = 0
    session.edit_mode = False

    demo.set_timeline_defaults(client.timeline, session.model_fps)
    client.timeline.set_current_frame(0)
    client.timeline.clear_prompts()
    if old_prompts and old_model_fps > 0:
        for prompt_text, start_frame, end_frame in old_prompts:
            start_sec = start_frame / old_model_fps
            end_sec = end_frame / old_model_fps
            new_start = max(0, min(int(round(start_sec * session.model_fps)), session.max_frame_idx))
            new_end = max(new_start, min(int(round(end_sec * session.model_fps)), session.max_frame_idx))
            client.timeline.add_prompt(prompt_text, new_start, new_end)

    session.examples_base_dir = demo.get_examples_base_dir(model_name, absolute=True)
    demo.add_character_motion(client, session.skeleton)
    demo.set_frame(client.client_id, 0)
    return get_session_state(demo, client.client_id)


def load_session_motion(
    demo: Any,
    *,
    client_id: int | None = None,
    input_path: str | Path,
    target_fps: float | None = None,
    auto_switch_model: bool = True,
) -> dict[str, Any]:
    """Load a Kimodo-compatible motion file into a live session."""

    session = get_session(demo, client_id)
    fps_arg = target_fps if target_fps is not None else session.model_fps if session.model_fps and session.model_fps > 0 else None
    motion_dict, num_joints_motion = load_motion_file(str(input_path), target_fps=fps_arg)

    target_skel = registry_skeleton_for_joint_count(num_joints_motion)
    current_info = get_model_info(session.model_name)
    current_skel = current_info.skeleton if current_info is not None else None

    if current_skel != target_skel:
        if not auto_switch_model:
            raise ValueError(
                f"Loaded motion has skeleton {target_skel} (J={num_joints_motion}), "
                f"but active model {session.model_name} uses {current_skel}."
            )
        dataset = current_info.dataset if current_info is not None else "RP"
        new_key = kimodo_short_key_for_skeleton_dataset(target_skel, dataset)
        if new_key is None:
            new_key = kimodo_short_key_for_skeleton_dataset(target_skel, "RP")
        if new_key is None:
            raise ValueError(f"No Kimodo model found for skeleton {target_skel} (motion has J={num_joints_motion}).")
        switch_session_model(demo, client_id=session.client.client_id, model_name=new_key)
        session = get_session(demo, session.client.client_id)

    joints_pos = motion_dict["posed_joints"].to(device=demo.device, dtype=torch.float32)
    joints_rot = motion_dict["global_rot_mats"].to(device=demo.device, dtype=torch.float32)
    foot_contacts = motion_dict.get("foot_contacts")
    if foot_contacts is not None:
        foot_contacts = foot_contacts.to(device=demo.device, dtype=torch.float32)

    if joints_pos.ndim == 4:
        joints_pos = joints_pos[0]
    if joints_rot.ndim == 5:
        joints_rot = joints_rot[0]
    if foot_contacts is not None and foot_contacts.ndim == 3:
        foot_contacts = foot_contacts[0]

    num_joints_loaded = int(joints_pos.shape[1])
    num_joints_skeleton = int(session.skeleton.nbjoints)
    if num_joints_loaded != num_joints_skeleton:
        if num_joints_loaded == 30 and num_joints_skeleton == 77 and isinstance(session.skeleton, SOMASkeleton77):
            from kimodo.skeleton import global_rots_to_local_rots

            skel30 = SOMASkeleton30().to(demo.device)
            if "local_rot_mats" in motion_dict:
                local_rot_30 = motion_dict["local_rot_mats"].to(device=demo.device, dtype=torch.float32)
                if local_rot_30.ndim == 4:
                    local_rot_30 = local_rot_30[0]
            else:
                local_rot_30 = global_rots_to_local_rots(joints_rot, skel30)
            local_rot_77 = skel30.to_SOMASkeleton77(local_rot_30)
            root_positions = joints_pos[:, skel30.root_idx, :]
            joints_rot, joints_pos, _ = session.skeleton.fk(local_rot_77, root_positions)
            if foot_contacts is not None and foot_contacts.shape[-1] == 4:
                foot_contacts = torch.cat(
                    [
                        foot_contacts[..., :2],
                        foot_contacts[..., 1:2],
                        foot_contacts[..., 2:4],
                        foot_contacts[..., 3:4],
                    ],
                    dim=-1,
                )
        else:
            raise ValueError(
                f"The loaded motion has {num_joints_loaded} joints but the current model "
                f"({session.model_name}) has {num_joints_skeleton} joints."
            )
    elif int(joints_rot.shape[1]) != num_joints_skeleton:
        raise ValueError(
            f"Rotation data has {joints_rot.shape[1]} joints but the current model has {num_joints_skeleton} joints."
        )

    if (
        "g1" in session.model_name
        and isinstance(session.skeleton, G1Skeleton34)
        and bool(session.gui_elements.gui_real_robot_rotations_checkbox.value)
    ):
        joints_pos, joints_rot = generation.apply_g1_real_robot_projection(session.skeleton, joints_pos, joints_rot)

    num_frames = int(joints_pos.shape[0])
    session.cur_duration = num_frames / float(session.model_fps)
    session.max_frame_idx = num_frames - 1

    demo.clear_motions(session.client.client_id)
    demo.add_character_motion(session.client, session.skeleton, joints_pos, joints_rot, foot_contacts)
    demo.set_frame(session.client.client_id, 0)
    return get_session_state(demo, session.client.client_id)


def save_session_constraints(
    demo: Any,
    *,
    client_id: int | None = None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Serialize active session constraints using Kimodo's native schema."""

    session = get_session(demo, client_id)
    path = _coerce_output_path(output_path, ext=".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = demo.load_model(session.model_name)
    num_frames = session.max_frame_idx + 1
    constraints_lst = demo.compute_model_constraints_lst(session, model_bundle, num_frames)
    save_constraints_lst(str(path), constraints_lst)
    return {"path": str(path), "constraint_count": len(constraints_lst)}


def clear_session_constraints(demo: Any, *, client_id: int | None = None) -> dict[str, Any]:
    session = get_session(demo, client_id)
    with session.timeline_data["keyframe_update_lock"]:
        for constraint in list(session.constraints.values()):
            constraint.clear()
        session.client.timeline.clear_keyframes()
        session.client.timeline.clear_intervals()
        session.timeline_data["keyframes"] = {}
        session.timeline_data["intervals"] = {}
    demo._apply_constraint_overlay_visibility(session)
    return {"cleared": True}


def _extract_intervals_and_singles(frame_indices: Any):
    values = frame_indices.detach().cpu().tolist() if hasattr(frame_indices, "detach") else list(frame_indices)
    values = [int(v) for v in values]
    intervals = []
    intervals_indices = []
    single_frames = []
    single_frames_indices = []
    start_idx = 0

    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[i - 1] + 1:
            run_length = i - start_idx
            if run_length >= 2:
                intervals.append((values[start_idx], values[i - 1]))
                intervals_indices.append((start_idx, i - 1))
            else:
                single_frames.append(values[start_idx])
                single_frames_indices.append(start_idx)
            start_idx = i

    return intervals, intervals_indices, single_frames, single_frames_indices


def load_session_constraints(
    demo: Any,
    *,
    client_id: int | None = None,
    input_path: str | Path,
    replace: bool = True,
) -> dict[str, Any]:
    """Load Kimodo constraints directly into a live session."""

    session = get_session(demo, client_id)
    path = Path(input_path)
    constraints_lst = load_constraints_lst(str(path), skeleton=session.skeleton)

    if replace:
        clear_session_constraints(demo, client_id=session.client.client_id)

    device = demo.device
    keyframe_count = 0
    interval_count = 0

    for constraint_obj in constraints_lst:
        constraint_type = constraint_obj.name
        (
            intervals,
            intervals_indices,
            single_frames,
            single_frames_indices,
        ) = _extract_intervals_and_singles(constraint_obj.frame_indices)

        load_targets: list[dict[str, Any]] = []
        root_pos = None

        if constraint_type == "root2d":
            num_frames = int(constraint_obj.smooth_root_2d.shape[0])
            root_pos = torch.zeros(num_frames, 3, device=device)
            root_pos[:, 0] = constraint_obj.smooth_root_2d[:, 0]
            root_pos[:, 2] = constraint_obj.smooth_root_2d[:, 1]
            load_targets = [{"track_name": "2D Root", "constraint_track": session.constraints["2D Root"]}]
        elif constraint_type == "fullbody":
            load_targets = [{"track_name": "Full-Body", "constraint_track": session.constraints["Full-Body"]}]
        elif constraint_type in {"left-hand", "right-hand", "left-foot", "right-foot"}:
            track_name = {
                "left-hand": "Left Hand",
                "right-hand": "Right Hand",
                "left-foot": "Left Foot",
                "right-foot": "Right Foot",
            }[constraint_type]
            load_targets = [
                {
                    "track_name": track_name,
                    "constraint_track": session.constraints["End-Effectors"],
                    "joint_names": constraint_obj.joint_names,
                    "end_effector_type": constraint_type,
                }
            ]
        elif constraint_type in {"end-effector", "end-effectors"}:
            joint_names_set = set(constraint_obj.joint_names)
            for joint_name, track_name, eff_type in [
                ("LeftHand", "Left Hand", "left-hand"),
                ("RightHand", "Right Hand", "right-hand"),
                ("LeftFoot", "Left Foot", "left-foot"),
                ("RightFoot", "Right Foot", "right-foot"),
            ]:
                if joint_name not in joint_names_set:
                    continue
                target_joint_names = [joint_name]
                if "Hips" in joint_names_set:
                    target_joint_names.append("Hips")
                load_targets.append(
                    {
                        "track_name": track_name,
                        "constraint_track": session.constraints["End-Effectors"],
                        "joint_names": target_joint_names,
                        "end_effector_type": eff_type,
                    }
                )
            if not load_targets:
                raise KeyError(f"No recognized end-effector joint in constraint joint_names={constraint_obj.joint_names}")
        else:
            raise KeyError(f"Unsupported constraint type in loader: {constraint_type}")

        with session.timeline_data["keyframe_update_lock"]:
            for target in load_targets:
                track_id = session.timeline_data["tracks_ids"][target["track_name"]]
                constraint_track = target["constraint_track"]

                for (start_idx, end_idx), (start_idx_t, end_idx_t) in zip(intervals, intervals_indices):
                    interval_id = session.client.timeline.add_interval(track_id, start_idx, end_idx)
                    session.timeline_data["intervals"][interval_id] = {
                        "track_id": track_id,
                        "start_frame_idx": start_idx,
                        "end_frame_idx": end_idx,
                        "locked": False,
                        "opacity": 1.0,
                        "value": None,
                    }
                    if constraint_type == "root2d":
                        constraint_track.add_interval(interval_id, start_idx, end_idx, root_pos[start_idx_t : end_idx_t + 1])
                    elif constraint_type == "fullbody":
                        constraint_track.add_interval(
                            interval_id,
                            start_idx,
                            end_idx,
                            constraint_obj.global_joints_positions[start_idx_t : end_idx_t + 1],
                            constraint_obj.global_joints_rots[start_idx_t : end_idx_t + 1],
                        )
                    else:
                        constraint_track.add_interval(
                            interval_id,
                            start_idx,
                            end_idx,
                            constraint_obj.global_joints_positions[start_idx_t : end_idx_t + 1],
                            constraint_obj.global_joints_rots[start_idx_t : end_idx_t + 1],
                            target["joint_names"],
                            target["end_effector_type"],
                        )
                    interval_count += 1

                for frame, frame_t in zip(single_frames, single_frames_indices):
                    keyframe_id = session.client.timeline.add_keyframe(track_id, frame)
                    session.timeline_data["keyframes"][keyframe_id] = {
                        "track_id": track_id,
                        "frame": frame,
                        "locked": False,
                        "opacity": 1.0,
                        "value": None,
                    }
                    if constraint_type == "root2d":
                        constraint_track.add_keyframe(keyframe_id, frame, root_pos[frame_t])
                    elif constraint_type == "fullbody":
                        constraint_track.add_keyframe(
                            keyframe_id,
                            frame,
                            constraint_obj.global_joints_positions[frame_t],
                            constraint_obj.global_joints_rots[frame_t],
                        )
                    else:
                        constraint_track.add_keyframe(
                            keyframe_id,
                            frame,
                            constraint_obj.global_joints_positions[frame_t],
                            constraint_obj.global_joints_rots[frame_t],
                            target["joint_names"],
                            target["end_effector_type"],
                        )
                    keyframe_count += 1

    demo._apply_constraint_overlay_visibility(session)
    return {
        "path": str(path),
        "constraint_count": len(constraints_lst),
        "keyframe_count": keyframe_count,
        "interval_count": interval_count,
    }


def generate_session_motion(
    demo: Any,
    *,
    client_id: int | None = None,
    prompts: list[str],
    num_frames: list[int],
    num_samples: int = 1,
    seed: int = 0,
    diffusion_steps: int = 100,
    cfg_weight: list[float] | None = None,
    cfg_type: str | None = None,
    postprocess_parameters: dict[str, Any] | None = None,
    transitions_parameters: dict[str, Any] | None = None,
    real_robot_rotations: bool = False,
) -> dict[str, Any]:
    """Run Kimodo generation in the active demo process."""

    session = get_session(demo, client_id)
    demo.generate(
        session.client,
        prompts=prompts,
        num_frames=num_frames,
        num_samples=num_samples,
        seed=seed,
        diffusion_steps=diffusion_steps,
        cfg_weight=cfg_weight,
        cfg_type=cfg_type,
        postprocess_parameters=postprocess_parameters,
        transitions_parameters=transitions_parameters,
        real_robot_rotations=real_robot_rotations,
    )
    return get_session_state(demo, session.client.client_id)


def set_session_frame(demo: Any, *, client_id: int | None = None, frame: int, update_timeline: bool = True) -> dict[str, Any]:
    session = get_session(demo, client_id)
    frame = max(0, min(int(frame), int(session.max_frame_idx)))
    demo.set_frame(session.client.client_id, frame, update_timeline=update_timeline)
    return get_session_state(demo, session.client.client_id)


def _apply_camera_dict(client: Any, camera: dict[str, Any]) -> None:
    if "position" in camera:
        client.camera.position = np.asarray(camera["position"], dtype=np.float64)
    if "look_at" in camera:
        client.camera.look_at = np.asarray(camera["look_at"], dtype=np.float64)
    if "up_direction" in camera:
        client.camera.up_direction = np.asarray(camera["up_direction"], dtype=np.float64)
    if "wxyz" in camera:
        client.camera.wxyz = np.asarray(camera["wxyz"], dtype=np.float64)
    if "fov" in camera:
        client.camera.fov = float(camera["fov"])
    if "fov_radians" in camera:
        client.camera.fov = float(camera["fov_radians"])
    if "fov_degrees" in camera:
        client.camera.fov = math.radians(float(camera["fov_degrees"]))
    if "near" in camera:
        client.camera.near = float(camera["near"])
    if "far" in camera:
        client.camera.far = float(camera["far"])


def set_session_camera(
    demo: Any,
    *,
    client_id: int | None = None,
    camera: dict[str, Any] | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    """Set a deterministic Viser camera pose for a live render client."""

    session = get_session(demo, client_id)
    if preset is None and camera is not None:
        preset = camera.get("preset")
    payload: dict[str, Any] = {}
    if preset is not None:
        if preset not in CAMERA_PRESETS:
            raise KeyError(f"Unknown camera preset '{preset}'. Expected one of: {sorted(CAMERA_PRESETS)}")
        payload.update(CAMERA_PRESETS[preset])
    if camera:
        payload.update({k: v for k, v in camera.items() if k != "preset"})

    _apply_camera_dict(session.client, payload)
    return get_camera_state(session)


def get_camera_state(session: ClientSession) -> dict[str, Any]:
    client = session.client
    return {
        "position": _as_list(client.camera.position),
        "look_at": _as_list(client.camera.look_at),
        "up_direction": _as_list(client.camera.up_direction),
        "wxyz": _as_list(client.camera.wxyz),
        "fov_radians": float(client.camera.fov),
        "fov_degrees": math.degrees(float(client.camera.fov)),
        "image_width": int(client.camera.image_width),
        "image_height": int(client.camera.image_height),
    }


def set_session_visual_options(
    demo: Any,
    *,
    client_id: int | None = None,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Apply visualization toggles used by render/review loops."""

    session = get_session(demo, client_id)
    gui = session.gui_elements

    if "show_skeleton" in options:
        visible = bool(options["show_skeleton"])
        gui.gui_viz_skeleton_checkbox.value = visible
        for motion in session.motions.values():
            motion.character.set_skeleton_visibility(visible)

    if "show_skinned_mesh" in options:
        visible = bool(options["show_skinned_mesh"])
        gui.gui_viz_skinned_mesh_checkbox.value = visible
        for motion in session.motions.values():
            motion.character.set_skinned_mesh_visibility(visible)

    if "show_foot_contacts" in options:
        show = bool(options["show_foot_contacts"])
        gui.gui_viz_foot_contacts_checkbox.value = show
        for motion in session.motions.values():
            motion.character.set_show_foot_contacts(show, frame_idx=session.frame_idx)

    if "skinned_mesh_opacity" in options:
        opacity = float(options["skinned_mesh_opacity"])
        gui.gui_viz_skinned_mesh_opacity_slider.value = opacity
        for motion in session.motions.values():
            motion.character.set_skinned_mesh_opacity(opacity)

    if "show_timeline" in options:
        session.client.timeline.set_visible(bool(options["show_timeline"]))

    if "show_constraint_tracks" in options:
        demo.set_constraint_tracks_visible(session, bool(options["show_constraint_tracks"]))

    if "show_only_current_constraint" in options:
        session.show_only_current_constraint = bool(options["show_only_current_constraint"])
        demo._apply_constraint_overlay_visibility(session)

    return get_session_state(demo, session.client.client_id)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _parse_resolution(resolution: str | tuple[int, int] | None, session: ClientSession) -> tuple[int, int]:
    if isinstance(resolution, tuple):
        width, height = resolution
    elif isinstance(resolution, str) and "x" in resolution:
        width_s, height_s = resolution.lower().split("x", 1)
        width, height = int(width_s), int(height_s)
    else:
        width = int(session.client.camera.image_width)
        height = int(session.client.camera.image_height)
        if width <= 0 or height <= 0:
            width, height = 1280, 720
    return _round_up_to_multiple(int(width), 16), _round_up_to_multiple(int(height), 16)


def capture_session_frame(
    demo: Any,
    *,
    client_id: int | None = None,
    output_path: str | Path,
    frame: int | None = None,
    camera: dict[str, Any] | None = None,
    preset: str | None = None,
    resolution: str | tuple[int, int] | None = "1280x720",
    transport_format: str = "png",
    render_timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Capture one Viser frame to a PNG/JPEG file."""

    session = get_session(demo, client_id)
    if frame is not None:
        set_session_frame(demo, client_id=session.client.client_id, frame=frame)
    if preset is not None or camera is not None:
        set_session_camera(demo, client_id=session.client.client_id, camera=camera, preset=preset)

    width, height = _parse_resolution(resolution, session)
    image = session.client.get_render(  # type: ignore[arg-type]
        height=height,
        width=width,
        transport_format=transport_format,
        timeout_s=render_timeout_s,
    )
    path = _coerce_output_path(output_path, ext=f".{transport_format}")
    path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio

    iio.imwrite(path, image)
    return {
        "path": str(path),
        "frame": int(session.frame_idx),
        "width": width,
        "height": height,
        "transport_format": transport_format,
    }


def _video_path_for_view(output_path: Path, view: str, *, multiple_views: bool) -> Path:
    path = _coerce_output_path(output_path, ext=".mp4")
    if view == "current" and not multiple_views:
        return path
    if path.stem.endswith(f"_{view}"):
        return path
    return path.with_name(f"{path.stem}_{view}{path.suffix}")


def _write_h264_mp4(
    *,
    path: Path,
    frames: Any,
    fps: float,
    width: int,
    height: int,
    codec: str,
    crf: int,
    preset: str,
    pix_fmt: str,
    faststart: bool,
) -> dict[str, Any]:
    """Encode RGB frames to MP4 with explicit quality and timestamps."""

    import av

    fps_fraction = Fraction.from_float(float(fps)).limit_denominator(1000)
    time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"movflags": "+faststart"} if faststart else None
    container = av.open(str(path), "w", format="mp4", options=options)
    stream = container.add_stream(codec, rate=fps_fraction)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = pix_fmt
    stream.time_base = time_base
    stream.options = {
        "crf": str(int(crf)),
        "preset": preset,
    }
    if codec in {"h264", "libx264"}:
        stream.options.setdefault("profile", "high")

    frame_count = 0
    try:
        for image in frames:
            if image.ndim == 3 and image.shape[-1] == 4:
                alpha = image[..., 3:4].astype(np.float32) / 255.0
                rgb = image[..., :3].astype(np.float32)
                image = np.clip((rgb * alpha) + (255.0 * (1.0 - alpha)), 0, 255).astype(np.uint8)
            video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            video_frame = video_frame.reformat(width=int(width), height=int(height), format=pix_fmt)
            video_frame.pts = frame_count
            video_frame.time_base = time_base
            for packet in stream.encode(video_frame):
                container.mux(packet)
            frame_count += 1
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()

    return {
        "path": str(path),
        "frame_count": frame_count,
        "codec": codec,
        "crf": int(crf),
        "preset": preset,
        "pix_fmt": pix_fmt,
        "faststart": faststart,
    }


def render_session_video(
    demo: Any,
    *,
    client_id: int | None = None,
    output_path: str | Path,
    views: list[str] | None = None,
    resolution: str | tuple[int, int] = "1280x720",
    frame_step: int = 1,
    max_frames: int | None = DEFAULT_REVIEW_FRAME_CAP,
    transport_format: str = "png",
    render_timeout_s: float = 60.0,
    video_codec: str = "libx264",
    video_crf: int = 14,
    video_preset: str = "medium",
    video_pix_fmt: str = "yuv420p",
    mp4_faststart: bool = True,
    camera_mode: str = "static_fit",
    camera_margin: float = 1.2,
    camera_orientation: str = "trajectory",
    camera_fov_degrees: float = 45.0,
    camera_min_displacement_m: float = 0.35,
) -> dict[str, Any]:
    """Render the active Kimodo session to MP4 using Viser's native client capture."""

    session = get_session(demo, client_id)
    motion = get_primary_motion(session)
    width, height = _parse_resolution(resolution, session)
    frame_step = max(1, int(frame_step))
    source_frame_count = int(motion.joints_pos.shape[0])
    frame_indices = list(range(0, int(session.max_frame_idx) + 1, frame_step))
    selected_source_frame_count = len(frame_indices)
    if max_frames is not None:
        frame_indices = frame_indices[: int(max_frames)]
    if not frame_indices:
        raise ValueError("No frames selected for render.")
    truncated_by_max_frames = len(frame_indices) < selected_source_frame_count

    selected_views = views or ["current"]
    output = Path(output_path)
    video_paths: list[str] = []
    encodes: list[dict[str, Any]] = []
    original_frame = int(session.frame_idx)
    non_current_views = [view for view in selected_views if view != "current"]
    if camera_mode == "static_fit":
        camera_plan = plan_static_fit_cameras(
            joints_pos=motion.joints_pos,
            root_index=int(session.skeleton.root_idx),
            frame_indices=frame_indices,
            views=non_current_views,
            width=width,
            height=height,
            camera_margin=camera_margin,
            camera_orientation=camera_orientation,
            camera_fov_degrees=camera_fov_degrees,
            camera_min_displacement_m=camera_min_displacement_m,
        )
    elif camera_mode == "preset":
        preset_views = {}
        for view in non_current_views:
            if view not in CAMERA_PRESETS:
                raise KeyError(f"Unknown camera preset '{view}'. Expected one of: {sorted(CAMERA_PRESETS)}")
            preset_views[view] = {"camera": CAMERA_PRESETS[view]}
        camera_plan = {"mode": "preset", "views": preset_views}
    else:
        raise ValueError("camera_mode must be 'static_fit' or 'preset'.")

    plan_path = output.parent / "camera_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(camera_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        session.client.timeline.disable_constraints()
        for view in selected_views:
            if view != "current":
                _apply_camera_dict(session.client, camera_plan["views"][view]["camera"])

            def _frames_for_view():
                for frame_idx in frame_indices:
                    demo.set_frame(session.client.client_id, frame_idx, update_timeline=True)
                    yield session.client.get_render(
                        height=height,
                        width=width,
                        transport_format=transport_format,  # type: ignore[arg-type]
                        timeout_s=render_timeout_s,
                    )

            target = _video_path_for_view(output, view, multiple_views=len(selected_views) > 1)
            encodes.append(
                _write_h264_mp4(
                    path=target,
                    frames=_frames_for_view(),
                    fps=float(session.model_fps) / frame_step,
                    width=width,
                    height=height,
                    codec=video_codec,
                    crf=video_crf,
                    preset=video_preset,
                    pix_fmt=video_pix_fmt,
                    faststart=mp4_faststart,
                )
            )
            video_paths.append(str(target))
    finally:
        demo.set_frame(session.client.client_id, original_frame)
        session.client.timeline.enable_constraints()

    return {
        "video_paths": video_paths,
        "views": selected_views,
        "frame_count": int(len(frame_indices)),
        "source_frame_count": source_frame_count,
        "selected_source_frame_count": int(selected_source_frame_count),
        "max_frames": int(max_frames) if max_frames is not None else None,
        "truncated_by_max_frames": bool(truncated_by_max_frames),
        "fps": float(session.model_fps) / frame_step,
        "width": width,
        "height": height,
        "transport_format": transport_format,
        "encodes": encodes,
        "camera_plan": camera_plan,
        "camera_plan_path": str(plan_path),
    }


def get_session_state(demo: Any, client_id: int | None = None) -> dict[str, Any]:
    session = get_session(demo, client_id)
    constraint_counts = {}
    for name, constraint in session.constraints.items():
        constraint_counts[name] = {
            "keyframes": len(getattr(constraint, "keyframes", {})),
            "intervals": len(getattr(constraint, "intervals", {})),
        }
    return {
        "client_id": int(session.client.client_id),
        "model_name": session.model_name,
        "fps": float(session.model_fps),
        "frame": int(session.frame_idx),
        "max_frame": int(session.max_frame_idx),
        "motion_count": len(session.motions),
        "motions": list(session.motions.keys()),
        "constraints": constraint_counts,
        "camera": get_camera_state(session),
    }
