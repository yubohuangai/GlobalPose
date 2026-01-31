import argparse
import os
import time
from typing import Any, Tuple

import numpy as np
import torch

import articulate as art


def _to_tensor(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.from_numpy(np.asarray(data))


def _select_sequence(data: Any, seq_idx: int) -> Any:
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Empty sequence list in results.")
        return data[min(seq_idx, len(data) - 1)]
    return data


def _pose_to_rotmat(pose: Any) -> torch.Tensor:
    pose_t = _to_tensor(pose)
    if pose_t.ndim == 4 and pose_t.shape[-2:] == (3, 3):
        return pose_t.float()
    if pose_t.ndim == 3 and pose_t.shape[1:] == (24, 3):
        pose_t = pose_t.reshape(-1, 24, 3)
        return art.math.axis_angle_to_rotation_matrix(pose_t).view(-1, 24, 3, 3).float()
    if pose_t.ndim == 2 and pose_t.shape[1] == 72:
        pose_t = pose_t.view(-1, 24, 3)
        return art.math.axis_angle_to_rotation_matrix(pose_t).view(-1, 24, 3, 3).float()
    if pose_t.ndim == 3 and pose_t.shape[-2:] == (3, 3):
        return pose_t.unsqueeze(0).float()
    raise ValueError(f"Unsupported pose shape: {tuple(pose_t.shape)}")


def _tran_to_seq(tran: Any, n_frames: int) -> torch.Tensor:
    if tran is None:
        return torch.zeros(n_frames, 3)
    tran_t = _to_tensor(tran).float()
    if tran_t.ndim == 1 and tran_t.shape[0] == 3:
        tran_t = tran_t.view(1, 3).repeat(n_frames, 1)
    if tran_t.ndim == 2 and tran_t.shape[1] == 3:
        if tran_t.shape[0] == n_frames:
            return tran_t
        if tran_t.shape[0] == 1:
            return tran_t.repeat(n_frames, 1)
    raise ValueError(f"Unsupported tran shape: {tuple(tran_t.shape)}")


def _safe_torch_load(file_path: str) -> Any:
    try:
        return torch.load(file_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(file_path, map_location="cpu")


def _load_motion(file_path: str, seq_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data = _safe_torch_load(file_path)
    if isinstance(data, dict):
        if "pose" in data:
            pose = _select_sequence(data["pose"], seq_idx)
            tran = _select_sequence(data.get("tran"), seq_idx) if "tran" in data else None
        elif "poses" in data:
            pose = _select_sequence(data["poses"], seq_idx)
            tran = _select_sequence(data.get("trans"), seq_idx) if "trans" in data else None
        else:
            raise ValueError("Unsupported result format: missing 'pose' key.")
    else:
        pose = _select_sequence(data, seq_idx)
        tran = None
    pose_rm = _pose_to_rotmat(pose)
    tran_seq = _tran_to_seq(tran, pose_rm.shape[0])
    return pose_rm, tran_seq


def _apply_upright(pose: torch.Tensor, tran: torch.Tensor, enabled: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return pose, tran
    rot = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=pose.dtype)
    pose_adj = pose.clone()
    pose_adj[:, 0] = rot.matmul(pose_adj[:, 0])
    tran_adj = tran.matmul(rot.t())
    return pose_adj, tran_adj


def _view_with_model(pose: torch.Tensor, tran: torch.Tensor, fps: float) -> None:
    model = art.ParametricModel("models/SMPL_male.pkl")
    model.view_motion([pose], [tran], fps=fps)


def _view_with_open3d(
    pose: torch.Tensor,
    tran: torch.Tensor,
    fps: float,
    show_axes: bool,
    upright: bool,
    save_dir: str = None,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for the open3d viewer.") from exc

    model = art.ParametricModel("models/SMPL_male.pkl")
    tran_local = tran - tran[:1]
    verts = model.forward_kinematics(pose, tran=tran_local, calc_mesh=True)[2].cpu().numpy()
    if upright:
        rot = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]])
        verts = verts @ rot.T
    faces = np.asarray(model.face, dtype=np.int32)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GlobalPose Viewer", width=960, height=720)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    vis.add_geometry(mesh)

    if show_axes:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(axis)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(verts.shape[0]):
        t0 = time.time()
        mesh.vertices = o3d.utility.Vector3dVector(verts[i])
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)
        if not vis.poll_events():
            break
        vis.update_renderer()
        if save_dir:
            vis.capture_screen_image(os.path.join(save_dir, f"frame_{i:06d}.png"), do_render=False)
        time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
    vis.destroy_window()


def _view_with_bullet(pose: torch.Tensor, tran: torch.Tensor, fps: float) -> None:
    from articulate.utils.bullet import MotionViewer
    viewer = MotionViewer(1, overlap=True)
    viewer.view_offline([pose], [tran], fps=fps)


def _view_with_unity(pose: torch.Tensor, tran: torch.Tensor, fps: float) -> None:
    from articulate.utils.unity import MotionViewer
    viewer = MotionViewer(1, overlap=True)
    viewer.view_offline([pose], [tran], fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize inferred motion results.")
    parser.add_argument("--file", required=True, help="Path to the results .pt file.")
    parser.add_argument("--seq", type=int, default=0, help="Sequence index (default: 0).")
    parser.add_argument("--fps", type=float, default=60.0, help="Playback FPS (default: 60).")
    parser.add_argument(
        "--viewer",
        choices=["model", "open3d", "bullet", "unity"],
        default="model",
        help="Viewer backend.",
    )
    parser.add_argument(
        "--upright",
        action="store_true",
        default=True,
        help="Rotate by 180 deg around X to fix upside-down motions.",
    )
    parser.add_argument(
        "--no-upright",
        dest="upright",
        action="store_false",
        help="Disable the upright rotation.",
    )
    parser.add_argument(
        "--axes",
        action="store_true",
        default=True,
        help="Show coordinate axes (open3d viewer only).",
    )
    parser.add_argument(
        "--no-axes",
        dest="axes",
        action="store_false",
        help="Hide coordinate axes.",
    )
    parser.add_argument(
        "--open3d-upright",
        action="store_true",
        default=True,
        help="Apply extra upright flip for open3d viewer.",
    )
    parser.add_argument(
        "--no-open3d-upright",
        dest="open3d_upright",
        action="store_false",
        help="Disable the open3d upright flip.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Directory to save rendered frames (open3d viewer only).",
    )
    args = parser.parse_args()

    file_path = os.path.normpath(args.file)
    pose, tran = _load_motion(file_path, args.seq)
    pose, tran = _apply_upright(pose, tran, args.upright)

    if args.viewer == "model":
        _view_with_model(pose, tran, args.fps)
        return

    if args.viewer == "open3d":
        save_dir = args.save_dir.strip() or None
        _view_with_open3d(pose, tran, args.fps, args.axes, args.open3d_upright, save_dir)
        return

    if args.viewer == "unity":
        _view_with_unity(pose, tran, args.fps)
        return

    try:
        _view_with_bullet(pose, tran, args.fps)
    except Exception as exc:
        print("Bullet viewer failed, falling back to model viewer.")
        print("Reason:", exc)
        _view_with_model(pose, tran, args.fps)


if __name__ == "__main__":
    main()
