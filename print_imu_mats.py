import argparse
import torch


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _get_seq(data, key, seq_idx):
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _as_tensor(value[min(seq_idx, len(value) - 1)])
    return _as_tensor(value)


def _print_mat(name, mat, max_rows=3):
    if mat is None:
        print(f"{name}: None")
        return
    print(f"{name} shape: {tuple(mat.shape)}")
    if mat.ndim == 3:
        print(f"{name}[0]:\n{mat[0].numpy()}")
    elif mat.ndim == 4:
        print(f"{name}[0,0]:\n{mat[0,0].numpy()}")
    else:
        print(f"{name}:\n{mat.numpy()}")


def main():
    parser = argparse.ArgumentParser(description="Print IMU matrices from a dataset .pt file.")
    parser.add_argument("--file", required=True, help="Path to .pt dataset file.")
    parser.add_argument("--seq", type=int, default=0, help="Sequence index.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (for RIS/aS/wS).")
    args = parser.parse_args()

    data = torch.load(args.file, map_location="cpu")

    RIM = _get_seq(data, "RIM", args.seq)
    RSB = _get_seq(data, "RSB", args.seq)
    RIS = _get_seq(data, "RIS", args.seq)
    aS = _get_seq(data, "aS", args.seq)
    wS = _get_seq(data, "wS", args.seq)
    mS = _get_seq(data, "mS", args.seq)

    print("=== IMU matrices ===")
    _print_mat("RIM", RIM)
    _print_mat("RSB", RSB)
    _print_mat("RIS", RIS)

    if RIS is not None and RIS.ndim == 4:
        f = max(0, min(args.frame, RIS.shape[0] - 1))
        print(f"RIS[{f}]:\n{RIS[f].numpy()}")

    if aS is not None and aS.ndim == 3:
        f = max(0, min(args.frame, aS.shape[0] - 1))
        print(f"aS[{f}]:\n{aS[f].numpy()}")
    if wS is not None and wS.ndim == 3:
        f = max(0, min(args.frame, wS.shape[0] - 1))
        print(f"wS[{f}]:\n{wS[f].numpy()}")
    if mS is not None and mS.ndim == 3:
        f = max(0, min(args.frame, mS.shape[0] - 1))
        print(f"mS[{f}]:\n{mS[f].numpy()}")


if __name__ == "__main__":
    main()
