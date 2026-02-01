import torch
import articulate as art


def main():
    model = art.ParametricModel("models/SMPL_male.pkl")

    r_local0 = torch.eye(3).view(1, 1, 3, 3).repeat(1, 24, 1, 1)
    r_global0 = model.forward_kinematics_R(r_local0)[0]

    joint_idx = torch.tensor([18, 19, 4, 5, 15, 0], dtype=torch.long)
    rmb_target = r_global0[joint_idx]

    q = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])

    q_t = q.t()
    rmb_q = q_t.unsqueeze(0).matmul(rmb_target).matmul(q.unsqueeze(0))
    diffs = rmb_target - rmb_q
    frob = torch.linalg.norm(diffs.view(diffs.shape[0], -1), dim=1)

    print("RMB_target and Q^T RMB_target Q per joint:")
    for j, val, r, rq in zip(joint_idx.tolist(), frob.tolist(), rmb_target, rmb_q):
        print(f"joint {j:2d} RMB_target=")
        print(r.numpy())
        print(f"joint {j:2d} Q^T RMB_target Q=")
        print(rq.numpy())
        print(f"joint {j:2d} ||diff||_F = {val:.6f}")
        print("")
    print("||RMB_target - Q^T RMB_target Q||_F per joint (summary):")
    for j, val in zip(joint_idx.tolist(), frob.tolist()):
        print(f"joint {j:2d}: {val:.6f}")
    print(f"mean: {frob.mean().item():.6f}")


if __name__ == "__main__":
    main()
