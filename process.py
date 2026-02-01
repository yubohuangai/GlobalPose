import os
import argparse
import pickle
import torch
import glob
import articulate as art
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d


def process_totalcapture(debug=False):
    print('======================== Processing TotalCapture Dataset ========================')
    joint_names = ['L_LowArm', 'R_LowArm', 'L_LowLeg', 'R_LowLeg', 'Head', 'Pelvis']
    vicon_gt_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/official'     # download from TotalCapture page
    imu_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/gryo_mag'          # download from TotalCapture page
    calib_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/imu'             # download from TotalCapture page
    DIP_smpl_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/DIP_smpl'     # SMPL pose calculated by DIP. Download from DIP page
    AMASS_smpl_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/AMASS_smpl' # SMPL pose calculated by AMASS. Download from AMASS page
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'AMASS_pose': [], 'DIP_pose': []}
    n_extracted_imus = len(joint_names)

    for subject_name in ['s1', 's2', 's3', 's4', 's5']:
        for action_name in sorted(os.listdir(os.path.join(imu_dir, subject_name))):
            # read imu file
            f = open(os.path.join(imu_dir, subject_name, action_name), 'r')
            line = f.readline().split('\t')
            n_sensors, n_frames = int(line[0]), int(line[1])
            R = torch.zeros(n_frames, n_extracted_imus, 4)
            a = torch.zeros(n_frames, n_extracted_imus, 3)
            w = torch.zeros(n_frames, n_extracted_imus, 3)
            m = torch.zeros(n_frames, n_extracted_imus, 3)
            for i in range(n_frames):
                assert int(f.readline()) == i + 1, 'parse imu file error'
                for _ in range(n_sensors):
                    line = f.readline().split('\t')
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        R[i, j] = torch.tensor([float(_) for _ in line[1:5]])  # wxyz
                        a[i, j] = torch.tensor([float(_) for _ in line[5:8]])
                        w[i, j] = torch.tensor([float(_) for _ in line[8:11]])
                        m[i, j] = torch.tensor([float(_) for _ in line[11:14]])
            R = art.math.quaternion_to_rotation_matrix(R).view(-1, n_extracted_imus, 3, 3)

            # read calibration file
            name = subject_name + '_' + action_name.split('_')[0].lower()
            RSB = torch.zeros(n_extracted_imus, 3, 3)
            RIM = torch.zeros(n_extracted_imus, 3, 3)
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_bone.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RSB[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_ref.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RIM[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            RSB = RSB.matmul(torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0.]]))  # change bone frame to SMPL
            RIM = RIM.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1.]]))   # change global frame to SMPL

            # read root translation
            tran = []
            with open(os.path.join(vicon_gt_dir, subject_name.upper(), action_name.split('_')[0].lower(), 'gt_skel_gbl_pos.txt')) as f:
                idx = f.readline().split('\t').index('Hips')
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    t = [float(_) * 0.0254 for _ in line.split('\t')[idx].split(' ')]   # inches_to_meters
                    tran.append([-t[0], t[1], -t[2]])
            tran = torch.tensor(tran)

            # read SMPL pose parameters calculated by AMASS
            f = os.path.join(AMASS_smpl_dir, subject_name, action_name.split('_')[0].lower() + '_poses.npz')
            AMASS_pose = None
            if os.path.exists(f):
                d = np.load(f)
                AMASS_pose = torch.from_numpy(d['poses'])[:, :72].float()
                root_rot = art.math.axis_angle_to_rotation_matrix(AMASS_pose[:, :3])
                root_rot = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).matmul(root_rot)  # align global frame
                root_rot = art.math.rotation_matrix_to_axis_angle(root_rot)
                AMASS_pose[:, :3] = root_rot
                AMASS_pose[:, 66:] = 0  # hand

            # read SMPL pose parameters calculated by DIP
            f = os.path.join(DIP_smpl_dir, name + '.pkl')
            DIP_pose = None
            if os.path.exists(f):
                d = pickle.load(open(f, 'rb'), encoding='latin1')
                DIP_pose = torch.from_numpy(d['gt']).float()

            # align data
            n_aligned_frames = min(n_frames, tran.shape[0], AMASS_pose.shape[0] if AMASS_pose is not None else 1e8, DIP_pose.shape[0] if DIP_pose is not None else 1e8)
            if AMASS_pose is not None:
                AMASS_pose = AMASS_pose[-n_aligned_frames:]
            if DIP_pose is not None:
                DIP_pose = DIP_pose[-n_aligned_frames:]
            tran = tran[-n_aligned_frames:] - tran[-n_aligned_frames]
            R = R[-n_aligned_frames:]
            a = a[-n_aligned_frames:]
            w = w[-n_aligned_frames:]
            m = m[-n_aligned_frames:]

            # validate data (for debug purpose)
            if debug and DIP_pose is not None:
                model = art.ParametricModel('models/SMPL_male.pkl')
                DIP_pose = art.math.axis_angle_to_rotation_matrix(DIP_pose).view(-1, 24, 3, 3)
                syn_RMB = model.forward_kinematics_R(DIP_pose)[:, [18, 19, 4, 5, 15, 0]]
                real_RMB = RIM.transpose(1, 2).matmul(R).matmul(RSB)
                real_aM = RIM.transpose(1, 2).matmul(R).matmul(a.unsqueeze(-1)).squeeze(-1)
                print('real-syn imu ori err:', art.math.radian_to_degree(art.math.angle_between(real_RMB, syn_RMB).mean()))
                print('mean acc in M:', real_aM.mean(dim=(0, 1)))   # (0, +g, 0)

            # save results
            data['name'].append(name)
            data['RIM'].append(RIM)
            data['RSB'].append(RSB)
            data['RIS'].append(R)
            data['aS'].append(a)
            data['wS'].append(w)
            data['mS'].append(m)
            data['tran'].append(tran)
            data['AMASS_pose'].append(AMASS_pose)
            data['DIP_pose'].append(DIP_pose)
            print('Finish Processing %s' % name, '(no AMASS pose)' if AMASS_pose is None else '', '(no DIP pose)' if DIP_pose is None else '')

    os.makedirs('data/dataset_work/TotalCapture', exist_ok=True)
    torch.save(data, 'data/dataset_work/TotalCapture/data.pt')


def process_dipimu():
    print('======================== Processing DIP_IMU Dataset ========================')
    data_dir = 'C:/yxy/datasets/DIP_IMU/dataset_raw'
    imu_mask = [7, 8, 11, 12, 0, 2]  # head, spine2, belly, lchest, rchest, lshoulder, rshoulder, lelbow, relbow, lhip, rhip, lknee, rknee, lwrist, lwrist, lankle, rankle
    subject_names = ['s_%02d' % i for i in range(1, 11)]
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'pose': []}
    g = torch.tensor([0, -9.798, 0])
    for subject_name in subject_names:
        for motion_name in os.listdir(os.path.join(data_dir, subject_name)):
            f = os.path.join(data_dir, subject_name, motion_name)
            d = pickle.load(open(f, 'rb'), encoding='latin1')
            acc = torch.from_numpy(d['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(d['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(d['gt']).float()

            # fill nan with linear interpolation
            for i in range(ori.shape[0]):
                for j in range(6):
                    if torch.isnan(ori[i, j]).sum() > 0:
                        k1, k2 = i - 1, i + 1
                        while k1 >= 0 and torch.isnan(ori[k1, j]).sum() > 0: k1 -= 1
                        while k2 < ori.shape[0] and torch.isnan(ori[k2, j]).sum() > 0: k2 += 1
                        if k1 >= 0 and k2 < ori.shape[0]:
                            slerp = Slerp([k1, k2], Rotation.from_matrix(ori[[k1, k2], j].numpy()))
                            ori[k1 + 1:k2, j] = torch.from_numpy(slerp(list(range(k1 + 1, k2))).as_matrix()).float()
                        elif k1 < 0:
                            ori[:k2, j] = ori[k2, j]
                        elif k2 >= ori.shape[0]:
                            ori[k1 + 1:, j] = ori[k1, j]
                    if torch.isnan(acc[i, j]).sum() > 0:
                        k1, k2 = i - 1, i + 1
                        while k1 >= 0 and torch.isnan(acc[k1, j]).sum() > 0: k1 -= 1
                        while k2 < ori.shape[0] and torch.isnan(acc[k2, j]).sum() > 0: k2 += 1
                        if k1 >= 0 and k2 < ori.shape[0]:
                            lerp = interp1d([k1, k2], acc[[k1, k2], j].numpy(), axis=0)
                            acc[k1 + 1:k2, j] = torch.from_numpy(lerp(list(range(k1 + 1, k2)))).float()
                        elif k1 < 0:
                            acc[:k2, j] = acc[k2, j]
                        elif k2 >= ori.shape[0]:
                            acc[k1 + 1:, j] = acc[k1, j]

            if torch.isnan(acc).sum() > 0 or torch.isnan(ori).sum() > 0 or torch.isnan(pose).sum() > 0:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
                continue

            # synthesize wS and mS, calculate aS
            w = art.math.rotation_matrix_to_axis_angle(ori[:-1].transpose(2, 3).matmul(ori[1:])).view(-1, ori.shape[1], 3) * 60
            w = torch.cat((w, torch.zeros_like(w[:1])))
            m = ori.transpose(2, 3).matmul(torch.tensor([1, 0, 0.]).unsqueeze(-1)).squeeze(-1)
            a = ori.transpose(2, 3).matmul((acc - g).unsqueeze(-1)).squeeze(-1)

            name = subject_name.replace('_', '') + '_' + motion_name[:-4]
            data['name'].append(name)
            data['RIM'].append(torch.eye(3).repeat(6, 1, 1))
            data['RSB'].append(torch.eye(3).repeat(6, 1, 1))
            data['RIS'].append(ori)
            data['aS'].append(a)
            data['wS'].append(w)
            data['mS'].append(m)
            data['tran'].append(torch.zeros(pose.shape[0], 3))
            data['pose'].append(pose)
            print('Finish Processing %s' % name)

    os.makedirs('data/dataset_work/DIP_IMU', exist_ok=True)
    torch.save(data, 'data/dataset_work/DIP_IMU/data.pt')


def process_amass():
    print('======================== Processing AMASS Dataset ========================')
    data_dir = 'D:/Admin/Data/AMASS/dataset_raw'
    names = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU', 'DFaust67',
             'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD', 'BioMotionLab_NTroje',
             'BMLhandball', 'MPI_Limits', 'TotalCapture']   # align with previous works
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'pose': [], 'gender': [], 'shape': []}
    for name in names:
        print('Processing %s' % name)
        for npz_fname in glob.glob(os.path.join(data_dir, name, name, '*/*_poses.npz')) + glob.glob(os.path.join(data_dir, name, name, '*/*_stageii.npz')):
            seq_name = npz_fname[npz_fname.rfind(name):-4]
            try:
                cdata = np.load(npz_fname, allow_pickle=True)
                if 'mocap_framerate' in cdata:
                    framerate = int(cdata['mocap_framerate'])
                elif 'mocap_frame_rate' in cdata:
                    framerate = int(cdata['mocap_frame_rate'])
                else:
                    print('\tFail to process %s: no framerate' % seq_name)
                    continue
                if cdata['poses'].shape[0] < framerate * 0.5:
                    print('\tFail to process %s: too short' % seq_name)
                    continue
                if framerate == 120:
                    pose = torch.from_numpy(cdata['poses'][::2].astype(np.float32)).view(-1, 156)[:, :72]
                    tran = torch.from_numpy(cdata['trans'][::2].astype(np.float32)).view(-1, 3)
                elif framerate == 60 or framerate == 59:
                    pose = torch.from_numpy(cdata['poses'].astype(np.float32)).view(-1, 156)[:, :72]
                    tran = torch.from_numpy(cdata['trans'].astype(np.float32)).view(-1, 3)
                else:
                    origin_pose = cdata['poses'].reshape(-1, 52, 3)
                    origin_tran = cdata['trans'].reshape(-1, 3)
                    origin_t = np.arange(origin_pose.shape[0]) / framerate
                    t = np.arange(0, origin_t[-1], 1 / 60)
                    pose = np.empty((len(t), 24, 3))
                    for i in range(24):
                        pose[:, i] = Slerp(origin_t, Rotation.from_rotvec(origin_pose[:, i]))(t).as_rotvec()
                    tran = interp1d(origin_t, origin_tran, axis=0)(t)
                    pose = torch.from_numpy(pose.astype(np.float32)).view(-1, 72)
                    tran = torch.from_numpy(tran.astype(np.float32)).view(-1, 3)
            except Exception as e:
                print('\tFail to process %s:' % seq_name, e)
                continue
            pose[:, :3] = art.math.rotation_matrix_to_axis_angle(amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, :3])))
            tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
            gender = str(cdata['gender'])
            if gender == "b'female'": gender = 'female'
            if gender == "b'male'": gender = 'male'
            data['name'].append(seq_name)
            data['pose'].append(pose.clone())
            data['tran'].append(tran.clone())
            data['gender'].append(gender)
            data['shape'].append(torch.from_numpy(cdata['betas'][:10]).float())
            print('\tFinish Processing %s: n_frames %d' % (seq_name, pose.shape[0]))

    assert len(data['name']) > 0, 'cannot find AMASS dataset'
    os.makedirs('data/dataset_work/AMASS', exist_ok=True)
    torch.save(data, 'data/dataset_work/AMASS/data.pt')


def _read_movella_dot_csv(file_path):
    arr = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 12:
        raise ValueError(f'Unexpected CSV format: {file_path}')
    arr = arr[:, :12]
    arr = arr[~np.isnan(arr).any(axis=1)]
    quat = arr[:, 2:6]  # wxyz
    acc = arr[:, 6:9]   # m/s^2
    gyr = arr[:, 9:12]  # deg/s
    return quat, acc, gyr


def _find_sensor_file(data_dir, prefix):
    candidates = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('.csv')])
    if len(candidates) == 0:
        raise FileNotFoundError(f'Cannot find CSV for prefix: {prefix}')
    return os.path.join(data_dir, candidates[0])


def _rotation_yaw(angle_rad: float) -> torch.Tensor:
    c = torch.cos(torch.tensor(angle_rad))
    s = torch.sin(torch.tensor(angle_rad))
    return torch.tensor([[c, 0.0, s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0, c]])


def _from_to_rotmat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    v = torch.cross(a, b, dim=0)
    c = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    s = v.norm()
    if s < 1e-8:
        if c > 0:
            return torch.eye(3)
        axis = torch.tensor([1.0, 0.0, 0.0])
        if torch.abs(a[0]) > 0.9:
            axis = torch.tensor([0.0, 1.0, 0.0])
        v = torch.cross(a, axis)
        v = v / (v.norm() + 1e-8)
        vx = art.math.hat(v.view(1, 3))[0]
        return torch.eye(3) + 2.0 * (vx @ vx)
    vx = art.math.hat(v.view(1, 3))[0]
    return torch.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _sensor_flip_matrix(flip: str) -> torch.Tensor:
    flip = (flip or "").lower()
    sign_x = -1.0 if "x" in flip else 1.0
    sign_y = -1.0 if "y" in flip else 1.0
    sign_z = -1.0 if "z" in flip else 1.0
    return torch.diag(torch.tensor([sign_x, sign_y, sign_z], dtype=torch.float32))


def process_kickoff_sync(data_dir='data/kickoff_sync',
                          output_path='data/test_datasets/kickoff_sync.pt',
                          tpose_start=600,
                          tpose_end=700,
                          rim_method='enu',
                          sensor_flip='',
                          global_yaw_deg=180):
    print('======================== Processing kickoff_sync ========================')
    sensor_order = ['leftarm2', 'rightarm2', 'leftleg2', 'rightleg2', 'head', 'pelvis2']
    seq_name = os.path.basename(os.path.normpath(data_dir))

    q_list, a_list, w_list = [], [], []
    min_len = None
    for prefix in sensor_order:
        f = _find_sensor_file(data_dir, prefix)
        quat, acc, gyr = _read_movella_dot_csv(f)
        min_len = quat.shape[0] if min_len is None else min(min_len, quat.shape[0])
        q_list.append(quat)
        a_list.append(acc)
        w_list.append(gyr)

    q_list = [q[:min_len] for q in q_list]
    a_list = [a[:min_len] for a in a_list]
    w_list = [w[:min_len] for w in w_list]

    quat = torch.from_numpy(np.stack(q_list, axis=1)).float()  # [T, 6, 4]
    aS = torch.from_numpy(np.stack(a_list, axis=1)).float()    # [T, 6, 3]
    wS = torch.from_numpy(np.stack(w_list, axis=1)).float()    # [T, 6, 3]
    wS = wS * (np.pi / 180.0)

    RIS = art.math.quaternion_to_rotation_matrix(quat.view(-1, 4)).view(min_len, 6, 3, 3)
    RIS = art.math.normalize_rotation_matrix(RIS.view(-1, 3, 3)).view(min_len, 6, 3, 3)

    C_S = _sensor_flip_matrix(sensor_flip)
    if not torch.allclose(C_S, torch.eye(3)):
        RIS = RIS.matmul(C_S)
        Ct = C_S.t()
        aS = Ct.view(1, 1, 3, 3).matmul(aS.unsqueeze(-1)).squeeze(-1)
        wS = Ct.view(1, 1, 3, 3).matmul(wS.unsqueeze(-1)).squeeze(-1)

    tpose_start = max(0, min(tpose_start, min_len - 1))
    tpose_end = max(tpose_start, min(tpose_end, min_len - 1))

    R_enu_to_smpl = torch.tensor([[1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0]])
    RIM = R_enu_to_smpl.unsqueeze(0).repeat(6, 1, 1)

    if rim_method == 'gravity':
        # Gravity alignment: estimate inertial -> model using T-pose specific force.
        aI_tpose = RIS[tpose_start:tpose_end + 1].matmul(aS[tpose_start:tpose_end + 1].unsqueeze(-1)).squeeze(-1)
        fI_est = aI_tpose.mean(dim=(0, 1))
        f_rest_model = torch.tensor([0.0, 9.8, 0.0])
        R_I_to_M = _from_to_rotmat(fI_est, f_rest_model)
        RIM = R_I_to_M.t().repeat(6, 1, 1)

    if global_yaw_deg != 0.0:
        print("RIM before yaw:")
        print(RIM[0].numpy())
        R_yaw = _rotation_yaw(global_yaw_deg * np.pi / 180.0)
        RIM = RIM.matmul(R_yaw.t())
        print("RIM after yaw:")
        print(RIM[0].numpy())
    RIM = art.math.normalize_rotation_matrix(RIM.view(-1, 3, 3)).view(6, 3, 3)

    # calibrate RSB so that RMB matches SMPL zero-pose global rotations
    model = art.ParametricModel("models/SMPL_male.pkl")
    R_local0 = torch.eye(3).view(1, 1, 3, 3).repeat(1, 24, 1, 1)
    R_global0 = model.forward_kinematics_R(R_local0)[0]
    joint_idx = torch.tensor([18, 19, 4, 5, 15, 0], dtype=torch.long)
    RMB_target = R_global0[joint_idx]

    A = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS[tpose_start:tpose_end + 1])
    RSB = torch.zeros(6, 3, 3)
    for j in range(6):
        RSB_j = A[:, j].transpose(1, 2).matmul(RMB_target[j].view(1, 3, 3)).mean(dim=0)
        RSB[j] = art.math.normalize_rotation_matrix(RSB_j.view(1, 3, 3))[0]
    RSB = art.math.normalize_rotation_matrix(RSB.view(-1, 3, 3)).view(6, 3, 3)

    mS = torch.zeros_like(aS)
    tran = torch.zeros(min_len, 3)

    data = {
        'name': [seq_name],
        'RIM': [RIM],
        'RSB': [RSB],
        'RIS': [RIS],
        'aS': [aS],
        'wS': [wS],
        'mS': [mS],
        'tran': [tran],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print('Saved to', output_path)


if __name__ == '__main__':
    # import sys; sys.stdout = open('data/dataset_work/log.txt', 'a')
    parser = argparse.ArgumentParser(description='Dataset preprocessing utilities.')
    parser.add_argument(
        '--dataset',
        choices=['kickoff_sync', 'dipimu', 'totalcapture', 'amass', 'all'],
        default='kickoff_sync',
        help='Which dataset to process.'
    )
    parser.add_argument('--kickoff-dir', type=str, default='data/kickoff_sync', help='Kickoff CSV folder.')
    parser.add_argument('--kickoff-out', type=str, default='data/test_datasets/kickoff_sync.pt', help='Output .pt path.')
    parser.add_argument('--tpose-start', type=int, default=600, help='T-pose start frame (inclusive).')
    parser.add_argument('--tpose-end', type=int, default=700, help='T-pose end frame (inclusive).')
    parser.add_argument(
        '--rim-method',
        choices=['enu', 'gravity'],
        default='enu',
        help='Global frame alignment method for Movella DOT.',
    )
    parser.add_argument(
        '--sensor-flip',
        type=str,
        default='',
        help='Flip sensor axes (any combo of x,y,z), e.g. "x", "yz", "xyz".',
    )
    parser.add_argument(
        '--global-yaw',
        type=float,
        default=0.0,
        help='Extra yaw correction in degrees (model up axis).',
    )
    args = parser.parse_args()

    if args.dataset in ('kickoff_sync', 'all'):
        process_kickoff_sync(
            data_dir=args.kickoff_dir,
            output_path=args.kickoff_out,
            tpose_start=args.tpose_start,
            tpose_end=args.tpose_end,
            rim_method=args.rim_method,
            sensor_flip=args.sensor_flip,
            global_yaw_deg=args.global_yaw,
        )
    if args.dataset in ('dipimu', 'all'):
        process_dipimu()
    if args.dataset in ('totalcapture', 'all'):
        process_totalcapture()
    if args.dataset in ('amass', 'all'):
        process_amass()
