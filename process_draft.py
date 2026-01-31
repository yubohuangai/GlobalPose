"""
process_draft.py
"""
import csv
import os
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


def process_movella(
    input_dir=r"C:\Users\yuboh\GitHub\MovellaExporter\data\kickoff_sync",
    out_path=r"data/dataset_work/Movella/data.pt",
    seq_name="movella_kickoff",
    tpose_range=(600, 700),
    sensor_order=("leftarm", "rightarm", "leftleg", "rightleg", "head", "pelvis"),
    gyro_unit="deg",   # "deg" or "rad"
    acc_unit="m/s2",   # Movella DOT export is usually m/s^2
):
    """
    Convert Movella DOT CSV exports to GlobalPose-style dataset dict.

    Produces:
      - RIS: [T, 6, 3, 3]  rotation matrices from IMU (sensor->inertial/world frame as exported)
      - aS : [T, 6, 3]     accelerations (as exported; usually sensor frame)
      - wS : [T, 6, 3]     angular vel (converted to rad/s)
      - RSB: [6, 3, 3]     sensor-to-bone offset (estimated from T-pose window)
      - RIM: [6, 3, 3]     inertial-to-model global alignment (identity for now)
      - tran: [T, 3]       zeros (no translation from IMU-only)
    """


    def from_to_rotmat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Return R such that R @ a ~= b. a,b are 3-vectors.
        Stable for most cases; handles a ~ +/- b.
        """
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)

        v = torch.cross(a, b, dim=0)
        c = torch.clamp(torch.dot(a, b), -1.0, 1.0)
        s = v.norm()

        if s < 1e-8:
            # parallel or anti-parallel
            if c > 0:
                return torch.eye(3)
            # 180 deg: choose any axis orthogonal to a
            axis = torch.tensor([1.0, 0.0, 0.0])
            if torch.abs(a[0]) > 0.9:
                axis = torch.tensor([0.0, 1.0, 0.0])
            v = torch.cross(a, axis)
            v = v / (v.norm() + 1e-8)
            vx = art.math.hat(v.view(1, 3))[0]
            return torch.eye(3) + 2.0 * (vx @ vx)

        vx = art.math.hat(v.view(1, 3))[0]
        R = torch.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        return R


    print("======================== Processing Movella DOT Dataset ========================")
    print("Input:", input_dir)
    print("Output:", out_path)

    # ------------- find files -------------
    # Your filenames look like: head_D422CD0096A6_20260123_145514.csv
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    # map prefix -> filepath
    # We'll match by prefix containing one of sensor_order keywords
    prefix_to_file = {}
    for f in files:
        low = f.lower()
        for key in sensor_order:
            if low.startswith(key):
                prefix_to_file[key] = os.path.join(input_dir, f)
                break

    missing = [k for k in sensor_order if k not in prefix_to_file]
    if missing:
        raise RuntimeError(f"Missing CSVs for sensors: {missing}\nFound: {list(prefix_to_file.keys())}")

    # ------------- read CSV helper -------------
    def read_one_csv(csv_path: str):
        """
        Read Movella DOT exported CSV.

        Expected columns (case-insensitive):
          PacketCounter, SampleTimeFine,
          Quat_W, Quat_X, Quat_Y, Quat_Z,
          Acc_X, Acc_Y, Acc_Z,
          Gyr_X, Gyr_Y, Gyr_Z
        """
        pcs, qs, accs, gyrs = [], [], [], []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            # Normalize column names (strip + lower)
            field_map = {(k or "").strip().lower(): k for k in (reader.fieldnames or [])}

            def get(row, name):
                k = field_map.get(name.lower(), None)
                if k is None:
                    return None
                v = row.get(k, "")
                if v is None:
                    return None
                v = v.strip()
                if v == "":
                    return None
                return v

            for row in reader:
                # PacketCounter must exist and be numeric
                pc_str = get(row, "PacketCounter")
                qw_str = get(row, "Quat_W")
                if pc_str is None or qw_str is None:
                    continue  # skip blank/bad rows

                try:
                    pc = int(float(pc_str))
                    qw = float(get(row, "Quat_W"))
                    qx = float(get(row, "Quat_X"))
                    qy = float(get(row, "Quat_Y"))
                    qz = float(get(row, "Quat_Z"))

                    ax = float(get(row, "Acc_X"))
                    ay = float(get(row, "Acc_Y"))
                    az = float(get(row, "Acc_Z"))

                    gx = float(get(row, "Gyr_X"))
                    gy = float(get(row, "Gyr_Y"))
                    gz = float(get(row, "Gyr_Z"))
                except (TypeError, ValueError):
                    continue  # skip any partially-written row

                pcs.append(pc)
                qs.append([qw, qx, qy, qz])
                accs.append([ax, ay, az])
                gyrs.append([gx, gy, gz])

        if len(pcs) == 0:
            raise RuntimeError(
                f"No valid rows parsed from: {csv_path}\n"
                f"Check header names and whether the file is empty."
            )

        pc = np.asarray(pcs, dtype=np.int64)
        q = np.asarray(qs, dtype=np.float32)  # (T,4)
        acc = np.asarray(accs, dtype=np.float32)  # (T,3)
        gyr = np.asarray(gyrs, dtype=np.float32)  # (T,3)
        return pc, q, acc, gyr

    # ------------- load all sensors -------------
    pcs, qs, accs, gyrs = [], [], [], []
    for key in sensor_order:
        pc, q, acc, gyr = read_one_csv(prefix_to_file[key])
        pcs.append(pc); qs.append(q); accs.append(acc); gyrs.append(gyr)

    # ------------- align length by min frames (safety) -------------
    T = min([q.shape[0] for q in qs])
    qs   = [q[:T] for q in qs]
    accs = [a[:T] for a in accs]
    gyrs = [w[:T] for w in gyrs]
    print(f"Loaded {len(sensor_order)} sensors, aligned length T={T}")

    # ------------- stack tensors: [T,6,*] -------------
    q = torch.from_numpy(np.stack(qs, axis=1)).float()        # [T,6,4] wxyz
    a = torch.from_numpy(np.stack(accs, axis=1)).float()      # [T,6,3]
    w = torch.from_numpy(np.stack(gyrs, axis=1)).float()      # [T,6,3]

    # convert gyro to rad/s if needed
    if gyro_unit.lower() == "deg":
        w = w * (torch.pi / 180.0)

    # ------------- RIS (rotation matrices) -------------
    RIS = art.math.quaternion_to_rotation_matrix(q.view(-1, 4)).view(T, 6, 3, 3)
    RIS = art.math.normalize_rotation_matrix(RIS.view(-1, 3, 3)).view(T, 6, 3, 3)

    # ---- sensor-frame convention fix (Movella sensor axes -> expected sensor axes) ----
    # If RIS maps sensor->inertial, then applying a constant sensor-axis change C_S means:
    #   RIS' = RIS @ C_S
    # and vectors expressed in sensor coords must also change:
    #   v' = C_S^T @ v   (for a, w)
    C_S = torch.eye(3, device=RIS.device, dtype=RIS.dtype)

    # Example candidates (only one at a time):
    # C_S = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=RIS.device, dtype=RIS.dtype))   # 180° about X
    # C_S = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=RIS.device, dtype=RIS.dtype))   # 180° about Y
    # C_S = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=RIS.device, dtype=RIS.dtype))   # 180° about Z

    if not torch.allclose(C_S, torch.eye(3, device=RIS.device, dtype=RIS.dtype)):
        RIS = RIS.matmul(C_S)  # [T,6,3,3]

        # Rotate IMU vectors into the same "new sensor frame"
        Ct = C_S.t()
        a = Ct.view(1, 1, 3, 3).matmul(a.unsqueeze(-1)).squeeze(-1)  # [T,6,3]
        w = Ct.view(1, 1, 3, 3).matmul(w.unsqueeze(-1)).squeeze(-1)  # [T,6,3]

    # ------------- T-pose calibration for RSB -------------
    s0, s1 = tpose_range
    s0 = max(0, min(T - 1, s0))
    s1 = max(s0 + 1, min(T, s1))
    print(f"Using T-pose frames [{s0} .. {s1-1}] for RSB")

    # ---- estimate rest specific-force direction in Movella inertial frame ----
    aI_tpose = RIS[s0:s1].matmul(a[s0:s1].unsqueeze(-1)).squeeze(-1)  # [K,6,3]
    print("mean aI_tpose:", aI_tpose.mean(dim=(0, 1)))

    fI_est = aI_tpose.mean(dim=(0, 1))  # specific force at rest (points "up")

    # ---- align Movella inertial to model frame using rest specific force ----
    f_rest_model = torch.tensor([0.0, +9.8, 0.0])  # equals -g_world_model
    R_I_to_M = from_to_rotmat(fI_est, f_rest_model)
    RIM = R_I_to_M.t().repeat(6, 1, 1)

    # ------------- T-pose calibration for RSB (use SMPL zero-pose target, not identity) -------------

    # 1) Get SMPL zero-pose global rotations as target RMB for the 6 joints
    model = art.ParametricModel("models/SMPL_male.pkl")

    R_local0 = torch.eye(3).view(1, 1, 3, 3).repeat(1, 24, 1, 1)  # [1,24,3,3] all identity local rotations
    R_global0 = model.forward_kinematics_R(R_local0)[0]  # [24,3,3] global rotations in SMPL zero pose

    # IMPORTANT: joint indices must match your sensor placement
    # This is the same mapping used in TotalCapture debug: [L_LowArm, R_LowArm, L_LowLeg, R_LowLeg, Head, Pelvis]
    # If your sensors are on upper-arm / thigh (not forearm / shank), you MUST change these indices.
    joint_idx = torch.tensor([18, 19, 4, 5, 15, 0], dtype=torch.long)
    RMB_target = R_global0[joint_idx]  # [6,3,3]

    # 2) Estimate RSB so that: (RIM^T * RIS) * RSB ~= RMB_target during T-pose
    RSB = torch.zeros(6, 3, 3)

    A = RIM.transpose(1, 2).unsqueeze(0).matmul(RIS[s0:s1])  # [K,6,3,3]  A = RIM^T * RIS

    for j in range(6):
        # RSB_j ≈ mean_k ( A_kj^T * RMB_target_j )
        RSB_j = A[:, j].transpose(1, 2).matmul(RMB_target[j].view(1, 3, 3)).mean(dim=0)
        RSB[j] = art.math.normalize_rotation_matrix(RSB_j.view(1, 3, 3))[0]

    RMB_tpose_est = RIM.transpose(1, 2).matmul(RIS[s0:s1]).matmul(RSB)  # [K,6,3,3]

    K = RMB_tpose_est.shape[0]
    RMB_target_K = RMB_target.unsqueeze(0).expand(K, -1, -1, -1)  # [K,6,3,3]

    err = art.math.radian_to_degree(
        art.math.angle_between(
            RMB_tpose_est.reshape(-1, 3, 3),
            RMB_target_K.reshape(-1, 3, 3),
        )
    ).mean()

    print("T-pose RMB error (deg):", float(err))


    # translation is unknown from IMU-only; keep zero
    tran = torch.zeros(T, 3)

    # ------------- build dataset dict -------------
    data = {
        "name": [seq_name],
        "RIM":  [RIM],
        "RSB":  [RSB],
        "RIS":  [RIS],
        "aS":   [a],
        "wS":   [w],
        "mS":   [torch.zeros_like(a)],  # Movella export you showed doesn't include mag; keep zeros
        "tran": [tran],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)
    print("Saved:", out_path)
    print("Keys:", list(data.keys()))
    print("Shapes:",
          "RIS", tuple(RIS.shape),
          "aS", tuple(a.shape),
          "wS", tuple(w.shape),
          "RIM", tuple(RIM.shape),
          "RSB", tuple(RSB.shape),
          "tran", tuple(tran.shape))

    # -------- sanity checks for test.py convention --------
    g = torch.tensor([0.0, -9.8, 0.0])
    aM = RIM.transpose(1, 2).matmul(RIS).matmul(a.unsqueeze(-1)).squeeze(-1) + g
    print("aM[s0:s1]: ", aM[s0:s1].mean(dim=(0, 1)))  # should be close to [0,0,0]
    RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB)

    for j, name in enumerate(["LArm", "RArm", "LLeg", "RLeg", "Head", "Pelvis"]):
        print(name)
        print("x:", RMB[s0, j, :, 0])
        print("y:", RMB[s0, j, :, 1])
        print("z:", RMB[s0, j, :, 2])


if __name__ == '__main__':
    # import sys; sys.stdout = open('data/dataset_work/log.txt', 'a')
    # process_amass()
    # process_totalcapture()
    # process_dipimu()
    process_movella()
