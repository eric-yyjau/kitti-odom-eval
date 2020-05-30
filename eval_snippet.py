
import argparse
import numpy as np

KITTI_PATH = './kitti-odom-eval/dataset/kitti_odom/gt_poses/'

class Pose_processor(object):
    def __init__(self):
        pass
    
    @staticmethod
    def abs2rel_poses(poses):
        """
        poses:
            T{1_w}, T{2_w}, ..., T{n_w}
        return:
            rel_poses: T{12}, T{23}, {T34}
        """
        rel_poses = []
        # p_0 = poses[0] # T1_world
        p_0 = Pose_processor.pose2mat(poses[0])
        from numpy.linalg import inv
        for p in poses:
            p = Pose_processor.pose2mat(p)
            # print(f"p: {p}")
            rel_poses.append(inv(p) @ p_0) # T_12 = (T2_world)^(-1) @ T1_world        
            p_0 = p
        return rel_poses

    @staticmethod
    def rel2abs_poses(rel_poses, T1w=None, kitti=False, world_coord=True):
        """
        Params:
            rel_poses: T{12}, T{23}, {T34}
        return:
            poses: T{1_w}, T{2_w}, T{3_w}, ...
            (T2_w = (T{12} @ Tw_1)^(-1) = T{1_w} @ T{21} )
        pose2mat: torch.tensor [B, 3, 4]
        """
        if T1w is None:
            T1w = np.identity(4)
        poses = [T1w]
        # p_0 = Photometric_frontend.pose2mat(T1w)
        p_0 = Pose_processor.pose2mat(T1w)
        from numpy.linalg import inv
        for p in rel_poses:
            # print(f"pose: {p}")
            p = Pose_processor.pose2mat(p)
            p_0 = p_0 @ inv(p) # be aware of p (3x4 -> 4x4, 6x1 -> 4x4)
            poses.append( p_0 )
        if kitti:
            poses = [p[:3,:] for p in poses]
        return poses

    @staticmethod
    def pose2mat(pose):
        """
        param:
            pose: only one single pose
        """
        if pose.shape == (4,4):
            return pose
        if pose.shape[0] == 6:
            from inverse_warp import pose_vec2mat
            pose_mat = pose_vec2mat(torch.tensor(pose[np.newaxis,...])).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        if len(pose.shape)==2 and pose.shape[1] == 6:
            from inverse_warp import pose_vec2mat
            pose_mat = pose_vec2mat(torch.tensor(pose)).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        if pose.shape == (3,4):
            pose_mat = np.vstack([pose, np.array([0, 0, 0, 1])])
        return pose_mat
        pass

    @staticmethod
    def align_scale(pred, gt_pose):
        """
        params:
            pose: [4,4] --> [3,4]
            gt_pose: [4,4] --> [3,4]
        """
        # snippet_length = gt.shape[0]
        scale_factor = np.sum(gt_pose[:3, -1] * pred[:3, -1]) / (np.sum(
            pred[:3, -1] ** 2
        ) + 1e-4)
        print(f"scale_factor: {scale_factor}")
        align = pred + 0
        align[:3,-1] = scale_factor * align[:3,-1]
        return align
        # ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))

    @staticmethod
    def save_poses(poses, filename='test.txt', length=-1):
        """
        # save absolute poses to file
        params: 
            poses: absolute poses [poses, gt_poses]
            filename -> string: path and name
        """
        # filename = f"{args.save_path}/est.txt"
        # arr = np.array(poses)[:length,:3]
        arr = np.array(poses)[:,:3]
        arr = arr.reshape(-1,12)
        np.savetxt(filename, arr)
        return True

def dump_json(dict, filename):
    import json
    json = json.dumps(dict)
    f = open(filename, "w")
    f.write(json)
    f.close()



from eval_tools import Exp_table_processor

def eval_trajectory_snippet(save_folder, seq, length=5):
    # seq = "10"
    table_processor = Exp_table_processor
    poses_gt = table_processor.read_gt_poses(path=KITTI_PATH, seq=seq)
    poses_est = np.genfromtxt(f'{save_folder}/{seq}.txt')
    # poses_est = poses_est[:,1:].reshape(-1,12)
    poses_est = poses_est.reshape(-1,12)
    poses_est = poses_est.reshape(-1,3,4)
    print(f"length est vs. gt: {len(poses_est)}, {len(poses_gt)}")
    assert len(poses_est) == len(poses_gt)
    data = table_processor.pose_seq_ate(poses_est, poses_gt, 5)
    entries = ["error_names", "mean_errors", "std_errors"]
    # results = { key: data[key] for key in entries }
    results = {}
    for i, n in enumerate(data["error_names"]):
        for item in entries[1:]:
            results[f"{n}_{item}"] = data[item][i].astype(float)
    dump_json(results, f"{save_folder}/snip_ate.yml")
    # print(data)    
    pass

def eval_trajectory_gt_scale(save_folder, seq):
    table_processor = Exp_table_processor
    poses_gt = table_processor.read_gt_poses(path=KITTI_PATH, seq=seq)
    poses_est = np.genfromtxt(f'{save_folder}/{seq}.txt')
    # poses_est = poses_est[:,1:].reshape(-1,12)
    poses_est = poses_est.reshape(-1,12)
    poses_est = poses_est.reshape(-1,3,4)
    print(f"length est vs. gt: {len(poses_est)}, {len(poses_gt)}")
    assert len(poses_est) == len(poses_gt)
    rel_poses_est = Pose_processor.abs2rel_poses(poses_est)
    rel_poses_gt = Pose_processor.abs2rel_poses(poses_gt)
    abs_poses_est = Pose_processor.rel2abs_poses(rel_poses_est)
    print("align scale")
    poses_align = []
    for pred, gt in zip(rel_poses_est, rel_poses_gt):
        p = Pose_processor.align_scale(pred, gt)
        poses_align.append(p)
    abs_poses_align = Pose_processor.rel2abs_poses(poses_align)
    if len(abs_poses_align) > len(poses_gt):
        diff = len(abs_poses_align) - len(poses_gt)
        abs_poses_align = abs_poses_align[diff:]
    # save
    print(f"length est vs. gt: {len(abs_poses_align)}, {len(poses_gt)}")
    Pose_processor.save_poses(abs_poses_align, f"{save_folder}/{seq}_align.txt")
    return abs_poses_align
    # print(f"poses_est: {poses_est[:5]}")
    # print(f"abs_poses_est: {abs_poses_est[:5]}")
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='KITTI evaluation')
    parser.add_argument('--result', type=str, required=True,
                        help="Result directory")
    parser.add_argument('--seqs', 
                    nargs="+",
                    type=int, 
                    help="sequences to be evaluated",
                    default=None)
    parser.add_argument('--mode', 
                    type=str, 
                    help="[snippet | scale]",
                    default='snippet')
    args = parser.parse_args()
    print(f"args: {args}")
    if args.mode == 'snippet':
        eval_trajectory_snippet(args.result, f"{args.seqs[0]}")
    elif args.mode == 'scale':
        eval_trajectory_gt_scale(args.result, f"{args.seqs[0]}")
    else:
        print(f"mode is not defined: {args.mode}")
    pass