import logging
import numpy as np

result_dict = {}
result_dict_entry = ["epi_dist_mean_gt", "num_prob", "num_warped_prob", "num_matches"]

import os
import sys
# module_path = os.path.abspath(os.path.join('.'))
# if module_path not in sys.path:
# sys.path.append(module_path)

# import dsac_tools.utils_geo as utils_geo
# error_R = utils_geo.rot12_to_angle_error(R_cam, delta_Rtij_inv[:3, :3])
# error_t = utils_geo.vector_angle(t_cam, delta_Rtij_inv[:3, 3:4])


#########  shared functions
# @staticmethod
def read_file_dict_to_dict(files_dict, allow_pickle=False):
    # load npz
    exp_dict = {}
    for i, en in enumerate(files_dict):
        file = files_dict[en]
        exp_dict[en] = np.load(file, allow_pickle=allow_pickle)
        # exp_list.append(err_dict)
    print(f"len of exp_dict: {len(list(exp_dict))}")
    return exp_dict






# from utils.eval_tools import Result_processor
class Result_processor(object):
    def __init__(self, result_dict_entry):
        self.result_dict_all = {}
        self.result_dict_entry = result_dict_entry
        self.result_processed = {}
        for ent in result_dict_entry:
            self.result_dict_all[ent] = []
        pass

    def add_config(self, config):
        """
        # add config: thd
        """
        pass
        

    def load_result(self, result_dict):
        """
        result_dict: dictionary with entries
        """
        for ent in self.result_dict_entry:
            if ent not in result_dict:
                logging.warning(f"{ent} not in the result dictionary")
            else:
                self.result_dict_all[ent].append(result_dict[ent])

    # def read_to_ratio_dict(self, files_dict, allow_pickle=True):
    #     exp_dict = read_file_dict_to_dict(files_dict=files_dict, allow_pickle=True)
    #     for i, exp in enumerate(exp_dict):
    #         result_processor.result_dict_all = exp_dict[exp]
    #         result = result_processor.inlier_ratio(
    #             "epi_dist_mean_gt", inlier_thd, if_print=True
    #         )
    #     pass

    def output_result(self, method=[None], **params):
        for m in method:
            if m is not None:
                func = getattr(self, m)
                func(params[m])
        pass

    # def inlier_ratio(
    #     self, entry, thd_list, mask_entry=None, mask_thd=1, if_print=False
    # ):
    def inlier_ratio(
        self, result_list, thd_list, mask_list=None, mask_thd=1, if_print=False
    ):
        """
        # mask_entry: ['mscores']
        input: 
            result_list: one nested results from a sequence
            mask_list: the score of the results (mscores for correspondences)
        """
        # result_list = self.result_dict_all[entry]
        # mask_list = None if mask_entry is None else self.result_dict_all[mask_entry]
        # thd_list = np.array(thd_list)
        table = []
        num_corrs = []
        for i, re in enumerate(result_list):
            est_arr = np.array(re)
            if mask_list is not None:
                m = self.get_mask(mask_list[i], mask_thd)
                assert (
                    m.shape[0] == est_arr.shape[0]
                ), "mask size not equal to estimated arr"
                est_arr = est_arr[m]
            num_corrs.append(est_arr.shape[0])

            ratio_list = self.inlier_ratio_from_est(est_arr, thd_list)

            table.append(ratio_list)
        table = np.array(table)
        # self.result_processed["inlier_ratio"] = table
        results = {"inlier_ratio": table.mean(axis=0), "num_corrs": np.array(num_corrs), 
                   "inlier_ratio_full": table}
        if if_print:
            print(f"inlier ratio thd: {thd_list}, result: {results}")
        return results

    # def collect_arr_from_result(self, entry, if_print=False):
    def get_entry_from_result(self, entry, if_print=False):
        if entry in self.result_dict_all:
            return self.result_dict_all[entry]
        else:
            logging.error(f"{entry} is not in the dictionary.")

    def ap_inlier_thd(
        self, inlier_entry, inlier_thds, mask_thds, mask_entry="mscores", if_print=False
    ):
        table = []
        num_corrs = []
        for j, thd in enumerate(mask_thds):
            # get inlier ratio under the thd
            results = self.inlier_ratio(
                inlier_entry,
                inlier_thds,
                mask_entry=mask_entry,
                mask_thd=thd,
                if_print=if_print,
            )
            table.append(results["inlier_ratio"])
            num_corrs.append(results["num_corrs"])
        table = np.array(table)
        num_corrs = np.array(num_corrs)
        print(f"table: {table.shape}")
        results = {
            "inlier_thd": table,
            "num_corrs": num_corrs,  # np [thds, Num of samples]
        }
        return results
        pass

    # def num_inlier_thd(self, inlier_entry, )

    def save_result(self, filename, item):
        if item == "result_dict_all":
            np.savez_compressed(filename, **self.result_dict_all)
        pass

    @staticmethod
    def get_mask(arr, thd):
        return np.array(arr) < thd


    
    def inlier_ratio_nested(self, est_nested, thd_list):
        table = []
        for i, re in enumerate(est_nested):
            est_arr = np.array(re)
            # print(f"est_arr: {est_arr}")
            # num_corrs.append(est_arr.shape[0])
            ratio_list = self.inlier_ratio_from_est(est_arr, thd_list)
            table.append(ratio_list)
        table = np.array(table)
        # self.result_processed["inlier_ratio"] = table
        # results = {"inlier_ratio": table.mean(axis=0), "num_corrs": np.array(num_corrs)}
        return table.mean(axis=0)

    @staticmethod
    def inlier_ratio_from_est(est_arr, thd_list):
        """
        inlier_ratio_from_est(est_arr, thd_list) -> list
        """
        ratio_list = []
        for thd in thd_list:
            ratio = np.sum(est_arr < thd) / est_arr.shape[0]
            ratio_list.append(ratio)
        return ratio_list


# from Result_processor import inlier_ratio_from_est
from pathlib import Path

# from . import Result_processor
class Exp_table_processor(Result_processor):
    """
    # process the results of different sequences.
    # sort into table
    """

    def __init__(self, config, seq_dict_name="seq_dict", debug=False, 
                if_mean=True, if_median=True, **params):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            format="[%(asctime)s %(levelname)s] %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # level=logging.INFO,
            level=level,
        )
        ## params
        self.ratio_dict = {}
        # thd_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        # err_mat = ['err_q', 'err_t', 'epi_dists']
        self.config = config
        self.thd_list = config["data"]["thresh"]["thd_list"]
        self.err_mat = config["data"]["err_mat"]
        self.seq_dict = {}
        if type(seq_dict_name) is list:
            for name in seq_dict_name:
                self.seq_dict.update(config["data"][name])
        else:
            self.seq_dict.update(config["data"][seq_dict_name])

        self.files_dict = self.read_file_list(
            self.seq_dict, base_path=config["data"]["base_path"]
        )
        self.if_mean = if_mean
        self.if_median = if_median
        # self.if_highlights = self.config
        self.exp_dict = None  # the most important dictionary
        ## print to check
        logging.debug(f"folder_list: {self.files_dict}")

        pass

    # def get_

    @staticmethod
    def read_file_list(seq_dict, base_path="", folder_idx=0, file_idx=1):
        files_dict = {}
        for i, en in enumerate(seq_dict):
            files_dict[en] = (
                Path(base_path) / seq_dict[en][folder_idx] / seq_dict[en][file_idx]
            )
        return files_dict

    @staticmethod
    def get_mean_median(name, arr, mean=False, median=False):
        result = {}
        if mean:
            result[name + "_mean"] = [arr.mean()]
        if median:
            result[name + "_median"] = [np.median(arr)]
        return result

    @staticmethod
    def write_to_writer(writer, result_dict):
        from tqdm import tqdm
        for i, err in enumerate(result_dict):
            print(f"element: {err}")
            for iter, n in tqdm(enumerate(result_dict[err])):
                writer.add_scalar(err, n, iter)

    @staticmethod
    def get_relative_poses(poses, delta_ij=1):
        from .dsac_tools import utils_misc as utils_misc
        poses_list = []
        for frame_num in range(len(poses)-delta_ij):
            j = delta_ij + frame_num
            relative_scene_pose = np.linalg.inv(
                utils_misc.Rt_pad(poses[j])
            ) @ utils_misc.Rt_pad(poses[frame_num])
            poses_list.append(relative_scene_pose)
        return poses_list

    @staticmethod
    def get_relative_poses_error(poses, est_poses):
        assert len(poses) == len(est_poses)
        from .dsac_tools import utils_misc as utils_misc
        poses_list = []
        for p, est_p in zip(poses, est_poses):
            # print(f"pose: {p}")
            pad = lambda p: utils_misc.Rt_pad(p) if p.shape == (3,4) else p
            relative_scene_pose = np.linalg.inv( pad(p) ) @ pad(est_p)
            poses_list.append(relative_scene_pose)
        return poses_list

    @staticmethod
    def relative_poses_to_rt(poses_list, trans_abs=False):
        import dsac_tools.utils_geo as utils_geo
        # from .dsac_tools import utils_geo as utils_geo
        rot_list = []
        trans_list = []
        for delta_Rtij_inv in poses_list:
            err_q = utils_geo.rot12_to_angle_error(np.identity(3), delta_Rtij_inv[:3, :3])
            err_t = utils_geo.vector_angle(np.array([0,0,-1]), delta_Rtij_inv[:3, 3:4])
            if trans_abs:
                err_t_r = utils_geo.vector_angle(np.array([0,0,1]), delta_Rtij_inv[:3, 3:4])
                err_t = err_t if err_t < err_t_r else err_t_r

            rot_list.append(err_q)
            trans_list.append(err_t)
        print(f"rot_list: {len(rot_list)}, trans_list: {len(trans_list)}")
        return {"rot": rot_list, "trans": trans_list}

    @staticmethod
    def read_gt_poses(path="/data/kitti/odometry/poses", seq=10):
        # import dsac_tools.utils_misc as utils_misc
        filename = f"{path}/{seq}.txt"
        logging.info(f"read from: {filename}")
        poses = np.genfromtxt(filename).astype(np.float32).reshape(-1, 3, 4)
        return poses

    @staticmethod
    def compensate_poses(poses):
        """
        # compensate part of the poses
        input: 
            poses: np[batch, 3, 4]

        """
        poses = np.stack([p for p in poses])
        first_pose = poses[0]
        poses[:, :, -1] -= first_pose[:, -1]
        compensated_poses = (
            np.linalg.inv(first_pose[:, :3]) @ poses
        )  # [3,3] @ [batch, 3, 4]
        return compensated_poses

    @staticmethod
    def get_abs_poses(poses, if_print=False):
        # table_processor.exp_dict['Si-D-k-f.k']['relative_poses_body']
        poses_abs = []
        poses_abs.append(np.identity(4)[:3])
        last_pose = np.identity(4)
        from numpy.linalg import inv

        # for i, pose in enumerate(table_processor.exp_dict['Si-D-k-f.k']['relative_poses_body']):
        for i, pose in enumerate(poses):
            #     print(f"{i}: {pose}")
            last_pose = pose @ last_pose  ## make sure the multiplication order.
            poses_abs.append(inv(last_pose)[:3])
            if if_print and i < 5:
                print(f"pose abs: {poses_abs[-1]}")
        poses_abs = np.array(poses_abs)
        logging.info(f"get abs poses: {poses_abs.shape}")
        return poses_abs

    def get_all_abs_poses(self, item="relative_poses_body"):
        poses_dict = {}
        exp_dict = self.exp_dict
        for i, exp in enumerate(exp_dict):
            poses = exp_dict[exp][item]
            poses_dict.update({exp: self.get_abs_poses(poses, if_print=False)})
        self.poses_dict = poses_dict
        logging.info(f"get poses from: {list(poses_dict)}")
        return poses_dict

    @staticmethod
    def export_poses(poses_dict, path="logs/", prefix="", postfix="_date"):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        for i, exp in enumerate(poses_dict):
            poses = poses_dict[exp]
            ### avoid using '.' in filename
            exp_outname = exp.replace(".", "_")
            filename = f"{path}/{prefix}{exp_outname}{postfix}.txt"
            logging.info(f"save poses to: {filename}")
            np.savetxt(filename, np.stack(poses).reshape(-1, 12), delimiter=" ")

    @staticmethod
    def compute_pose_error(gt, pred):
        RE = 0
        snippet_length = gt.shape[0]
        scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(
            pred[:, :, -1] ** 2
        )
        ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
        for gt_pose, pred_pose in zip(gt, pred):
            # Residual matrix to which we compute angle's sin and cos
            R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
            s = np.linalg.norm(
                [R[0, 1] - R[1, 0], R[1, 2] - R[2, 1], R[0, 2] - R[2, 0]]
            )
            c = np.trace(R) - 1
            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += np.arctan2(s, c)

        # return ATE/snippet_length, RE/snippet_length
        return {
            "ATE": ATE / snippet_length,
            "RE": RE / snippet_length,
            "scale_factor": scale_factor,
        }

    @staticmethod
    def pose_seq_ate(est_poses, gt_poses, seq_length=5):
        """
        # compute absolute translation error on small snippets
        input:
            est_poses: np[N, 3, 4]
            gt_poses: np[N, 3, 4]
        """
        assert len(est_poses) <= len(gt_poses)
        # from evaluations.pose_evaluation_utils import poses_compensate
        est_length = len(est_poses) - seq_length
        errors = np.zeros((est_length, 2), np.float32)
        scale_factors = []
        algined_poses = []
        for i in range(est_length):
            est_pose_snip = Exp_table_processor.compensate_poses(
                est_poses[i : i + seq_length]
            )
            gt_pose_snip = Exp_table_processor.compensate_poses(
                gt_poses[i : i + seq_length]
            )
            results = Exp_table_processor.compute_pose_error(
                # est_pose_snip, gt_pose_snip
                ## need to check
                gt_pose_snip, est_pose_snip
            )

            errors[i] = results["ATE"], results["RE"]
            pose = np.copy(est_poses[i])
            pose[:, -1] = pose[:, -1] * results["scale_factor"]  ### buggy
            algined_poses.append(pose)
            scale_factors.append(results["scale_factor"])
        mean_errors = errors.mean(0)
        std_errors = errors.std(0)
        error_names = ["ATE", "RE"]
        print("")
        print("Results")
        print("\t {:>10}, {:>10}".format(*error_names))
        print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
        print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

        return {
            "error_names": error_names,
            "mean_errors": mean_errors,
            "std_errors": std_errors,
            "errors": errors,
            "scale_factors": scale_factors,
            "aligned_poses": algined_poses,
        }


    def get_entry_mean_med(self, mean_mat, if_print=False, allow_pickle=False):
        ratio_dict = self.ratio_dict
        # exp_names = list(self.files_dict)
        # exp_dict = read_file_dict_to_dict(self.files_dict, allow_pickle)
        exp_dict = self.exp_dict
        # assert self.exp_dict is not None
        for i, exp in enumerate(exp_dict):
            for en in mean_mat:
                arr = exp_dict[exp][en]
                arr = np.array(arr)
                print(f"arr: {arr}")
                # get mean                                
                temp_dict = self.get_mean_median(
                    en, arr.flatten(), mean=True, median=True
                )
                print(f"temp_dict: {temp_dict}")
                ratio_dict[exp].update(temp_dict)
        self.ratio_dict = ratio_dict
        print(f"ratio_dict: {ratio_dict}")
        pass

    def get_result_dict(self, if_print=False, nested=False, allow_pickle=False):
        ratio_dict = {}
        # read exps
        err_mat = self.err_mat
        exp_names = list(self.files_dict)
        exp_dict = read_file_dict_to_dict(self.files_dict, allow_pickle)
        self.exp_dict = exp_dict
        thd_list = self.thd_list

        print(f"err_mat: {err_mat}, exp_names: {exp_names}")
        assert len(list(exp_dict)) == len(exp_names)
        # loop through experiments
        for i, exp in enumerate(exp_dict):
            name = exp  # exp_names[i]
            ratio_dict[name] = {}
            if if_print:
                print(f"- name: {name}")
            # loop through metrics
            for en in err_mat:
                arr = exp_dict[exp][en]
                print(f"exp_dict[exp]: {list(exp_dict[exp])}")
                # print(f"{exp}: {en}, {arr.shape}, ")
                # print(f"{exp}: {en}, {arr[0].shape}, {type(arr[0])}")
                # if arr[0] is np.ndarray:
                inlier_ratio_full = None
                if nested:
                    # ratio_list = self.inlier_ratio_nested(arr, thd_list)
                    results = self.inlier_ratio(
                        arr, thd_list, if_print=True, mask_list=exp_dict[exp]['mscores'], mask_thd=1.0
                    # "epi_dist_mean_gt", inlier_thd, if_print=True,
                    )
                    ratio_list = results['inlier_ratio']
                    inlier_ratio_full = results['inlier_ratio_full']
                else:
                    ratio_list = self.inlier_ratio_from_est(arr.reshape(-1, 1), thd_list)
                ratio_list = np.array(ratio_list)
                ratio_dict[name][en] = ratio_list
                if inlier_ratio_full is not None:
                    ratio_dict[name][f"{en}_full"] = inlier_ratio_full
                # get mean and median
                if self.if_mean or self.if_median:
                    temp_dict = self.get_mean_median(
                        en, arr.flatten(), mean=self.if_mean, median=self.if_median
                    )
                    ratio_dict[name].update(temp_dict)
                if if_print:
                    print(f"  - exp: {en}")
                #         print(f"arr shape: {arr.shape}")
                if if_print:
                    print(f"{ratio_list}")

        if if_print:
            print(f"ratio_dict: {ratio_dict}")
        self.ratio_dict = ratio_dict
        pass

    @staticmethod
    def get_highlights_table(reverse_arr, line_list, top_k=1):
        ## highligh the numbers
        # top_k = 2
        # reverse_arr = np.array([1, -1, -1]*4)
        table_nums = np.array(line_list) * reverse_arr

        def find_orders(x):  # for 1d array
            """
            # https://github.com/numpy/numpy/issues/8757
            """
            idx = np.empty(len(x), np.intp)
            idx[np.argsort(x)] = np.arange(len(x))[::-1]  ## reverse order, 0 is largest
            return idx

        table_argsort = np.array([find_orders(nums) for nums in table_nums.transpose()])
        table_highlights = table_argsort.transpose() < top_k
        # print(f"table_argsort: {table_argsort}")
        # print(f"table_argsort: {table_highlights}")
        return {"table_argsort": table_argsort, "table_highlights": table_highlights}

    def print_tables(self, if_name=True, table_list=[""]):
        def get_numbers(result_dict, entry_dict):
            nums = []
            for i, en in enumerate(entry_dict):
                idxs = entry_dict[en]
                print(f"entry: {en}, idxs: {idxs}, nums: {nums}")
                nums.extend([result_dict[en][i] for i in idxs])
            return nums
            pass

        def list_to_style(item, a_list, seperator=" & ", highlights=None):
            if highlights is None:
                highlights = np.zeros(len(a_list))
            highli = lambda x, hi: f"\\textbf{{{x:.3f}}}" if hi else f"{x:.3f}"
            line = (
                f"{item} & "
                + seperator.join([highli(i, hi) for (i, hi) in zip(a_list, highlights)])
                + "\n"
            )
            line += "\\\\ \\hline"
            return line
            # table_body.append(line)

        ## mapping models
        ratio_dict = self.ratio_dict
        name_dict = self.config["data"]["symbol_dict"]["models"] if if_name else None
        models_desc = self.config["data"]["symbol_dict"]["models"]
        for table in table_list:
            table_body = []
            spec = self.config["output"].get(table, None)
            if spec is None:
                continue
            ## print out the corresponding results
            sep = spec["sep"]
            name_list = []
            line_list = []
            for i, r in enumerate(spec["row"]):
                line = []
                for j, c in enumerate(spec["col"]):
                    seq_name = f"{r+sep+c}"
                    line.extend(get_numbers(ratio_dict[seq_name], spec["entries"]))
                    # name = name_dict[r] if if_name is not None else r
                    # name = seq_name
                    # name = [r][0] if if_name is not None else r

                    name = name_dict.get(r, None)
                    name = name[0] if name is not None else r

                name_list.append(name)
                line_list.append(line)
                # table_row = list_to_style(item=name, a_list=line)
                # table_body.append(table_row)

            highlights = spec.get('highlight', False)
            if highlights == True:
                reverse_arr = np.array([1, -1, -1] * 2)
                data = self.get_highlights_table(reverse_arr, line_list, top_k=1)
                table_highlights = data["table_highlights"]
            else:
                table_highlights = np.zeros_like(np.array(line_list))

            table_body = [
                list_to_style(item=name, a_list=line, highlights=hi)
                for (name, line, hi) in zip(name_list, line_list, table_highlights)
            ]
            table_ready = "\n".join(table_body)
            ## just print
            print(f"table: {table}")
            print(table_ready)
        pass

    @staticmethod
    def plot_table_for_metrics(exp, constraint=None, if_table=True):
        """
        input:
            constraint: list [the index to extract]
        """
        seperator = " & "
        print(f"=== exp: {exp} ===")
        table_body = []
        for i, name in enumerate(exp_names):
            exp_dict = ratio_dict[name]
            line = ""
            # for exp in err_mat:
            a_list = [f"{i:.3f}" for i in exp_dict[exp]]
            if constraint is not None:
                a_list = [a_list[i] for i in constraint]
                a_list = np.array(a_list).tolist()
            # print(a_list)
            line += f"({i+1}) & " + seperator.join(a_list) + "\n"
            line = f"" + line + "\\\\ \\hline"
            table_body.append(line)
        #     print(f"line: {line}")

        # titles
        if if_table:
            table = table_begin + "\n".join(table_body) + table_end
        else:
            table = "\n".join(table_body)
        return table


    @staticmethod
    def read_file_to_list(folder_list, result_names, allow_pickle=False):
        # load npz
        assert len(folder_list) == len(result_names)
        exp_list = []
        for exp, result_name in zip(folder_list, result_names):
            err_dict = np.load(exp + "/" + result_name, allow_pickle=allow_pickle)
            exp_list.append(err_dict)
        print(f"len of exp_list: {len(exp_list)}")
        return exp_list


class Val_pipeline_frontend(object):
    def __init__(self, config, device="cpu"):
        self.config = config
        self.net_dict = {}
        self.device = device
        ## for net_SP
        self.net_SP_helper = {}

    def add_net(self, name, net):
        self.net_dict[name] = net

    def net_toeval(self):
        """
        # convert all the nets to eval model
        """
        for i, name in enumerate(self.net_dict):
            self.net_dict[name] = self.net_dict[name].eval()
        pass

    def load_net_deepF(self, name="net_deepF"):
        from train_good_corr_4_vals_goodF_baseline import prepare_model
        from utils.loader import modelLoader

        device = self.device
        config = self.config
        img_zoom_xy = (
            config["data"]["preprocessing"]["resize"][1]
            / config["data"]["image"]["size"][1],
            config["data"]["preprocessing"]["resize"][0]
            / config["data"]["image"]["size"][0],
        )
        model_params = {
            "depth": config["model"]["depth"],
            "img_zoom_xy": img_zoom_xy,
            "image_size": config["data"]["image"]["size"],
            "quality_size": config["model"]["quality_size"],
            "if_quality": config["model"]["if_quality"],
            "if_img_des_to_pointnet": config["model"]["if_img_des_to_pointnet"],
            "if_goodCorresArch": config["model"]["if_goodCorresArch"],
            "if_img_feat": config["model"]["if_img_feat"],
            "if_cpu_svd": config["model"]["if_cpu_svd"],
            "if_learn_offsets": config["model"]["if_learn_offsets"],
            "if_tri_depth": config["model"]["if_tri_depth"],
            "if_sample_loss": config["model"]["if_sample_loss"],
        }
        net = modelLoader(config["model"]["name"], **model_params)
        net, optimizer, n_iter, n_iter_val = prepare_model(
            config, net, device, n_iter=0, n_iter_val=0, net_postfix=""
        )
        self.net_dict[name] = net
        pass

    ## superpoint
    def load_net_SP(self, name="net_SP"):
        config = self.config
        device = self.device
        SP_params = {
            "out_num_points": 2000,
            "patch_size": 5,
            "device": device,
            "nms_dist": 4,
            "conf_thresh": 0.015,
        }
        from models.model_utils import SuperPointNet_process
        from models.model_wrap import PointTracker
        from models.SuperPointNet_gauss2 import SuperPointNet_gauss2
        from train_good_corr_4_vals_goodF_baseline import prepare_model

        SP_processer = SuperPointNet_process(**SP_params)
        SP_tracker = PointTracker(max_length=2, nn_thresh=1.2)
        net_SP = SuperPointNet_gauss2()

        net_SP, optimizer_SP, n_iter_SP, n_iter_val_SP = prepare_model(
            config,
            net_SP,
            device,
            n_iter=0,
            n_iter_val=0,
            net_postfix="_SP",
            train=False,
        )

        logging.info("+++[Train]+++  training superpoint")
        ## put to class
        self.net_SP_helper = {"SP_processer": SP_processer, "SP_tracker": SP_tracker}
        self.net_dict[name] = net_SP
        pass

    def eval_one_sample(self, sample):
        import torch
        import dsac_tools.utils_F as utils_F  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
        import dsac_tools.utils_opencv as utils_opencv  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
        import dsac_tools.utils_vis as utils_vis  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
        import dsac_tools.utils_misc as utils_misc  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
        import dsac_tools.utils_geo as utils_geo  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
        from train_good_utils import val_rt, get_matches_from_SP

        # params
        config = self.config
        net_dict = self.net_dict
        if_SP = self.config["model"]["if_SP"]
        if_quality = self.config["model"]["if_quality"]
        device = self.device
        net_SP_helper = self.net_SP_helper

        task = "validating"
        imgs = sample["imgs"]  # [batch_size, H, W, 3]
        Ks = sample["K"].to(device)  # [batch_size, 3, 3]
        K_invs = sample["K_inv"].to(device)  # [batch_size, 3, 3]
        batch_size = Ks.size(0)
        scene_names = sample["scene_name"]
        frame_ids = sample["frame_ids"]
        scene_poses = sample[
            "relative_scene_poses"
        ]  # list of sequence_length tensors, which with size [batch_size, 4, 4]; the first being identity, the rest are [[R; t], [0, 1]]
        if config["data"]["read_what"]["with_X"]:
            Xs = sample[
                "X_cam2s"
            ]  # list of [batch_size, 3, Ni]; only support batch_size=1 because of variable points Ni for each sample
        # sift_kps, sift_deses = sample['sift_kps'], sample['sift_deses']
        assert sample["get_flags"]["have_matches"][
            0
        ].numpy(), "Did not find the corres files!"
        matches_all, matches_good = sample["matches_all"], sample["matches_good"]
        quality_all, quality_good = sample["quality_all"], sample["quality_good"]

        delta_Rtijs_4_4 = scene_poses[
            1
        ].float()  # [batch_size, 4, 4], asserting we have 2 frames where scene_poses[0] are all identities
        E_gts, F_gts = sample["E"], sample["F"]
        pts1_virt_normalizedK, pts2_virt_normalizedK = (
            sample["pts1_virt_normalized"].to(device),
            sample["pts2_virt_normalized"].to(device),
        )
        pts1_virt_ori, pts2_virt_ori = (
            sample["pts1_virt"].to(device),
            sample["pts2_virt"].to(device),
        )
        # pts1_virt_ori, pts2_virt_ori = sample['pts1_velo'].to(device), sample['pts2_velo'].to(device)

        # Get and Normalize points
        if if_SP:
            net_SP = net_dict["net_SP"]
            SP_processer, SP_tracker = (
                net_SP_helper["SP_processer"],
                net_SP_helper["SP_tracker"],
            )
            xs, offsets, quality = get_matches_from_SP(
                sample["imgs_grey"], net_SP, SP_processer, SP_tracker
            )
            matches_use = xs + offsets
            # matches_use = xs + offsets
            quality_use = quality
        else:
            # Get and Normalize points
            matches_use = matches_good  # [SWITCH!!!]
            quality_use = quality_good.to(device) if if_quality else None  # [SWITCH!!!]

        ## process x1, x2
        matches_use = matches_use.to(device)

        N_corres = matches_use.shape[1]  # 1311 for matches_good, 2000 for matches_all
        x1, x2 = (
            matches_use[:, :, :2],
            matches_use[:, :, 2:],
        )  # [batch_size, N, 2(W, H)]
        x1_normalizedK = utils_misc._de_homo(
            torch.matmul(
                torch.inverse(Ks), utils_misc._homo(x1).transpose(1, 2)
            ).transpose(1, 2)
        )  # [batch_size, N, 2(W, H)], min/max_X=[-W/2/f, W/2/f]
        x2_normalizedK = utils_misc._de_homo(
            torch.matmul(
                torch.inverse(Ks), utils_misc._homo(x2).transpose(1, 2)
            ).transpose(1, 2)
        )  # [batch_size, N, 2(W, H)], min/max_X=[-W/2/f, W/2/f]
        matches_use_normalizedK = torch.cat((x1_normalizedK, x2_normalizedK), 2)

        matches_use_ori = torch.cat((x1, x2), 2)

        # Get image feats
        if config["model"]["if_img_feat"]:
            imgs = sample["imgs"]  # [batch_size, H, W, 3]
            imgs_stack = ((torch.cat(imgs, 3).float() - 127.5) / 127.5).permute(
                0, 3, 1, 2
            )

        qs_scene = sample["q_scene"].to(device)  # [B, 4, 1]
        ts_scene = sample["t_scene"].to(device)  # [B, 3, 1]
        qs_cam = sample["q_cam"].to(device)  # [B, 4, 1]
        ts_cam = sample["t_cam"].to(device)  # [B, 3, 1]

        t_scene_scale = torch.norm(ts_scene, p=2, dim=1, keepdim=True)

        # image_height, image_width = config['data']['image']['size'][0], config['data']['image']['size'][1]
        # mask_x1 = (matches_use_ori[:, :, 0] > (image_width/8.*3.)).byte() & (matches_use_ori[:, :, 0] < (image_width/8.*5.)).byte()
        # mask_x2 = (matches_use_ori[:, :, 2] > (image_width/8.*3.)).byte() & (matches_use_ori[:, :, 2] < (image_width/8.*5.)).byte()
        # mask_y1 = (matches_use_ori[:, :, 1] > (image_height/8.*3.)).byte() & (matches_use_ori[:, :, 1] < (image_height/8.*5.)).byte()
        # mask_y2 = (matches_use_ori[:, :, 3] > (image_height/8.*3.)).byte() & (matches_use_ori[:, :, 3] < (image_height/8.*5.)).byte()
        # mask_center = (~(mask_x1 & mask_y1)) & (~(mask_x2 & mask_y2))
        # matches_use_ori = (mask_center.float()).unsqueeze(-1) * matches_use_ori + torch.tensor([image_width/2., image_height/2., image_width/2., image_height/2.]).to(device).unsqueeze(0).unsqueeze(0) * (1- (mask_center.float()).unsqueeze(-1))
        # x1, x2 = matches_use_ori[:, :, :2], matches_use_ori[:, :, 2:] # [batch_size, N, 2(W, H)]

        data_batch = {
            "matches_xy_ori": matches_use_ori,
            "quality": quality_use,
            "x1_normalizedK": x1_normalizedK,
            "x2_normalizedK": x2_normalizedK,
            "Ks": Ks,
            "K_invs": K_invs,
            "matches_good_unique_nums": sample["matches_good_unique_nums"],
            "t_scene_scale": t_scene_scale,
        }
        # loss_params = {'model': config['model']['name'], 'clamp_at':config['model']['clamp_at'], 'depth': config['model']['depth']}
        loss_params = {
            "model": config["model"]["name"],
            "clamp_at": config["model"]["clamp_at"],
            "depth": config["model"]["depth"],
        }

        with torch.no_grad():
            outs = net_dict["net_deepF"](data_batch)

            pts1_eval, pts2_eval = pts1_virt_ori, pts2_virt_ori

            #     logits = outs['logits'] # [batch_size, N]
            #     logits_weights = F.softmax(logits, dim=1)
            logits_weights = outs["weights"]
            loss_E = 0.0

            F_out, T1, T2, out_a = (
                outs["F_est"],
                outs["T1"],
                outs["T2"],
                outs["out_layers"],
            )
            pts1_eval = torch.bmm(T1, pts1_virt_ori.permute(0, 2, 1)).permute(0, 2, 1)
            pts2_eval = torch.bmm(T2, pts2_virt_ori.permute(0, 2, 1)).permute(0, 2, 1)

            # pts1_eval = utils_misc._homo(F.normalize(pts1_eval[:, :, :2], dim=2))
            # pts2_eval = utils_misc._homo(F.normalize(pts2_eval[:, :, :2], dim=2))

            loss_layers = []
            losses_layers = []
            # losses = utils_F.compute_epi_residual(pts1_eval, pts2_eval, F_est, loss_params['clamp_at']) #- res.mean()
            # losses_layers.append(losses)
            # loss_all = losses.mean()
            # loss_layers.append(loss_all)
            out_a.append(F_out)
            loss_all = 0.0
            for iter in range(loss_params["depth"]):
                losses = utils_F.compute_epi_residual(
                    pts1_eval, pts2_eval, out_a[iter], loss_params["clamp_at"]
                )
                # losses = utils_F._YFX(pts1_eval, pts2_eval, out_a[iter], if_homo=True, clamp_at=loss_params['clamp_at'])
                losses_layers.append(losses)
                loss = losses.mean()
                loss_layers.append(loss)
                loss_all += loss

            loss_all = loss_all / len(loss_layers)

            F_ests = T2.permute(0, 2, 1).bmm(F_out.bmm(T1))
            E_ests = Ks.transpose(1, 2) @ F_ests @ Ks

            last_losses = losses_layers[-1].detach().cpu().numpy()
            print(last_losses)
            print(np.amax(last_losses, axis=1))

        # E_ests_list = []
        # for x1_single, x2_single, K, w in zip(x1, x2, Ks, logits_weights):
        #     E_est = utils_F._E_from_XY(x1_single, x2_single, K, torch.diag(w))
        #     E_ests_list.append(E_est)
        # E_ests = torch.stack(E_ests_list).to(device)
        # F_ests = utils_F._E_to_F(E_ests, Ks)
        K_np = Ks.cpu().detach().numpy()
        x1_np, x2_np = x1.cpu().detach().numpy(), x2.cpu().detach().numpy()
        E_est_np = E_ests.cpu().detach().numpy()
        F_est_np = F_ests.cpu().detach().numpy()
        delta_Rtijs_4_4_cpu_np = delta_Rtijs_4_4.cpu().numpy()

        # Tests and vis
        idx = 0
        img1 = imgs[0][idx].numpy().astype(np.uint8)
        img2 = imgs[1][idx].numpy().astype(np.uint8)
        img1_rgb, img2_rgb = img1, img2
        img1_rgb_np, img2_rgb_np = img1, img2
        im_shape = img1.shape
        x1 = x1_np[idx]
        x2 = x2_np[idx]
        #         utils_vis.draw_corr(img1, img2, x1, x2)

        delta_Rtij = delta_Rtijs_4_4_cpu_np[idx]
        print("----- delta_Rtij", delta_Rtij)
        delta_Rtij_inv = np.linalg.inv(delta_Rtij)
        K = K_np[idx]
        F_gt_th = F_gts[idx].cpu()
        F_gt = F_gt_th.numpy()
        E_gt_th = E_gts[idx].cpu()
        E_gt = E_gt_th.numpy()
        F_est = F_est_np[idx]
        E_est = E_est_np[idx]

        unique_rows_all, unique_rows_all_idxes = np.unique(
            np.hstack((x1, x2)), axis=0, return_index=True
        )
        mask_sample = np.random.choice(x1.shape[0], 100)
        angle_R = utils_geo.rot12_to_angle_error(np.eye(3), delta_Rtij_inv[:3, :3])
        angle_t = utils_geo.vector_angle(
            np.array([[0.0], [0.0], [1.0]]), delta_Rtij_inv[:3, 3:4]
        )
        print(
            ">>>>>>>>>>>>>>>> Between frames: The rotation angle (degree) %.4f, and translation angle (degree) %.4f"
            % (angle_R, angle_t)
        )
        utils_vis.draw_corr(
            img1_rgb,
            img2_rgb,
            x1[mask_sample],
            x2[mask_sample],
            linewidth=2.0,
            title="Sample of 100 corres.",
        )

        #         ## Baseline: 8-points
        #         M_8point, error_Rt_8point, mask2_8point, E_est_8point = utils_opencv.recover_camera_opencv(K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.01)

        ## Baseline: 5-points
        five_point = False
        M_opencv, error_Rt_opencv, mask2, E_return = utils_opencv.recover_camera_opencv(
            K, x1, x2, delta_Rtij_inv, five_point=five_point, threshold=0.01
        )

        if five_point:
            E_est_opencv = E_return
            F_est_opencv = utils_F.E_to_F_np(E_est_opencv, K)
        else:
            E_est_opencv, F_est_opencv = E_return[0], E_return[1]

        ## Check geo dists
        print(f"K: {K}")
        x1_normalizedK = utils_misc.de_homo_np(
            (np.linalg.inv(K) @ utils_misc.homo_np(x1).T).T
        )
        x2_normalizedK = utils_misc.de_homo_np(
            (np.linalg.inv(K) @ utils_misc.homo_np(x2).T).T
        )
        K_th = torch.from_numpy(K)
        F_gt_normalized = K_th.t() @ F_gt_th @ K_th  # Should be identical to E_gts[idx]

        geo_dists = utils_F._sym_epi_dist(
            F_gt_normalized,
            torch.from_numpy(x1_normalizedK),
            torch.from_numpy(x2_normalizedK),
        ).numpy()
        geo_thres = 1e-4
        mask_in = geo_dists < geo_thres
        mask_out = geo_dists >= geo_thres

        mask_sample = mask2
        print(mask2.shape)
        np.set_printoptions(precision=8, suppress=True)

        ## Ours: Some analysis
        print("----- Oursssssssssss")
        scores_ori = logits_weights.cpu().numpy().flatten()
        import matplotlib.pyplot as plt

        plt.hist(scores_ori, 100)
        plt.show()
        sort_idxes = np.argsort(scores_ori[unique_rows_all_idxes])[::-1]
        scores = scores_ori[unique_rows_all_idxes][sort_idxes]
        num_corr = 100
        mask_conf = sort_idxes[:num_corr]
        # mask_sample = np.array(range(x1.shape[0]))[mask_sample][:20]

        utils_vis.draw_corr(
            img1_rgb,
            img2_rgb,
            x1[unique_rows_all_idxes],
            x2[unique_rows_all_idxes],
            linewidth=2.0,
            title=f"All {unique_rows_all_idxes.shape[0]} correspondences",
        )

        utils_vis.draw_corr(
            img1_rgb,
            img2_rgb,
            x1[unique_rows_all_idxes][mask_conf, :],
            x2[unique_rows_all_idxes][mask_conf, :],
            linewidth=2.0,
            title=f"Ours top {num_corr} confidents",
        )
        #         print('(%d unique corres)'%scores.shape[0])
        utils_vis.show_epipolar_rui_gtEst(
            x2[unique_rows_all_idxes][mask_conf, :],
            x1[unique_rows_all_idxes][mask_conf, :],
            img2_rgb,
            img1_rgb,
            F_gt.T,
            F_est.T,
            weights=scores_ori[unique_rows_all_idxes][mask_conf],
            im_shape=im_shape,
            title_append="Ours top %d with largest score points" % mask_conf.shape[0],
        )
        print(f"F_gt: {F_gt/F_gt[2, 2]}")
        print(f"F_est: {F_est/F_est[2, 2]}")
        error_Rt_est_ours, epi_dist_mean_est_ours, _, _, _, _, _, M_estW = val_rt(
            idx,
            K,
            x1,
            x2,
            E_est,
            E_gt,
            F_est,
            F_gt,
            delta_Rtij,
            five_point=False,
            if_opencv=False,
        )
        print(
            "Recovered by ours (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f"
            % (error_Rt_est_ours[0], error_Rt_est_ours[1])
        )
        #         print(epi_dist_mean_est_ours, np.mean(epi_dist_mean_est_ours))
        print(
            "%.2f, %.2f"
            % (
                np.sum(epi_dist_mean_est_ours < 0.1) / epi_dist_mean_est_ours.shape[0],
                np.sum(epi_dist_mean_est_ours < 1) / epi_dist_mean_est_ours.shape[0],
            )
        )

        ## OpenCV: Some analysis
        corres = np.hstack((x1[mask_sample, :], x2[mask_sample, :]))

        unique_rows = np.unique(corres, axis=0) if corres.shape[0] > 0 else corres

        opencv_name = "5-point" if five_point else "8-point"
        utils_vis.draw_corr(
            img1_rgb,
            img2_rgb,
            x1[mask_sample, :],
            x2[mask_sample, :],
            linewidth=2.0,
            title=f"OpenCV {opencv_name} inliers",
        )

        print(
            "----- OpenCV %s (%d unique inliers)" % (opencv_name, unique_rows.shape[0])
        )
        utils_vis.show_epipolar_rui_gtEst(
            x2[mask_sample, :],
            x1[mask_sample, :],
            img2_rgb,
            img1_rgb,
            F_gt.T,
            F_est_opencv.T,
            weights=scores_ori[mask_sample],
            im_shape=im_shape,
            title_append="OpenCV 5-point with its inliers",
        )
        print(F_gt / F_gt[2, 2])
        print(F_est_opencv / F_est_opencv[2, 2])
        error_Rt_est_5p, epi_dist_mean_est_5p, _, _, _, _, _, M_estOpenCV = val_rt(
            idx,
            K,
            x1,
            x2,
            E_est_opencv,
            E_gt,
            F_est_opencv,
            F_gt,
            delta_Rtij,
            five_point=False,
            if_opencv=False,
        )
        print(
            "Recovered by OpenCV %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f"
            % (opencv_name, error_Rt_est_5p[0], error_Rt_est_5p[1])
        )
        print(
            "%.2f, %.2f"
            % (
                np.sum(epi_dist_mean_est_5p < 0.1) / epi_dist_mean_est_5p.shape[0],
                np.sum(epi_dist_mean_est_5p < 1) / epi_dist_mean_est_5p.shape[0],
            )
        )
        # dict_of_lists['opencv5p'].append((np.sum(epi_dist_mean_est_5p<0.1)/epi_dist_mean_est_5p.shape[0], np.sum(epi_dist_mean_est_5p<1)/epi_dist_mean_est_5p.shape[0]))
        # dict_of_lists['ours'].append((np.sum(epi_dist_mean_est_ours<0.1)/epi_dist_mean_est_ours.shape[0], np.sum(epi_dist_mean_est_ours<1)/epi_dist_mean_est_ours.shape[0]))

        print("+++ GT, Opencv_5p, Ours")
        np.set_printoptions(precision=4, suppress=True)
        print(delta_Rtij_inv[:3])
        print(
            np.hstack(
                (
                    M_opencv[:, :3],
                    M_opencv[:, 3:4] / M_opencv[2, 3] * delta_Rtij_inv[2, 3],
                )
            )
        )
        print(
            np.hstack(
                (M_estW[:, :3], M_estW[:, 3:4] / M_estW[2, 3] * delta_Rtij_inv[2, 3])
            )
        )

        return {"img1_rgb": img1_rgb, "img2_rgb": img2_rgb, "delta_Rtij": delta_Rtij}


# data = eval_one_sample(config, sample)

if __name__ == "__main__":
    import logging
    import argparse
    import yaml
    from settings import EXPER_PATH

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )

    # add parser
    parser = argparse.ArgumentParser(description="Foo")

    # Training command
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", type=str)
    args = parser.parse_args(
        "configs/table_trans_rot_kitti_apollo.yaml table_rot_test".split()
    )
    print(args)

    ## load configs
    with open(args.config, "r") as f:
        config = yaml.load(f)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    print(output_dir)
    print(f"config: {config}")

    ## read results
    from utils.eval_tools import Exp_table_processor

    table_processor = Exp_table_processor(config, debug=True)
    table_processor.get_result_dict()

    ## print out table
    table_processor.print_tables(if_name=True, table_list=["table_1"])
