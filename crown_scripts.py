
from import_script import *
# ======================================================================
# 1. setup and running scripts for crown
# ======================================================================


def simple_box_filename(x_max, x_min):
    dim = len(x_max)
    X = torch.tensor([[(x_max[i] - x_min[i]) / 2. + x_min[i] for i in range(dim)]]).float()
    temp = X[0].tolist()
    new_temp = []
    for val in temp:
        new_temp.append(float('%.3f'%val))

    return str(new_temp)


def setup_crown_yaml_dkl(network_dims, mode, dim, crown_dir, global_dir_name, use_reLU, num_layers):
    if use_reLU:
        if num_layers == 1:
            data = {"attack": {"pgd_order": "skip"},
                    "bab": {"branching": {"method": "fsb"}, "timeout": 300, "get_upper_bound": "true"},
                    "data": {"dataset": None, "num_outputs": network_dims[3]},
                    "general": {"device": "cpu"},
                    "model": {
                        "name": "Customized(\"custom_functions\",\"DynModelNetRelu1\",d={}, width_1={}, width_2={},"
                                "out_dim={})".format(network_dims[0], network_dims[1], network_dims[2],
                                                     network_dims[3]),
                        "path": "models/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)},
                    "solver": {"beta-crown": {"iteration": 30}}, "specification": {"type": "bound"}}
        elif num_layers == 2:
            data = {"attack": {"pgd_order": "skip"},
                    "bab": {"branching": {"method": "fsb"}, "timeout": 300, "get_upper_bound": "true"},
                    "data": {"dataset": None, "num_outputs": network_dims[3]},
                    "general": {"device": "cpu"},
                    "model": {
                        "name": "Customized(\"custom_functions\",\"DynModelNetRelu2\",d={}, width_1={}, width_2={},"
                                "out_dim={})".format(network_dims[0], network_dims[1], network_dims[2],
                                                     network_dims[3]),
                        "path": "models/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)},
                    "solver": {"beta-crown": {"iteration": 30}}, "specification": {"type": "bound"}}
        else:
            data = {"attack": {"pgd_order": "skip"},
                    "bab": {"branching": {"method": "fsb"}, "timeout": 300, "get_upper_bound": "true"},
                    "data": {"dataset": None, "num_outputs": network_dims[3]},
                    "general": {"device": "cpu"},
                    "model": {
                        "name": "Customized(\"custom_functions\",\"DynModelNetRelu3\",d={}, width_1={}, width_2={},"
                                "out_dim={})".format(network_dims[0], network_dims[1], network_dims[2],
                                                     network_dims[3]),
                        "path": "models/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)},
                    "solver": {"beta-crown": {"iteration": 30}}, "specification": {"type": "bound"}}
    else:
        data = {"attack": {"pgd_order": "skip"},
                "bab": {"branching": {"method": "fsb"}, "timeout": 300, "get_upper_bound": "true"},
                "data": {"dataset": None, "num_outputs": network_dims[3]},
                "general": {"device": "cpu"},
                "model": {
                    "name": "Customized(\"custom_functions\",\"DynModelNetTanh1\",d={}, width_1={}, width_2={},"
                            "out_dim={})".format(network_dims[0], network_dims[1], network_dims[2], network_dims[3]),
                    "path": "models/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)},
                "solver": {"beta-crown": {"iteration": 30}}, "specification": {"type": "bound"}}

    file_name = crown_dir + "/exp_configs/nn_mode_{}_dim_{}.yaml".format(mode, dim)
    file = open(file_name, "w")
    yaml.dump(data, file)


def run_dkl_crown_parallel(region_area, crown_dir, d, mode, dim, region_idx):
    x_min = [region_area[k][0] for k in list(region_area)]
    x_max = [region_area[k][1] for k in list(region_area)]

    file_addon = simple_box_filename(x_max, x_min)

    replacement_dataset = "Customized(\"custom_functions\",\"simple_box_data_nD\", x_max = {}, x_min = {})".format(
        x_max, x_min)

    file_name = crown_dir + "/exp_configs/nn_mode_{}_dim_{}.yaml".format(mode, dim)
    des_yaml = crown_dir + "/exp_configs/nn_mode_{}_{}_{}.yaml".format(mode, dim, region_idx)
    shutil.copyfile(file_name, des_yaml)

    file = open(des_yaml, "r")
    data = yaml.full_load(file)
    data["data"]["dataset"] = replacement_dataset
    file.close()

    file = open(des_yaml, "w")
    yaml.dump(data, file)
    file.close()

    # ensure activate is run on the correct mode
    file_name = crown_dir + "/activate.sh"
    des_dir = crown_dir + "/{}_{}".format(region_idx, dim)
    des_shell = des_dir + "/activate.sh"
    if not os.path.isdir(des_dir):
        os.mkdir(des_dir)
    else:
        if os.path.exists(des_shell):
            os.remove(des_shell)  # this is a file from a previously canceled run, delete it
    shutil.copyfile(file_name, des_shell)

    file = open(des_shell, "r")
    data = file.readlines()
    replacement_line = "cd ../\npython3 abcrown.py --config exp_configs/nn_mode_{}_{}_{}.yaml\n".format(mode, dim,
                                                                                                        region_idx)
    data[5] = replacement_line
    file.close()

    os.chmod(des_shell, 0o777)
    file = open(des_shell, "w+")
    file.writelines(data)
    file.close()

    os.chdir(des_dir)
    subprocess.call("./activate.sh", shell=True)
    os.chdir(crown_dir)

    # remove unneeded yamls to not clutter files
    os.remove(des_yaml)
    os.remove(des_shell)
    os.rmdir(des_dir)

    file_name_A_mats = crown_dir + "/transform_mats_{}.npy".format(file_addon)
    A_mats = np.load(file_name_A_mats)

    lA = A_mats[0]
    uA = A_mats[1]

    file_name_bias = crown_dir + "/bias_mats_{}.npy".format(file_addon)
    bias_mats = np.load(file_name_bias)

    l_bias = bias_mats[0].reshape(1, d)
    u_bias = bias_mats[1].reshape(1, d)

    file_name_bounds = crown_dir + "/bounds_{}.npy".format(file_addon)
    boundaries = np.load(file_name_bounds)
    post_xmin = boundaries[0][0]
    post_xmax = boundaries[1][0]

    linear_transform = (lA, uA, l_bias, u_bias, post_xmin, post_xmax)

    # also remove these files to not clutter folders
    os.remove(file_name_A_mats)
    os.remove(file_name_bias)
    os.remove(file_name_bounds)

    return linear_transform

