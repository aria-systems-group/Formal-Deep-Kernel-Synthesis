# This script generates dynamics data, regresses a deep kernel and gets bounds on the NN portion of the kernel
# it is run by calling python3 setup_nn_bounds.py [THREADS] [EXPERIMENT]
# where [THREADS] is how many parallel threads will be used during bounding and EXPERIMENT is which experiment to run,
# defined by numbers

from generic_fnc import *
from dynamics_script import *
from gp_scripts import *
from space_discretize_scripts import discretize_space
from gp_parallel import *

from juliacall import Main as jl
# jl.seval("using Pkg")
# Pkg_add = jl.seval("Pkg.add")
# Pkg_add(url="https://github.com/aria-systems-group/PosteriorBounds.jl")
jl.seval("using PosteriorBounds")
theta_vectors = jl.seval("PosteriorBounds.theta_vectors")

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR + experiment_type
crown_dir = EXPERIMENT_DIR + "/alpha-beta-CROWN/complete_verifier"

reuse_regions = False
use_regular_gp = False

epochs = 800
learning_rate = 0.001
measurement_dist = None
process_dist = {"sig": 0.01, "dist": "normal"}
width_1 = 64
width_2 = 64
num_layers = 2
random_Seed = 11
use_reLU = True
known_fnc = None
nn_epochs = 1000
nn_lr = 1e-4

threads = int(sys.argv[1])
experiment_number = int(sys.argv[2])

# ======================================================================
# 0. Setup space and dynamics
# ======================================================================

if experiment_number == 2:
    # 2D experiment, 4 modes, 200 data points per mode
    global_dir_name = "sys_2d"
    unknown_modes_list = [g_2d_mode0, g_2d_mode1, g_2d_mode2, g_2d_mode3]
    X = {"x1": [-2., 2.], "x2": [-2., 2.]}
    GP_data_points = 200
elif experiment_number == 1:
    # 3D experiment, 5 modes, 1000 data points per mode
    global_dir_name = "sys_3d_"
    process_dist = {"mu": [0., 0., 0.], "sig": [0.01, 0.01, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    GP_data_points = 1000
    nn_epochs = 4000
    epochs = 400
elif experiment_number == 3:
    # 3D experiment, 5 modes, 2000 data points per mode
    global_dir_name = "sys_3d"
    process_dist = {"mu": [0., 0., 0.], "sig": [0.01, 0.01, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    GP_data_points = 2000
    nn_epochs = 4000
    epochs = 400
elif experiment_number == 4:
    # 3D experiment, 5 modes, 2000 data points per mode, uses a standard GP rather than a deep kernel
    use_regular_gp = True
    global_dir_name = "sys_3d_gp"
    process_dist = {"mu": [0., 0., 0.], "sig": [0.01, 0.01, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    GP_data_points = 2000
    epochs = 400
elif experiment_number == 5:
    # 5D experiement, 4 modes, 15000 data points per mode
    global_dir_name = "sys_5d"
    unknown_modes_list = [g_5d_mode0, g_5d_mode1, g_5d_mode2, g_5d_mode3]
    X = {"x1": [-2., 2.], "x2": [-2., 2.], "x3": [-0.4, 0.4], "x4": [-0.4, 0.4], "x5": [-0.4, 0.4]}
    GP_data_points = 15000
    width_1 = 128
    nn_epochs = 10000
    epochs = 800
    specification = "!b U a"
    unsafe_set = [None]
    goal_set = [[[1.5, 2.], [0.5, 2], [-0.4, 0.4], [-0.4, 0.4], [-0.4, 0.4]]]
    region_labels = {"a": goal_set, "b": unsafe_set}
else:
    exit()

modes = [i for i in range(len(unknown_modes_list))]

global_exp_dir = exp_dir + "/" + global_dir_name

if os.path.isdir(exp_dir):
    if not os.path.isdir(global_exp_dir):
        os.mkdir(global_exp_dir)
else:
    os.mkdir(exp_dir)
    os.mkdir(global_exp_dir)

# abstraction grid size
grid_len = 0.125
grid_size, large_grid = get_grid_info(X, grid_len)

# Neural Network dimensions and training info
d = len(X)  # dimensionality
out_dim = d

network_dims = [d, width_1, width_2, out_dim]

extents, boundaries = discretize_space(X, grid_size)
extents.pop()
states = [i for i in range(len(extents))]
num_extents = len(extents)
print(f"Number of extents = {num_extents}")

# =====================================================================================
# 1. Generate training data and learn the functions with deep kernel GPs
# =====================================================================================
unknown_dyn_gp = []
region_info = []
for mode in modes:
    unknown_dyn_gp.append(None)
    region_info.append(None)


file_name = global_exp_dir + "/training_data.pkl"
if reuse_regions and os.path.exists(file_name):
    print("Loading previous training data...")
    all_data = dict_load(file_name)
else:
    print("Generating training data...")
    all_data = [[None, None] for _ in modes]
    for mode in modes:
        g = unknown_modes_list[mode]
        x_train, y_train = generate_training_data(g, X, GP_data_points, known_fnc=known_fnc,
                                                  random_seed=random_Seed + mode, process_dist=process_dist,
                                                  measurement_dist=measurement_dist)

        all_data[mode][0] = x_train
        all_data[mode][1] = y_train
    # don't carry around large variables that are stored elsewhere
    del x_train, y_train
    dict_save(file_name, all_data)

keys = list(X)

gp_file_dir = global_exp_dir + "/gp_parameters"
if not os.path.isdir(gp_file_dir):
    os.mkdir(gp_file_dir)

file_name = gp_file_dir + "/deep_gp_data"
if False and reuse_regions and os.path.exists(file_name + '_0_0.pth'):
    # TODO, figure out why this doesn't save/load properly
    print("Loading previous GP data...")
    unknown_dyn_gp = dkl_load(file_name, unknown_dyn_gp, all_data, use_reLU, num_layers, network_dims)

tic = time.perf_counter()
for mode in modes:
    if unknown_dyn_gp[mode] is None:
        unknown_dyn_gp[mode], region_info[mode] = deep_kernel_fixed_nn_local(all_data, mode, keys, use_reLU,
                                                                             num_layers, network_dims, crown_dir,
                                                                             global_dir_name, large_grid, X,
                                                                             process_noise=process_dist,
                                                                             lr=learning_rate,
                                                                             training_iter=epochs,
                                                                             random_seed=(mode + 2) * random_Seed,
                                                                             epochs=nn_epochs, nn_lr=nn_lr,
                                                                             use_regular_gp=use_regular_gp)

        print(f'Finished deep kernel regression for mode {mode+1}\n')
        dkl_save(file_name, unknown_dyn_gp, mode, True)
toc = time.perf_counter()
print(f"It took {toc - tic} seconds to train all the Deep Kernel GP models.")

# =====================================================================================
# 2. Get posteriors of NN and save to numpy files
# =====================================================================================

num_modes = len(modes)
num_dims = len(X)
num_regions = len(extents)
num_sub_regions = len(region_info[0])

nn_bounds_dir = global_exp_dir + "/nn_bounds"
if not os.path.isdir(nn_bounds_dir):
    os.mkdir(nn_bounds_dir)

filename = global_exp_dir + "/general_info"
np.save(filename, np.array([num_modes, num_dims, num_sub_regions, num_regions]))

tic = time.perf_counter()
linear_bounds = [[[] for dim in range(num_dims)] for idx in range(num_regions)]
for mode in modes:
    region_inf = region_info[mode]

    filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}"

    if not use_regular_gp:
        linear_bounds = run_dkl_in_parallel_just_bounds(extents, mode, range(num_dims), out_dim, crown_dir, EXPERIMENT_DIR,
                                                        linear_bounds, threads=threads, use_regular_gp=use_regular_gp)
        print(f"Finished bounding the NN for mode {mode+1}")
    else:
        if mode == 0:
            linear_bounds = run_dkl_in_parallel_just_bounds(extents, mode, range(num_dims), out_dim, crown_dir,
                                                            EXPERIMENT_DIR, linear_bounds, threads=threads,
                                                            use_regular_gp=use_regular_gp)
    np.save(filename, linear_bounds)
    for sub_idx in range(num_sub_regions):
        region = region_inf[sub_idx][2]
        x_data = torch.tensor(np.transpose(region_inf[sub_idx][0]), dtype=torch.float32)
        y_data = np.transpose(region_inf[sub_idx][1])
        n_obs = max(np.shape(y_data))

        # identify which extents are in this region and their indices
        specific_extents = extents_in_region(extents, region)

        for dim in range(num_dims):
            dim_region_filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_{sub_idx + 1}_{dim + 1}"

            model = unknown_dyn_gp[mode][sub_idx][dim][0]
            model.cpu()  # unfortunately needs to be on cpu to access values
            if not use_regular_gp:
                nn_portion = model.feature_extractor
                with torch.no_grad():
                    kernel_inputs = nn_portion.forward(x_data)
            else:
                kernel_inputs = x_data
            noise = model.likelihood.noise.item()
            noise_mat = noise * np.identity(np.shape(kernel_inputs)[0])

            covar_module = model.covar_module
            kernel_mat = covar_module(kernel_inputs)
            kernel_mat = kernel_mat.evaluate()
            K = kernel_mat.detach().numpy() + noise_mat
            # enforce symmetry, it is very close but causes errors when computing sig bounds
            K = (K + K.transpose()) / 2.
            K_inv = np.linalg.inv(K)  # only need to do this once per dim, yay

            y_dim = np.reshape(y_data[:, dim], (n_obs,))
            alpha_vec = K_inv @ y_dim
            length_scale = model.covar_module.base_kernel.lengthscale.item()
            output_scale = model.covar_module.outputscale.item()

            # convert to julia input structure
            x_gp = np.array(np.transpose(kernel_inputs.detach().numpy())).astype(np.float64)
            K = np.array(K)
            K_inv = np.array(K_inv)
            alpha_vec = np.array(alpha_vec)
            out_2 = output_scale
            len_2 = length_scale ** 2.
            theta_vec, theta_vec_2 = theta_vectors(x_gp, len_2)
            K_inv_s = out_2 * K_inv

            # need to save x_gp, K, K_inv, alpha_vec, out_2, len_2, theta_vec, theta_vec_2, K_inv_s all individually
            np.save(dim_region_filename+"_x_gp", x_gp)
            np.save(dim_region_filename+"_theta_vec", theta_vec)
            np.save(dim_region_filename+"_theta_vec_2", theta_vec_2)
            np.save(dim_region_filename+"_K", K)
            np.save(dim_region_filename+"_K_inv", K_inv)
            np.save(dim_region_filename+"_alpha", alpha_vec)
            np.save(dim_region_filename+"_K_inv_s", K_inv_s)
            np.save(dim_region_filename+"_kernel", np.array([out_2, len_2, ]))
            np.save(dim_region_filename+"_these_indices", np.array(specific_extents))

toc = time.perf_counter()
if not use_regular_gp:
    print(f"Finished bounding NN portion for all modes in {toc-tic} seconds, moving to Julia \n")
else:
    print(f"Finished storing needed data, moving to Julia \n")
