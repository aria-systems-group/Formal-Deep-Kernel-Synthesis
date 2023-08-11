# This script generates dynamics data, regresses a deep kernel and gets bounds on the NN portion of the kernel
# it is run by calling python3 setup_nn_bounds.py [THREADS] [EXPERIMENT]
# where [THREADS] is how many parallel threads will be used during bounding and EXPERIMENT is which experiment to run,
# defined by numbers

from generic_fnc import *
from dynamics_script import *
from gp_scripts import *
from space_discretize_scripts import discretize_space_list, merge_extent_fnc
from juliacall import Main as jl

jl.seval("using PosteriorBounds")
theta_vectors = jl.seval("PosteriorBounds.theta_vectors")

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR + experiment_type
crown_dir = EXPERIMENT_DIR + "/alpha-beta-CROWN/complete_verifier"

# define some standard variables, may be changed depending on the experiment
reuse_regions = False
use_regular_gp = False

epochs = 400
learning_rate = 0.001
measurement_dist = None
process_dist = {"sig": 0.01, "dist": "normal"}
width_1 = 64
width_2 = 64
num_layers = 2
random_Seed = 11
use_reLU = True
single_dim_nn = False
known_fnc = None
use_scaling = False
nn_epochs = 1000
nn_lr = 1e-4
merge_extents = False
merge_bounds = None
kernel_data_points = None
grid_len = 1

refinement = 0  # don't change this

threads = int(sys.argv[1])
experiment_number = int(sys.argv[2])

# ======================================================================
# 0. Define space, dynamics, and how much data to use
# ======================================================================


if experiment_number == 0:
    # 2D linear experiment, 3 modes, 200 data points per mode
    global_dir_name = "sys_2d_lin"
    process_dist = {"mu": [0., 0.], "sig": [0.01, 0.01], "dist": "multi_norm"}
    unknown_modes_list = [g_lin1, g_lin2, g_lin3]
    X = {"x1": [-4., 4.], "x2": [-4., 4.]}
    GP_data_points = 200
    nn_epochs = 200
    learning_rate = 5e-4
    grid_len = 0.125
elif experiment_number == 1:
    # 2D nonlinear dkl experiment, 4 modes, 200 data points per mode
    global_dir_name = "sys_2d"
    process_dist = {"mu": [0., 0.], "sig": [0.0001, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_2d_mode0, g_2d_mode1, g_2d_mode2, g_2d_mode3]
    X = {"x1": [-2., 2.], "x2": [-2., 2.]}
    GP_data_points = 1000
    kernel_data_points = 200
    nn_epochs = 500  # 200
    epochs = 600
    grid_len = 0.125
    use_scaling = True
    merge_extents = True
    merge_bounds = {"unsafe": [[[-1.75, -1.25], [-0.75, 1.0]],
                               [[-1.0, -0.5], [-1.25, -0.875]],
                               [[0.5, 1.125], [-1.75, -1.25]],
                               [[0.75, 1.0], [-0.5, 0.5]],
                               [[0.75, 1.75], [0.75, 1.75]]]}
elif experiment_number == 2:
    # 2D nonlinear gp experiment, 4 modes, 200 data points per mode
    global_dir_name = "sys_2d_gp"
    process_dist = {"mu": [0., 0.], "sig": [0.0001, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_2d_mode0, g_2d_mode1, g_2d_mode2, g_2d_mode3]
    X = {"x1": [-2., 2.], "x2": [-2., 2.]}
    use_regular_gp = True
    GP_data_points = 1000
    kernel_data_points = 200
    epochs = 600
    grid_len = 0.125
    merge_extents = True
    merge_bounds = {"unsafe": [[[-1.75, -1.25], [-0.75, 1.0]],
                               [[-1.0, -0.5], [-1.25, -0.875]],
                               [[0.5, 1.125], [-1.75, -1.25]],
                               [[0.75, 1.0], [-0.5, 0.5]],
                               [[0.75, 1.75], [0.75, 1.75]]]}
elif experiment_number == 3:
    # 3D experiment, 7 modes, 400 data points per mode
    global_dir_name = "dubins_sys"
    process_dist = {"mu": [0., 0., 0.], "sig": [0.0001, 0.0001, 0.0001], "dist": "multi_norm", "theta_dim": [0, 1, 2]}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5, g_3d_mode6, g_3d_mode7]
    X = {"x1": [0., 10.], "x2": [0., 2.], "x3": [-.5, .5]}
    GP_data_points = 10000
    kernel_data_points = 400
    nn_epochs = 3000
    width_1 = 128
    width_2 = 128
    num_layers = 2
    epochs = 600
    use_scaling = True
    learning_rate = 1e-4
    merge_extents = True
    merge_bounds = {"unsafe": [[[4., 6], [0., 1.], [-0.5, 0.5]]], "goal": [[[8., 10.], [0., 1.], [-0.5, 0.5]]]}
elif experiment_number == 4:
    # 3D experiment, 7 modes, 400 data points per mode
    global_dir_name = "dubins_sys_gp"
    process_dist = {"mu": [0., 0., 0.], "sig": [0.0001, 0.0001, 0.0001], "dist": "multi_norm", "theta_dim": [0, 1, 2]}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5, g_3d_mode6, g_3d_mode7]
    X = {"x1": [0., 10.], "x2": [0., 2.], "x3": [-.5, .5]}
    use_regular_gp = True
    GP_data_points = 10000
    kernel_data_points = 400
    epochs = 600
    merge_extents = True
    merge_bounds = {"unsafe": [[[4., 6], [0., 1.], [-0.5, 0.5]]], "goal": [[[8., 10.], [0., 1.], [-0.5, 0.5]]]}
elif experiment_number == 5:
    # 5D experiment, 3 modes, 50000 data points per mode but only 250 used in the kernel
    global_dir_name = "sys_5d"
    unknown_modes_list = [g_5d_mode0, g_5d_mode1, g_5d_mode2]
    process_dist = {"mu": [0., 0., 0., 0., 0.], "sig": [0.0025, 0.0025, 0.0001, 0.0001, 0.0001], "dist": "multi_norm",
                    "theta_dim": [0, 1, 2, 3, 4]}
    X = {"x1": [-2., 2.], "x2": [-2., 2.], "x3": [-0.3, 0.3], "x4": [-0.3, 0.3], "x5": [-0.3, 0.3]}
    GP_data_points = 50000
    kernel_data_points = 250
    width_1 = 64
    width_2 = 64
    num_layers = 3
    use_scaling = True
    nn_epochs = 10000
    merge_extents = True
    merge_bounds = {"unsafe": [[[-0.75, 0.0], [0.5, 2.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]],
                               [[0.5, 2.0], [-0.75, 0.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]],
                    "goal": [[[1.0, 2.0], [0.5, 2.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]]}
    single_dim_nn = True
elif experiment_number == 6:
    # 3D experiment, 7 modes, 10000 data points per mode but only 400 used in the kernel
    global_dir_name = "dubins_sys_sd"
    process_dist = {"mu": [0., 0., 0.], "sig": [0.0001, 0.0001, 0.0001], "dist": "multi_norm", "theta_dim": [0, 1, 2]}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5, g_3d_mode6, g_3d_mode7]
    X = {"x1": [0., 10.], "x2": [0., 2.], "x3": [-.5, .5]}
    GP_data_points = 10000
    kernel_data_points = 400
    nn_epochs = 5000
    width_1 = 128
    width_2 = 128
    num_layers = 2
    epochs = 600
    use_scaling = True
    learning_rate = 1e-4
    merge_extents = True
    merge_bounds = {"unsafe": [[[4., 6], [0., 1.], [-0.5, 0.5]]], "goal": [[[8., 10.], [0., 1.], [-0.5, 0.5]]]}
    single_dim_nn = True
else:
    exit()

if kernel_data_points is None:
    kernel_data_points = GP_data_points

modes = [i for i in range(len(unknown_modes_list))]

global_exp_dir = exp_dir + "/" + global_dir_name

if os.path.isdir(exp_dir):
    if not os.path.isdir(global_exp_dir):
        os.mkdir(global_exp_dir)
else:
    os.mkdir(exp_dir)
    os.mkdir(global_exp_dir)

# abstraction grid size
grid_size, large_grid = get_grid_info(X, grid_len)

# Neural Network dimensions and training info
d = len(X)  # dimensionality
out_dim = d
network_dims = [d, width_1, width_2, out_dim]

extents = discretize_space_list(X, grid_size)
if merge_extents:
    # knowledge of the space may allow some extents to be merged, do this now to remove unnecessary states
    extents = merge_extent_fnc(extents, merge_bounds, d)

filename = global_exp_dir + f"/extents_{refinement}"
np.save(filename, np.array(extents))
extents.pop()

if reuse_regions and os.path.exists(global_exp_dir + f"/nn_bounds/linear_bounds_{len(modes)}_0.npy"):
    print("Using saved DKM data to enforce identical results (i.e. seeding won't change the answer)")
    exit()

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

gp_file_dir = global_exp_dir + "/gp_parameters"
if not os.path.isdir(gp_file_dir):
    os.mkdir(gp_file_dir)

file_name = gp_file_dir + "/deep_gp_data"
if False and reuse_regions and os.path.exists(file_name + '_0_0.pth'):
    # TODO, figure out why this doesn't save/load properly
    print("Loading previous GP data...")
    unknown_dyn_gp = dkl_load(file_name, unknown_dyn_gp, all_data, use_reLU, num_layers, network_dims)

tic = time.perf_counter()
keys = list(X)
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
                                                                             use_regular_gp=use_regular_gp,
                                                                             use_scaling=use_scaling,
                                                                             kernel_data_points=kernel_data_points,
                                                                             single_dim_nn=single_dim_nn)

        print(f'Finished deep kernel regression for mode {mode + 1}\n')
        dkl_save(file_name, unknown_dyn_gp, mode, True)
toc = time.perf_counter()
print(f"It took {toc - tic} seconds to train all the Deep Kernel GP models.")

# =====================================================================================
# 2. Get posteriors of NN and save as numpy files
# =====================================================================================

num_modes = len(modes)
num_dims = len(X)
num_regions = len(extents)
num_sub_regions = len(region_info[0])

print(f"Number of extents = {num_regions}")

nn_bounds_dir = global_exp_dir + "/nn_bounds"
if not os.path.isdir(nn_bounds_dir):
    os.mkdir(nn_bounds_dir)

filename = global_exp_dir + f"/general_info"
use_personal = 0
if single_dim_nn:
    use_personal = 1
np.save(filename, np.array([num_modes, num_dims, num_sub_regions, num_regions, use_personal]))

tic = time.perf_counter()
lin_bounds = [[] for idx in range(num_regions)]  # using the same NN across all dimensions for one mode
linear_trans_m = [[] for idx in range(num_regions)]
linear_trans_b = [[] for idx in range(num_regions)]
for mode in modes:
    region_inf = region_info[mode]

    if not use_regular_gp:
        lin_bounds, linear_trans_m, linear_trans_b = run_dkl_in_parallel_just_bounds(extents, mode, out_dim, crown_dir,
                                                                                     global_dir_name, EXPERIMENT_DIR,
                                                                                     lin_bounds,
                                                                                     linear_trans_m, linear_trans_b,
                                                                                     threads=threads,
                                                                                     use_regular_gp=use_regular_gp,
                                                                                     merged=merge_bounds)
        print(f"Finished bounding the NN for mode {mode + 1}")
    else:
        if mode == 0:
            lin_bounds, linear_trans_m, linear_trans_b = run_dkl_in_parallel_just_bounds(extents, mode, out_dim,
                                                                                         crown_dir, global_dir_name,
                                                                                         EXPERIMENT_DIR,
                                                                                         lin_bounds, linear_trans_m,
                                                                                         linear_trans_b,
                                                                                         threads=threads,
                                                                                         use_regular_gp=use_regular_gp)

    filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_{refinement}"
    np.save(filename, np.array(lin_bounds))
    filename = nn_bounds_dir + f"/linear_trans_m_{mode + 1}_{refinement}"
    np.save(filename, np.array(linear_trans_m))
    filename = nn_bounds_dir + f"/linear_trans_b_{mode + 1}_{refinement}"
    np.save(filename, np.array(linear_trans_b))
    for sub_idx in range(num_sub_regions):
        region = region_inf[sub_idx][2]
        x_data = torch.tensor(np.transpose(region_inf[sub_idx][0]), dtype=torch.float32)
        y_data = np.transpose(region_inf[sub_idx][1])
        n_obs = max(np.shape(y_data))

        # identify which extents are in this region and their indices
        specific_extents = [j for j in range(0, num_regions)]  # extents_in_region(extents, region)

        for dim in range(num_dims):
            dim_region_filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_{sub_idx + 1}_{dim + 1}"

            model = unknown_dyn_gp[mode][sub_idx][dim][0]
            model.cpu()  # unfortunately needs to be on cpu to access values
            if not use_regular_gp:
                nn_portion = model.feature_extractor
                with torch.no_grad():
                    kernel_inputs = nn_portion.forward(x_data)
                if single_dim_nn:
                    kernel_inputs = torch.index_select(kernel_inputs, 1, torch.tensor(dim))
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
            np.save(dim_region_filename + "_x_gp", x_gp)
            np.save(dim_region_filename + "_theta_vec", theta_vec)
            np.save(dim_region_filename + "_theta_vec_2", theta_vec_2)
            np.save(dim_region_filename + "_K", K)
            np.save(dim_region_filename + "_K_inv", K_inv)
            np.save(dim_region_filename + "_alpha", alpha_vec)
            np.save(dim_region_filename + "_K_inv_s", K_inv_s)
            np.save(dim_region_filename + "_kernel", np.array([out_2, len_2, ]))
            np.save(dim_region_filename + f"_these_indices_{refinement}", np.array(specific_extents))

toc = time.perf_counter()

if not use_regular_gp:
    print(f"Finished bounding NN portion for all modes in {toc - tic} seconds, moving to Julia \n")
else:
    print(f"Finished storing needed data, moving to Julia \n")
