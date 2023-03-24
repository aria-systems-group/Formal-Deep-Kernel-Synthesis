import time

from generic_fnc import *

sys.path.append('/alpha-beta-CROWN/complete_verifier')

from dynamics_script import *
from gp_scripts import *
from space_discretize_scripts import *
from gp_parallel import *
from dfa_scripts import *

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR + experiment_type
crown_dir = EXPERIMENT_DIR + "/alpha-beta-CROWN/complete_verifier"

reuse_regions = False
use_fixed_nn = True
use_local_gp = False
do_synthesis = False

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
threads = 10  # number of cpu threads
nn_epochs = 1000
nn_lr = 1e-4

# ======================================================================
# 0. Setup space and dynamics
# ======================================================================
experiment_number = 3
if experiment_number == 0:
    # 1D gp test
    global_dir_name = "chaos_1d"
    unknown_modes_list = [crazy_1d]
    X = {"x1": [0., 6.]}
    GP_data_points = 30
    process_dist = {"sig": 0.01}
    num_layers = 2
    specification = "F(a)"
    unsafe_set = [None]
    goal_set = [{"x1": [0., 2.]}]
    region_labels = {"b": unsafe_set, "a": goal_set}
    learning_rate = 0.001
    epochs = 300
elif experiment_number == 1:
    # 1D gp test
    global_dir_name = "sys_1d_x_tan"
    unknown_modes_list = [g_x_sin]
    X = {"x1": [0., 10]}
    GP_data_points = 30
    process_dist = {"sig": 0.01}
    num_layers = 1
    specification = "F(a)"
    unsafe_set = [None]
    goal_set = [{"x1": [0., 2.]}]
    region_labels = {"b": unsafe_set, "a": goal_set}
    learning_rate = 0.01
    epochs = 2000
elif experiment_number == 2:
    # 2D experiment, 4 modes
    global_dir_name = "sys_2d"
    unknown_modes_list = [g_2d_mode0, g_2d_mode1, g_2d_mode2, g_2d_mode3]
    X = {"x1": [-2., 2.], "x2": [-2., 2.]}
    GP_data_points = 200
    specification = "G(!b) & F(a) & F(c)"
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0]],
                  [[-1.0, -0.5], [-1.25, -0.875]],
                  [[0.5, 1.125], [-1.75, -1.25]],
                  [[0.75, 1.0], [-0.5, 0.5]],
                  [[0.75, 1.75], [0.75, 1.75]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75]]]
    goal_c = [{"x1": [-1.75, 0.0], "x2": [-2.0, -1.5]},
              {"x1": [1.125, 2.0], "x2": [-1.75, 0.0]},
              {"x1": [-1.25, 0.0], "x2": [1.0, 1.875]}]
    region_labels = {"b": unsafe_set, "a": goal_a} #, "c": goal_c}
    goal_set = []
    goal_set.extend(goal_a)
    goal_set.extend(goal_c)
elif experiment_number == 3:
    global_dir_name = "sys_3d_retry"
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    use_local_gp = True
    GP_data_points = 2000
    nn_epochs = 4000
    epochs = 400
    specification = "!b U a"
    # unsafe_set = [[[4., 6.], [0., 1.], [-0.5, 0.5]]]  # a list of lists, each inner list being a set of bounds for the label
    # goal_set = [[[8., 10.], [0., 1.], [-0.5, 0.5]]]

    unsafe_set = [[[0., 1], [0., 1.], [-0.5, 0.5]]]  # a list of lists, each inner list being a set of bounds for the label
    goal_set = [[[3., 5.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
elif experiment_number == 3.5:
    global_dir_name = "sys_3d_too_coarse"
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]
    X = {"x1": [0., 10.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    use_local_gp = True
    GP_data_points = 2000
    nn_epochs = 4000
    epochs = 400
    specification = "!b U a"
    unsafe_set = [[[4., 6.], [0., 1.], [-0.5, 0.5]]]  # a list of lists, each inner list being a set of bounds for the label
    goal_set = [[[8., 10.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
elif experiment_number == 5:
    global_dir_name = "sys_5d"
    unknown_modes_list = [g_5d_mode0, g_5d_mode1, g_5d_mode2, g_5d_mode3]
    X = {"x1": [-2., 2.], "x2": [-2., 2.], "x3": [-0.4, 0.4], "x4": [-0.4, 0.4], "x5": [-0.4, 0.4]}
    use_local_gp = True
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
labels = region_labels

plot_labels = {"obs": unsafe_set, "goal": goal_set}

modes = [i for i in range(len(unknown_modes_list))]

global_exp_dir = exp_dir + "/" + global_dir_name

if os.path.isdir(exp_dir):
    if not os.path.isdir(global_exp_dir):
        os.mkdir(global_exp_dir)
else:
    os.mkdir(exp_dir)
    os.mkdir(global_exp_dir)

grid_len = 0.125

if len(X) < 3:
    grid_size = {k: grid_len for k in list(X)}
    large_grid = X
elif len(X) == 3:
    grid_size = {"x1": 0.1, "x2": 0.1, "x3": 0.05}
    if global_dir_name == "sys_3d_too_coarse":
        grid_size = {"x1": 0.5, "x2": 0.5, "x3": 0.1}
    if global_dir_name == "sys_3d_less_fine":
        grid_size = {"x1": 0.25, "x2": 0.25, "x3": 0.1}
    if global_dir_name == "sys_3d":
        grid_size = {"x1": 0.25, "x2": 0.25, "x3": 0.05}

    large_grid = {"x1": 5, "x2": 1, "x3": 1}
elif len(X) == 5:
    grid_size = {"x1": 0.5, "x2": 0.5, "x3": 0.2, "x4": 0.2, "x5": 0.2}
    large_grid = {"x1": 2, "x2": 2, "x3": 0.4, "x4": 0.4, "x5": 0.4}
else:
    exit()

# Neural Network dimensions and training info
d = len(X)  # dimensionality
if use_fixed_nn:
    out_dim = d
else:
    out_dim = min(3, d)

network_dims = [d, width_1, width_2, out_dim]

extents, boundaries = discretize_space(X, grid_size)
states = [i for i in range(len(extents) - 1)]
num_extents = len(extents)-1
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

run_tests = False
get_probs = True

keys = list(X)
if reuse_regions and os.path.exists(global_exp_dir + "/transition_probs.pkl"):
    print('Skipping training')
else:
    file_name = global_exp_dir + "/deep_gp_data"
    if False and reuse_regions and os.path.exists(file_name + '_0_0.pth'):
        # TODO, figure out why this doesn't save/load properly
        print("Loading previous GP data...")
        unknown_dyn_gp = dkl_load(file_name, unknown_dyn_gp, all_data, use_reLU, num_layers, network_dims)

    tic = time.perf_counter()
    for mode in modes:
        if unknown_dyn_gp[mode] is None:
            if use_fixed_nn:
                if use_local_gp:
                    unknown_dyn_gp[mode], region_info[mode] = deep_kernel_fixed_nn_local(all_data, mode, keys, use_reLU,
                                                                                         num_layers, network_dims, crown_dir
                                                                                         , global_dir_name, large_grid, X,
                                                                                         process_noise=process_dist,
                                                                                         lr=learning_rate,
                                                                                         training_iter=epochs,
                                                                                         random_seed=(mode+2)*random_Seed,
                                                                                         epochs=nn_epochs, nn_lr=nn_lr)
                else:
                    unknown_dyn_gp[mode] = deep_kernel_with_fixed_nn(all_data, mode, keys, use_reLU, num_layers,
                                                                     network_dims, crown_dir, global_dir_name,
                                                                     process_noise=process_dist, lr=learning_rate,
                                                                     training_iter=epochs, random_seed=(mode+2)*random_Seed,
                                                                     epochs=nn_epochs, nn_lr=nn_lr)
            else:
                unknown_dyn_gp[mode] = deep_kernel_learn(all_data, mode, keys, use_reLU, num_layers, network_dims,
                                                         crown_dir, global_dir_name, process_noise=process_dist,
                                                         lr=learning_rate, training_iter=epochs,
                                                         random_seed=(mode+2)*random_Seed)
            print(f'Finished deep kernel regression for mode {mode}\n')
            dkl_save(file_name, unknown_dyn_gp, mode, use_local_gp)
    toc = time.perf_counter()
    print(f"It took {toc - tic} seconds to train all the Deep Kernel GP models.")

    # if len(X) == 1:
    #     deep_kernel_plot(unknown_dyn_gp[0], all_data, unknown_modes_list, 0, X, global_exp_dir)
    # else:
    #     if use_local_gp:
    #         gp_plot = unknown_dyn_gp[0][1]
    #         plot_domain = region_info[0][1][2]
    #         deep_kernel_plot_nD(gp_plot, region_info[0][1], 0, 0, plot_domain, global_exp_dir)
    #         # exit()

# =====================================================================================
# 3. Get posteriors of GP
# =====================================================================================

if get_probs:
    file_name = global_exp_dir + "/region_data.pkl"
    region_data = None
    if reuse_regions and os.path.exists(file_name):
        print("Loading previous region data...")
        region_data = dict_load(file_name)

    if (region_data is None) or any(region_data[mode][0] is None for mode in modes):
        print("Calculating post-regions...")

        if region_data is None:
            # a list where each input has the data for a different mode
            # data for each mode organized as [mean bounds, sig bounds, NN linear transform]
            # mean bounds is a tuple of tuples ((dim_upper, dim_lower) for each dim), same for sig
            # NN linear transform is a tuple of tuples ((lA, uA, l_bias, u_bias, bounds) for each dim)
            region_data = [[None, None, None] for mode in modes]

        tic = time.perf_counter()

        for mode in modes:
            sub_tic = time.perf_counter()
            if region_data[mode][0] is None:
                if use_fixed_nn and use_local_gp:
                    result = local_dkl_posts_fixed_nn(unknown_dyn_gp[mode], mode, extents, region_info[mode],
                                                      out_dim, crown_dir, EXPERIMENT_DIR, threads=threads)
                else:
                    x_data = torch.tensor(np.transpose(all_data[mode][0]), dtype=torch.float32)
                    y_data = np.transpose(all_data[mode][1])
                    if use_fixed_nn:
                        result = dkl_posts_fixed_nn(unknown_dyn_gp[mode], x_data, y_data, mode, extents,
                                                    out_dim, crown_dir, EXPERIMENT_DIR, threads=threads)
                    else:
                        result = dkl_posts(unknown_dyn_gp[mode], x_data, y_data, mode, extents,
                                           out_dim, crown_dir, EXPERIMENT_DIR, threads=threads)
                region_data[mode] = result

                dict_save(file_name, region_data)

            sub_toc = time.perf_counter()
            print(f"Finished post region calculation for mode {mode} in {sub_toc - sub_tic:0.4f} seconds.")

        toc = time.perf_counter()
        print(f"Region generation took {toc - tic:0.4f} seconds.")
        dict_save(file_name, region_data)

    if not run_tests:
        del unknown_dyn_gp  # no longer need to carry around all the gps
    # =====================================================================================
    # 3. Construct transition bounds from posteriors
    # =====================================================================================

    file_name = global_exp_dir + "/transition_probs.pkl"
    extents = discretize_space_list(X, grid_size)
    # reuse_regions = False
    if reuse_regions and os.path.exists(file_name):
        print("Loading previous transitions data...")
        probs = dict_load(file_name)
        min_probs = probs["min"]
        max_probs = probs["max"]
        del probs
    else:
        print('Calculating transition probabilities...')
        tic = time.perf_counter()
        threads = 12
        if num_extents*len(modes) > 10000:
            # need to do this in sections to save space
            file_name_partial = global_exp_dir + "/transition_probs_partial.pkl"
            min_probs, max_probs = generate_trans_par_dkl_subsections(extents, region_data, modes, threads,
                                                                      file_name_partial)
        else:
            min_probs, max_probs = generate_trans_par_dkl(extents, region_data, modes, threads)
            
        toc = time.perf_counter()
        print(f"It took {toc - tic} seconds to get the transition probabilities")
        probs = {"min": min_probs, "max": max_probs}
        dict_save(file_name, probs)
        del probs

    # =====================================================================================
    # 4. Run synthesis/verification
    # =====================================================================================
    print('Generating IMDP Model')
    unsafe_label = "b"
    label_fn = label_states(labels, extents, unsafe_label)
    imdp = IMDPModel(states, modes, min_probs, max_probs, label_fn, extents)
    del max_probs, min_probs, extents

    if do_synthesis:
        dfa = specification_to_dfa(specification)
        pimdp = construct_pimdp(dfa, imdp)
    else:
        # for now just a bounded until of !bUa
        k = 5  # number of time steps
        imdp_filepath = global_exp_dir + f"/imdp-{k}.txt"
        phi1 = "!b"
        phi2 = "a"
        # phi1 U<k phi2
        res = bounded_until(imdp, phi1, phi2, k, imdp_filepath, synthesis_flag=True)

        # now plot the results
        print('plotting results')
        plot_verification_results(res, imdp, global_exp_dir, k, region_labels, min_threshold=.9)

print("Finished")
