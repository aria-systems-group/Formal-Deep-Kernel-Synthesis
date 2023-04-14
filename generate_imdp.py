from generic_fnc import *
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
threads = int(sys.argv[1])
experiment_number = int(sys.argv[2])

if experiment_number == 2:
    global_dir_name = "sys_2d"
    X = {"x1": [-2., 2.], "x2": [-2., 2.]}
    unknown_modes_list = [g_2d_mode0, g_2d_mode1, g_2d_mode2, g_2d_mode3]
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0]],
                  [[-1.0, -0.5], [-1.25, -0.875]],
                  [[0.5, 1.125], [-1.75, -1.25]],
                  [[0.75, 1.0], [-0.5, 0.5]],
                  [[0.75, 1.75], [0.75, 1.75]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75]]]
    goal_c = [{"x1": [-1.75, 0.0], "x2": [-2.0, -1.5]},
              {"x1": [1.125, 2.0], "x2": [-1.75, 0.0]},
              {"x1": [-1.25, 0.0], "x2": [1.0, 1.875]}]
    region_labels = {"b": unsafe_set, "a": goal_a, "c": goal_c}
    goal_set = []
    goal_set.extend(goal_a)
    goal_set.extend(goal_c)
elif experiment_number == 3:
    global_dir_name = "sys_3d"
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]

    unsafe_set = [[[0., 1], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[3., 5.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
elif experiment_number == 3.5:
    global_dir_name = "sys_3d_gp"
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]

    unsafe_set = [[[0., 1], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[3., 5.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
else:
    exit()

labels = region_labels
plot_labels = {"obs": unsafe_set, "goal": goal_set}
modes = [i for i in range(len(unknown_modes_list))]
global_exp_dir = exp_dir + "/" + global_dir_name

# abstraction grid size
grid_len = 0.125
grid_size, large_grid = get_grid_info(X, grid_len)

# =====================================================================================
# 4. Construct transition bounds from posteriors
# =====================================================================================
region_data = [[None, None, None] for mode in modes]

# load in julia results, don't need NN linear bounds at the moment
mean_bounds = np.load(global_exp_dir+"/mean_data.npy")
sig_bounds = np.load(global_exp_dir+"/sig_data.npy")

for mode in modes:
    region_data[mode][0] = mean_bounds[mode]
    region_data[mode][1] = sig_bounds[mode]

file_name = global_exp_dir + "/transition_probs.pkl"
extents = discretize_space_list(X, grid_size)
states = [i for i in range(len(extents) - 1)]

if reuse_regions and os.path.exists(file_name):
    print("Loading previous transitions data...")
    probs = dict_load(file_name)
    min_probs = probs["min"]
    max_probs = probs["max"]
else:
    print('Calculating transition probabilities...')
    tic = time.perf_counter()
    # do this in sections to save space just in case
    file_name_partial = global_exp_dir + "/transition_probs_partial.pkl"
    min_probs, max_probs = generate_trans_par_dkl_subsections(extents, region_data, modes, threads,
                                                              file_name_partial)

    toc = time.perf_counter()
    print(f"It took {toc - tic} seconds to get the transition probabilities")
    probs = {"min": min_probs, "max": max_probs}
    dict_save(file_name, probs)

del probs

# =====================================================================================
# 4. Run synthesis/verification
# =====================================================================================

# TODO, create product IMDP


print('Generating IMDP Model')
unsafe_label = "b"
label_fn = label_states(labels, extents, unsafe_label)
imdp = IMDPModel(states, modes, min_probs, max_probs, label_fn, extents)
del max_probs, min_probs, extents

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