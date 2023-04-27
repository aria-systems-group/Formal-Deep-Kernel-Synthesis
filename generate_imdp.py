from generic_fnc import *
from dynamics_script import *
from gp_scripts import *
from space_discretize_scripts import *
from dfa_scripts import *

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR + experiment_type
crown_dir = EXPERIMENT_DIR + "/alpha-beta-CROWN/complete_verifier"

reuse_regions = False
threads = int(sys.argv[1])
experiment_number = int(sys.argv[2])
refinement = int(sys.argv[3])
continue_refine = int(sys.argv[4])

if experiment_number == 2:
    global_dir_name = "sys_2d"
    dfa = dict_load(EXPERIMENT_DIR + "/dfa_reach_avoid_complex")
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
elif experiment_number == 1:
    global_dir_name = "sys_3d_"
    dfa = dict_load(EXPERIMENT_DIR + "/dfa_reach_avoid")
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    process_dist = {"mu": [0., 0., 0.], "sig": [0.01, 0.01, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]

    unsafe_set = [[[0., 1], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[3., 5.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
elif experiment_number == 3:
    global_dir_name = "sys_3d"
    dfa = dict_load(EXPERIMENT_DIR + "/dfa_reach_avoid")
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    process_dist = {"mu": [0., 0., 0.], "sig": [0.01, 0.01, 0.0001], "dist": "multi_norm"}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]

    unsafe_set = [[[0., 1], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[3., 5.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
elif experiment_number == 4:
    global_dir_name = "sys_3d_gp"
    dfa = dict_load(EXPERIMENT_DIR + "/dfa_reach_avoid")
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    unknown_modes_list = [g_3d_mode1, g_3d_mode2, g_3d_mode3, g_3d_mode4, g_3d_mode5]

    unsafe_set = [[[0., 1], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[3., 5.], [0., 1.], [-0.5, 0.5]]]

    region_labels = {"a": goal_set, "b": unsafe_set}
elif experiment_number == 999:
    global_dir_name = "test_refine"
    dfa = dict_load(EXPERIMENT_DIR + "/dfa_reach_avoid")
    X = {"x1": [0., 5.], "x2": [0., 2.], "x3": [-0.5, 0.5]}
    process_dist = {"mu": [0., 0., 0.], "sig": [0.01, 0.01, 0.0001], "dist": "multi_norm"}
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
dim = len(X)

# =====================================================================================
# 4. Construct transition bounds from posteriors
# =====================================================================================
region_data = [[None, None, None] for mode in modes]

# load in julia results, don't need NN linear bounds at the moment
nn_bounds_dir = global_exp_dir + "/nn_bounds"
for mode in modes:
    mean_bounds = np.load(global_exp_dir+f"/mean_data_{mode+1}_{refinement}.npy")
    sig_bounds = np.load(global_exp_dir+f"/sig_data_{mode+1}_{refinement}.npy")
    lin_transform = np.load(nn_bounds_dir + f"/linear_trans_m_{mode + 1}_{refinement}.npy")
    region_data[mode][0] = mean_bounds
    region_data[mode][1] = sig_bounds
    region_data[mode][2] = lin_transform

filename = global_exp_dir + f"/extents_{refinement}.npy"
extents = np.load(filename)
num_states = len(extents) - 1
states = [i for i in range(num_states)]
print(f"IMDP will have {num_states} states.")

# TODO, allow this to reuse data from previous refinements
file_name = global_exp_dir + f"/transition_probs_{refinement}.pkl"
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
    min_probs, max_probs = generate_trans_par_dkl(extents, region_data, modes, threads, file_name_partial, process_dist)

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
del max_probs, min_probs

k = 4  # number of time steps
imdp_filepath = global_exp_dir + f"/imdp-{k}_{refinement}.txt"

# TODO, create product IMDP, mild debugging
# print('Generating PIMDP Model')
# pimdp = construct_pimdp(dfa, imdp)
# print('Writing PIMDP to file')
# write_pimdp_to_file(pimdp, imdp_filepath)
# print('Running Synthesis')
# res = run_imdp_synthesis(imdp_filepath, k, mode1="maximize")


# for now just a bounded until of !bUa
phi1 = "!b"
phi2 = "a"
# phi1 U<k phi2
res = bounded_until(imdp, phi1, phi2, k, imdp_filepath, synthesis_flag=True)

res_filepath = global_exp_dir + f"/res-{k}_{refinement}.txt"
dict_save(res_filepath, res)

# now plot the results
print('plotting results')
q_refine = plot_verification_results(res, imdp, global_exp_dir, k, refinement, region_labels, min_threshold=.9)

print(len(q_refine))
# =====================================================================================
# 5. Begin refinement setup
# =====================================================================================

if continue_refine == 1:
    print("Identifying states to refine")
    # check states to refine
    n_refine = 500
    refine_states = refine_check(imdp, res, q_refine, n_refine)

    # identify which dimensions need split in these extents, do this by checking which dimension has the largest
    # expansion from the NN linearization
    refinement_algorithm(refine_states, region_data, extents, modes, crown_dir, dim, global_exp_dir, EXPERIMENT_DIR,
                         nn_bounds_dir, refinement, threshold=1e-5, threads=8, use_regular_gp=False)
    print("Running Julia portion to bound new regions")
else:
    print("Finished")
