
using Distributed
using JLD

args = ARGS
addprocs(parse(Int, args[1]))

include("imdp_construction.jl")
include("bound_gp_outputs.jl")
include("visualize.jl")
include("dynamics.jl")
# include("refinement_algorithm.jl")

@pyimport pickle

EXPERIMENT_DIR = @__DIR__
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR * experiment_type

experiment_number = parse(Int, args[2])
refinements = parse(Int, args[3])

if experiment_number == 0
    global_dir_name = "sys_2d_lin"
    dyn_noise = [0.1, 0.1]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    unsafe_set = [nothing]
    goal_set = [[[-1., 1.], [-1., 1.]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
elseif experiment_number == 1
    global_dir_name = "sys_2d"
    dyn_noise = [0.01, 0.01]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid_complex", "r"))
    dyn_modes = sys_2d_dynamics(dyn_noise[1])
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0]],
                  [[-1.0, -0.5], [-1.25, -0.875]],
                  [[0.5, 1.125], [-1.75, -1.25]],
                  [[0.75, 1.0], [-0.5, 0.5]],
                  [[0.75, 1.75], [0.75, 1.75]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75]]]
    goal_c = [[[-1.75, 0.0], [-2.0, -1.5]],
              [[1.125, 2.0], [-1.75, 0.0]],
              [[-1.25, 0.0], [1.0, 1.875]]]
    labels = Dict("b" => unsafe_set, "a" => goal_a, "c" => goal_c)
elseif experiment_number == 2
    global_dir_name = "sys_2d_gp"
    dyn_noise = [0.01, 0.01]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid_complex", "r"))
    dyn_modes = sys_2d_dynamics(dyn_noise[1])
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0]],
                  [[-1.0, -0.5], [-1.25, -0.875]],
                  [[0.5, 1.125], [-1.75, -1.25]],
                  [[0.75, 1.0], [-0.5, 0.5]],
                  [[0.75, 1.75], [0.75, 1.75]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75]]]
    goal_c = [[[-1.75, 0.0], [-2.0, -1.5]],
              [[1.125, 2.0], [-1.75, 0.0]],
              [[-1.25, 0.0], [1.0, 1.875]]]
    labels = Dict("b" => unsafe_set, "a" => goal_a, "c" => goal_c)
elseif experiment_number == 3
    global_dir_name = "sys_3d"
    dyn_noise = [0.1, 0.1, 0.1]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid_complex", "r"))
    dyn_modes = sys_2d_as_3d_dynamics(dyn_noise[1])
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0], [-2., 2.]],
                  [[-1.0, -0.5], [-1.25, -0.875], [-2., 2.]],
                  [[0.5, 1.125], [-1.75, -1.25], [-2., 2.]],
                  [[0.75, 1.0], [-0.5, 0.5], [-2., 2.]],
                  [[0.75, 1.75], [0.75, 1.75], [-2., 2.]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75], [-2., 2.]]]
    goal_c = [[[-1.75, 0.0], [-2.0, -1.5], [-2., 2.]],
              [[1.125, 2.0], [-1.75, 0.0], [-2., 2.]],
              [[-1.25, 0.0], [1.0, 1.875], [-2., 2.]]]
    labels = Dict("b" => unsafe_set, "a" => goal_a, "c" => goal_c)
elseif experiment_number == 4
    global_dir_name = "sys_3d_gp"
    dyn_noise = [0.1, 0.1, 0.1]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid_complex", "r"))
    dyn_modes = sys_2d_as_3d_dynamics(dyn_noise[1])
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0], [-2., 2.]],
                  [[-1.0, -0.5], [-1.25, -0.875], [-2., 2.]],
                  [[0.5, 1.125], [-1.75, -1.25], [-2., 2.]],
                  [[0.75, 1.0], [-0.5, 0.5], [-2., 2.]],
                  [[0.75, 1.75], [0.75, 1.75], [-2., 2.]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75], [-2., 2.]]]
    goal_c = [[[-1.75, 0.0], [-2.0, -1.5], [-2., 2.]],
              [[1.125, 2.0], [-1.75, 0.0], [-2., 2.]],
              [[-1.25, 0.0], [1.0, 1.875], [-2., 2.]]]
    labels = Dict("b" => unsafe_set, "a" => goal_a, "c" => goal_c)
elseif experiment_number == 5
    global_dir_name = "sys_5d"
    dyn_noise = [0.1, 0.1, 0.01, 0.01, 0.01]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    dyn_modes = sys_5d(dyn_noise)
    unsafe_set = [[[-0.75, 0.0], [0.5, 2.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]],
                  [[0.5, 2.0], [-0.75, 0.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]]
    goal_set = [[[1.0, 2.0], [0.5, 2.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
elseif experiment_number == 30
    global_dir_name = "dubins_sys"
    dyn_noise = [0.01, 0.01, 0.01]  # this is std dev of process noise
    dyn_modes = sys_3d_dynamics(dyn_noise)
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    unsafe_set = [[[4., 6], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[0., 10.], [0., 1.], [-0.5, 0.5]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
elseif experiment_number == 40
    global_dir_name = "dubins_sys_gp"
    dyn_noise = [0.01, 0.01, 0.01]  # this is std dev of process noise
    dyn_modes = sys_3d_dynamics(dyn_noise)
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    unsafe_set = [[[4., 6], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[8., 10.], [0., 1.], [-0.5, 0.5]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
elseif experiment_number == 20
    global_dir_name = "sys_2d_fine"
    dyn_noise = [0.01, 0.01]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid_complex", "r"))
    dyn_modes = sys_2d_dynamics(dyn_noise[1])
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0]],
                  [[-1.0, -0.5], [-1.25, -0.875]],
                  [[0.5, 1.125], [-1.75, -1.25]],
                  [[0.75, 1.0], [-0.5, 0.5]],
                  [[0.75, 1.75], [0.75, 1.75]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75]]]
    goal_c = [[[-1.75, 0.0], [-2.0, -1.5]],
              [[1.125, 2.0], [-1.75, 0.0]],
              [[-1.25, 0.0], [1.0, 1.875]]]
    labels = Dict("b" => unsafe_set, "a" => goal_a, "c" => goal_c)
elseif experiment_number == 21
    global_dir_name = "sys_2d_gp_fine"
    dyn_noise = [0.01, 0.01] # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid_complex", "r"))
    dyn_modes = sys_2d_dynamics(dyn_noise[1])
    unsafe_set = [[[-1.75, -1.25], [-0.75, 1.0]],
                  [[-1.0, -0.5], [-1.25, -0.875]],
                  [[0.5, 1.125], [-1.75, -1.25]],
                  [[0.75, 1.0], [-0.5, 0.5]],
                  [[0.75, 1.75], [0.75, 1.75]]]
    goal_a = [[[-0.75, 0.75], [-0.75, 0.75]]]
    goal_c = [[[-1.75, 0.0], [-2.0, -1.5]],
              [[1.125, 2.0], [-1.75, 0.0]],
              [[-1.25, 0.0], [1.0, 1.875]]]
    labels = Dict("b" => unsafe_set, "a" => goal_a, "c" => goal_c)
elseif experiment_number == 60
    global_dir_name = "dubins_sys_expanded"
    dyn_noise = [0.01, 0.01, 0.01]  # this is std dev of process noise
    dyn_modes = sys_3d_dynamics(dyn_noise)
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    unsafe_set = [nothing]
    goal_set = [[[8., 10.], [0., 1.], [-.5, .5]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
end

global_exp_dir = exp_dir * "/" * global_dir_name
general_data = numpy.load(global_exp_dir * "/general_info.npy")
nn_bounds_dir = global_exp_dir * "/nn_bounds"

reuse_regions = true
prob_plot = true

num_modes = general_data[1]
num_dims = general_data[2]
num_sub_regions = general_data[3]

for refinement in 0:refinements
    extents = numpy.load(global_exp_dir * "/extents_$refinement.npy")
    num_regions = size(extents)[1] - 1

    # Get mean and sig bounds on gp, this is inefficient currently
    @info "Bounding the GP mean and variance"
    bound_gp(num_regions, num_modes, num_dims, refinement, global_exp_dir, reuse_regions)

    # get transition probabilities
    if reuse_regions && isfile(global_exp_dir*"/transition_probs_max_$refinement.jld")
        @info "Loading prior transition probabilities"
        minPrs = load(global_exp_dir*"/transition_probs_min_$refinement.jld", "minPrs")
        maxPrs = load(global_exp_dir*"/transition_probs_max_$refinement.jld", "maxPrs")
    else
        @info "Constructing transition probabilities"
        trans_probs_runtime = @elapsed begin
            minPrs, maxPrs = imdp_probs(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims)
        end
        @info "Finished getting transition probabilities in $trans_probs_runtime seconds"
        save(global_exp_dir*"/transition_probs_min_$refinement.jld", "minPrs", minPrs)
        save(global_exp_dir*"/transition_probs_max_$refinement.jld", "maxPrs", maxPrs)

    end

    # construct an IMDP from the transition probabilities
    unsafe_label = "b"
    label_fn = label_states(labels, extents, unsafe_label, num_dims, num_regions)

    modes = [i for i in 1:num_modes]
    states = [i for i in 1:num_regions+1]
    imdp = IMDPModel(states, modes, minPrs, maxPrs, label_fn, extents)

    # build Product IMDP from dfa and imdp then run synthesis on that
    k = -1
    pimdp_filepath = global_exp_dir * "/pimdp_$(refinement).txt"

    if false && isfile(pimdp_filepath)
        @info "Using saved PIMDP"
        # generate a pimdp model for plotting
        pimdp = fast_pimdp(dfa, imdp)
    else
        @info "Constructing PIMDP Model"
        pimdp = construct_DFA_IMDP_product(dfa, imdp)
        @info "Writing PIMDP to file"
        write_file_time = @elapsed begin
            write_pimdp_to_file(pimdp, pimdp_filepath)
        end
        @info "Took $write_file_time seconds to write the file"
    end

    @info "Running Synthesis"
    accuracy = 1e-3
    if prob_plot
        accuracy = 1e-6
    end
    res = run_synthesis(pimdp_filepath, k, refinement; ep=accuracy)

    # plot results
    plot_dir = global_exp_dir * "/plots"
    if !isdir(plot_dir)
        mkdir(plot_dir)
    end

    @info "Plotting Results"
    obs_key = "b"
    x0 = nothing
    if length(dyn_noise) == 2
        x0 = [[-1.8, -1.7], [1.8, -1.82], [1.74, 0.53], [-1.74, 1.81]]
        x0 = [[1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7], [1.8, -1.7]]
    elseif length(dyn_noise) == 3
        x0 = [[0.221, 0.213, 0.21]]
        x0 = [[5.221, 1.813, 0.01]]
    end

    if length(dyn_noise) > 3
        global prob_plot = false
    end
    thresh = .95
    plot_nd_results(res, extents, num_regions, num_dims, plot_dir, dfa, pimdp, refinement; num_dfa_states=length(dfa["states"]),
                    min_threshold=thresh, labeled_regions=labels, obs_key=obs_key, prob_plots=prob_plot, x0=x0,
                    modes=dyn_modes)

end