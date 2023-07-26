
using Distributed
using JLD

args = ARGS
addprocs(parse(Int, args[1]))

include("imdp_construction.jl")
include("bound_gp_outputs.jl")
include("visualize.jl")
include("dynamics.jl")
include("refinement_algorithm.jl")

@pyimport pickle

EXPERIMENT_DIR = @__DIR__
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR * experiment_type

experiment_number = parse(Int, args[2])
refinements = parse(Int, args[3])
skip_labels = [nothing]
refine_threshold = 1e-5
use_regular_gp = false  # don't change this here

reuse_bounds = true  # feel free to edit any of these
reuse_pimdp = false
reuse_policy = true
reuse_refinement = false
prob_plot = true

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
    skip_labels = ["b∧!a∧!c"]
elseif experiment_number == 2
    global_dir_name = "sys_2d_gp"
    use_regular_gp = true
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
    skip_labels = ["b∧!a∧!c"]
elseif experiment_number == 3
    global_dir_name = "dubins_sys"
    dyn_noise = [0.01, 0.01, 0.01]  # this is std dev of process noise
    dyn_modes = sys_3d_dynamics(dyn_noise)
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    unsafe_set = [[[4., 6], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[8., 10.], [0., 1.], [-0.5, 0.5]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
    prob_plot = false
    skip_labels = ["b∧!a", "a∧!b"]
    refine_threshold = 0.0124 # this is to adjust for numerical errors
elseif experiment_number == 4
    global_dir_name = "dubins_sys_gp"
    dyn_noise = [0.01, 0.01, 0.01]  # this is std dev of process noise
    dyn_modes = sys_3d_dynamics(dyn_noise)
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    unsafe_set = [[[4., 6], [0., 1.], [-0.5, 0.5]]]
    goal_set = [[[8., 10.], [0., 1.], [-0.5, 0.5]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
    prob_plot = false
    skip_labels = ["b∧!a", "a∧!b"]
elseif experiment_number == 5
    global_dir_name = "sys_5d"
    dyn_noise = [0.1, 0.1, 0.01, 0.01, 0.01]  # this is std dev of process noise
    dfa = pickle.load(open(EXPERIMENT_DIR * "/dfa_reach_avoid", "r"))
    dyn_modes = sys_5d(dyn_noise)
    unsafe_set = [[[-0.75, 0.0], [0.5, 2.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]],
                  [[0.5, 2.0], [-0.75, 0.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]]
    goal_set = [[[1.0, 2.0], [0.5, 2.0], [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]]]
    labels = Dict("b" => unsafe_set, "a" => goal_set)
    prob_plot = false
    skip_labels = ["b∧!a", "a∧!b"]
end

unsafe_label = "b"

global_exp_dir = exp_dir * "/" * global_dir_name
general_data = numpy.load(global_exp_dir * "/general_info.npy")
nn_bounds_dir = global_exp_dir * "/nn_bounds"

n_best = 3000  # refine 3000 states?
satisfaction_threshold = .95

num_dfa_states = length(dfa["states"])
if !isnothing(dfa["accept"])
    num_dfa_states -= 1
end
if !isnothing(dfa["sink"])
    num_dfa_states -= 1
end

num_modes = general_data[1]
num_dims = general_data[2]
num_sub_regions = general_data[3]

use_prism = false

for refinement in 0:refinements

    pimdp_filepath = global_exp_dir * "/pimdp_$(refinement).txt"

    extents = numpy.load(global_exp_dir * "/extents_$refinement.npy")
    num_regions = size(extents)[1] - 1
    @info "This abstraction has $num_regions states"

    # define labels for every extent, this can be used to skip bounding obstacle posteriors
    label_fn = label_states(labels, extents, unsafe_label, num_dims, num_regions)

    # Get mean and sig bounds on gp
    @info "Bounding the GP mean and variance"
    reuse_check = reuse_bounds && ((refinement == 0) || reuse_refinement)
    bound_gp(num_regions, num_modes, num_dims, refinement, global_exp_dir, reuse_check, label_fn, skip_labels)

    # setup imdp structure
    modes = [i for i in 1:num_modes]
    states = [i for i in 1:num_regions+1]
    imdp = IMDPModel(states, modes, nothing, nothing, label_fn, extents)

    if use_prism
        prism_filepath = global_exp_dir * "/pimdp_$(refinement)"
        pimdp = prism_pimdp(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                       label_fn, skip_labels, dfa, imdp, prism_filepath)

        prism_res = run_prism(global_exp_dir, prism_filepath, refinement, pimdp)

        @info "Recreating pimdp for satisfaction probabilities T.T"
        pimdp = pimdp_from_prism_res(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                       label_fn, skip_labels, dfa, imdp, pimdp_filepath, prism_res)

        @info "Calculating satisfaction probabilities"
        res = run_synthesis(pimdp_filepath, k, refinement, EXPERIMENT_DIR; ep=1e-6)
    else
        reuse_check = reuse_pimdp && reuse_bounds && ((refinement == 0) || reuse_refinement)
        if reuse_check && isfile(pimdp_filepath)
            @info "Using saved PIMDP"
            # generate a pimdp model for plotting
            pimdp = fast_pimdp(dfa, imdp, num_regions)
            p_action_diff = numpy.load(global_exp_dir*"/p_act_diff_$refinement.npy")
        else
            @info "Constructing and saving the PIMDP Model"
            pimdp, p_action_diff = direct_pimdp_construction(extents, dyn_noise, global_exp_dir, refinement, num_modes,
                                                             num_regions, num_dims, label_fn, skip_labels, dfa, imdp,
                                                             pimdp_filepath)
            numpy.save(global_exp_dir*"/p_act_diff_$refinement", p_action_diff)
        end

        imdp = nothing

        res_filepath = global_exp_dir * "/policy_$(refinement).jld"
        reuse_check = reuse_policy && reuse_pimdp && reuse_bounds && (refinement < 1 || reuse_refinement)
        if reuse_check && isfile(res_filepath)
            @info "Using saved policy"
            res = load(res_filepath, "res")
        else
            @info "Running Synthesis"
            accuracy = 1e-6
            if length(dyn_noise) > 3
                # it is too slow on higher dims, allow the upper bound some slack
                accuracy = 1e-3
            end
            res = run_synthesis(pimdp_filepath, -1, refinement, EXPERIMENT_DIR; ep=accuracy)
            save(res_filepath, "res", res)
        end

    end

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
    elseif length(dyn_noise) == 3
        x0 = [[0.221, 0.213, 0.21]]
    elseif length(dyn_noise) == 5
        x0 = [[0.9801, -1.213, 0.001, 0.001, 0.001], [-1.001, -1.213, 0.001, 0.001, 0.001]]
    end

    q_refine = plot_nd_results(res, extents, num_regions, num_dims, plot_dir, dfa, pimdp, refinement;
                               num_dfa_states=num_dfa_states, min_threshold=satisfaction_threshold,
                               labeled_regions=labels, obs_key=unsafe_label, prob_plots=prob_plot, x0=x0,
                               modes=dyn_modes)

    if refinement < refinements
        if reuse_refinement && isfile(nn_bounds_dir * "/linear_trans_m_$(num_modes)_$(refinement+1).npy")

        else
            @info "Beginning refinement algorithm"
            refine_states = refine_check(res, q_refine, n_best, num_dfa_states, p_action_diff; dfa_init_state=1)

            refinement_algorithm(refine_states, extents, modes, num_dims, global_dir_name, nn_bounds_dir, refinement;
                                threshold=refine_threshold)

            @info "Refined regions created"
            print("\n")
        end
    else
        @info "Done!"
    end
end