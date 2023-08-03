
using SpecialFunctions
using SparseArrays
using Base.Threads
using Printf
using ProgressBars
using PyCall
@pyimport numpy


struct IMDPModel
    states
    actions
    Pmin
    Pmax
    labels
    extents
end


struct PIMDPModel
    states
    actions
    Pmin
    Pmax
    labels
    accepting_labels
    sink_labels
    extents
end


#======================================================================================================================#
#  Transition probability calculation
#======================================================================================================================#


function matching_label(test_label, compare_label)
    if isnothing(compare_label)
        return false
    end
    if test_label == "true"
        # any observation satisfies true
        return true
    end
    separated_test = Set(split(test_label, '∧'))
    separated_compare = Set(split(compare_label, '∧'))
    if issubset(separated_test, separated_compare)
        return true
    end
    if issubset(separated_compare, separated_test)
        return true
    end
    return false
end


function prob_via_erf(lb, la, mean, sigma)
    # Pr(la <= X <= lb) when X ~ N(mean, sigma)
    return 0.5 * (erf((lb - mean) / (sqrt(2) * sigma)) - erf((la - mean) / (sqrt(2) * sigma)))
end


function direct_pimdp_construction(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                   label_fn, skip_labels, dfa, imdp, filename)
    # this one is going to avoid making a sparse array and directly construct the pimdp file

    # define basic pimdp aspects from states and dfa
    dfa_states = sort(dfa["states"])
    states_sans_ends = copy(dfa_states)
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)
    addon = 0
    if !isnothing(dfa_acc_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_acc_state, states_sans_ends)
    end

    if !isnothing(dfa_sink_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_sink_state, states_sans_ends)
    end

    dfa_num_map = Dict()
    for i in 1:length(states_sans_ends)
        dfa_num_map[states_sans_ends[i]] = i
    end

    # Transition matrix size will be Nx|Q| -- or the original transition matrix permuted with the number of states in DFA
    N = (num_regions + 1) * num_modes
    M = num_regions + 1

    pimdp_states = []

    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    for s in imdp.states
        for q in states_sans_ends
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end

    acc_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_acc_state)
        acc_labels[M*sizeQ + 1] = 1
        push!(pimdp_states, (-1, dfa_acc_state))
    end

    sink_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_sink_state)
        sink_labels[end] = 1
        push!(pimdp_states, (-1, dfa_sink_state))
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, nothing, nothing, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)

    # here is where we get transition probabilities from one state to another
    pbar = ProgressBar(total=(num_regions+1)*sizeQ*sizeA)
    p_min_vec = zeros(sizeA, num_regions+1)
    p_max_vec = zeros(sizeA, num_regions+1)

    # pre-load all bounds and store in new array?
    all_means = Dict()
    all_sigs = Dict()
    gp_bounds_dir = global_exp_dir  * "/gp_bounds"
    for mode in 1:(num_modes::Int)
        mean_bounds = numpy.load(gp_bounds_dir*"/mean_data_$mode" * "_$refinement.npy")
        sig_bounds = numpy.load(gp_bounds_dir*"/sig_data_$mode" * "_$refinement.npy")
        all_means[mode] = mean_bounds
        all_sigs[mode] = sig_bounds
    end

    p_action_diff = []
    p_in_diff = zeros(num_regions+1)

    # this is actually super wasteful since it calculates the transitions for each dfa state ...
    open(filename, "w") do f
        state_num = length(pimdp.states)
        action_num = length(pimdp.actions)

        @printf(f, "%d \n", state_num)
        @printf(f, "%d \n", length(pimdp.actions))

        # Get number of accepting states from the labels vector
        pimdp_acc_labels = pimdp.accepting_labels
        acc_states = findall(>(0.), pimdp_acc_labels[:])

        @printf(f, "%d \n", sum(pimdp_acc_labels))
        [@printf(f, "%d ", acc_state-1) for acc_state in acc_states]
        @printf(f, "\n")

        avg_transitions = 0

        for i in 1:(num_regions::Int)
            p_min_vec .= 0.0
            p_max_vec .= 0.0
            for q in states_sans_ends
                qp_test = delta_(q, dfa, imdp.labels[i])  # which dfa state it transitions to
                i_pimdp = (i-1)*sizeQ + dfa_num_map[q]   # this is the pimdp state number
                p_action = 0
                num_trans = Atomic{Float64}(0)
                for mode in 1:(num_modes::Int)

                    mean_bounds = all_means[mode]
                    sig_bounds = all_sigs[mode]
                    sum_up = Atomic{Float64}(0)
                    if sum(p_max_vec[mode, :]) < 1
                        # calculate transitions
                        if any([matching_label(label_fn[i], skip_) for skip_ in skip_labels])
                            # don't actually care about the transitions on this region either, just put 1's to itself
                            p_min_vec[mode, i] = 1.0
                            p_max_vec[mode, i] = 1.0
                        else
                            # find transition probabilities to
                            Threads.@threads for j in 1:(num_regions+1::Int)
                                post = extents[j, :, :]
                                p_min = 1.0
                                p_max = 1.0
                                for dim in 1:(num_dims::Int)
                                    if p_max == 0.0
                                        continue
                                    end

                                    # bounds on the mean of the image of the pre-region
                                    lower_mean = mean_bounds[i, dim, 1]
                                    upper_mean = mean_bounds[i, dim, 2]

                                    lower_sigma = sig_bounds[i, dim, 1] + dyn_noise[dim]
                                    upper_sigma = sig_bounds[i, dim, 2] + dyn_noise[dim]  # this is a std deviation

                                    post_bounds = post[dim, :]
                                    post_low = post_bounds[1]
                                    post_up = post_bounds[2]
                                    post_mean = post_low + (post_up - post_low) / 2.0

                                    if (lower_mean < post_mean) && (upper_mean > post_mean)
                                        # image contains the center of the post region
                                        middle_mean = post_mean
                                    elseif upper_mean < post_mean
                                        # upper bound is closer to the center of the post
                                        middle_mean = upper_mean
                                    else
                                        # lower bound is closer to the center of the post
                                        middle_mean = lower_mean
                                    end

                                    p_max *= max(prob_via_erf(post_up, post_low, middle_mean, upper_sigma),
                                                 prob_via_erf(post_up, post_low, middle_mean, lower_sigma))

                                    # min prob will be one of the four combos of largest/smallest mean/sigma
                                    p_min *= minimum([prob_via_erf(post_up, post_low, lower_mean, lower_sigma),
                                                      prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                                                      prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                                                      prob_via_erf(post_up, post_low, upper_mean, lower_sigma)])

                                end
                                p_min = round(p_min; digits=5)
                                p_max = round(p_max; digits=5)

                                if j == num_regions + 1
                                    p_min_ = p_min
                                    p_min = 1.0 - p_max
                                    p_max = 1.0 - p_min_
                                end
                                atomic_add!(sum_up, p_max)

                                p_min_vec[mode, j] = p_min
                                p_max_vec[mode, j] = p_max
                                if p_max > 0
                                    atomic_add!(num_trans, 1.0)
                                end
                            end
                            if sum_up[] < 1
                                if sum_up[] > .999
                                    # this is a numerical error, adjust the upper bounds to be equal to 1
                                    p_max_vec[mode, :] ./ sum_up[]
                                else
                                    @info "upper bound is bad, $(sum_up[]), $i, $mode"
                                    exit()
                                end
                            end
                        end
                    end
                    p_action += sum(p_max_vec[mode, :]) - sum(p_min_vec[mode, :])

                    p_in_diff += (p_max_vec[mode, :] - p_min_vec[mode, :])

                    # here I have the transition probabilities from state "i" under action "mode" and dfa state "q"
                    if qp_test == dfa_acc_state
                        @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ), sum(p_min_vec[mode, :]), sum(p_max_vec[mode, :]))
                    elseif qp_test == dfa_sink_state
                        @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ)+addon-1, sum(p_min_vec[mode, :]), sum(p_max_vec[mode, :]))
                    else
                        ij = findall(>(0.), p_max_vec[mode, :])
                        qp_use = dfa_num_map[qp_test]
                        for s in ij
                            col_idx = (s-1)*sizeQ + qp_use
                            @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, col_idx-1, p_min_vec[mode, s], p_max_vec[mode, s])
                        end
                    end
                    update(pbar)
                end
                avg_transitions += num_trans[]
                push!(p_action_diff, p_action)
            end
        end

        # self transitions outside of the defined space
        # do the same file writing for this state
        for q in states_sans_ends
            qp_test = delta_(q, dfa, imdp.labels[num_regions+1])  # which dfa state it transitions to
            i_pimdp = (num_regions)*sizeQ + dfa_num_map[q]   # this is the pimdp state number
            for mode in 1:(num_modes::Int)
                if qp_test == dfa_acc_state
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ), 1.0, 1.0)
                elseif qp_test == dfa_sink_state
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ)+addon-1, 1.0, 1.0)
                else
                    qp_use = dfa_num_map[qp_test]
                    col_idx = (num_regions)*sizeQ + qp_use
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, col_idx-1, 1.0, 1.0)
                end
                update(pbar)
            end
        end

        if !isnothing(dfa_acc_state)
            for mode in 1:(num_modes::Int)
                @printf(f, "%d %d %d %f %f", (M*sizeQ), mode-1, (M*sizeQ), 1.0, 1.0)
                if !isnothing(dfa_sink_state) || mode < num_modes
                    @printf(f, "\n")
                end
            end
        end

        if !isnothing(dfa_sink_state)
            for mode in 1:(num_modes::Int)
                @printf(f, "%d %d %d %f %f", (M*sizeQ)+addon-1, mode-1, (M*sizeQ)+addon-1, 1.0, 1.0)
                if mode < num_modes
                    @printf(f, "\n")
                end
            end
        end

        @info "This model has an average of $(avg_transitions/num_regions) transitions from each state"
    end

    # which one? p_action_diff or p_in_diff
    return pimdp, p_action_diff, p_in_diff
end


function prism_pimdp(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                   label_fn, skip_labels, dfa, imdp, filename)
    # this one is going to avoid making a sparse array and directly construct the pimdp file

    # define basic pimdp aspects from states and dfa
    dfa_states = sort(dfa["states"])
    states_sans_ends = dfa_states
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)
    addon = 0
    if !isnothing(dfa_acc_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_acc_state, states_sans_ends)
    end

    if !isnothing(dfa_sink_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_sink_state, states_sans_ends)
    end

    dfa_num_map = Dict()
    for i in 1:length(states_sans_ends)
        dfa_num_map[states_sans_ends[i]] = i
    end

    # Transition matrix size will be Nx|Q| -- or the original transition matrix permuted with the number of states in DFA
    N = (num_regions + 1) * num_modes
    M = num_regions + 1

    pimdp_states = []

    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    for s in imdp.states
        for q in states_sans_ends
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end


    acc_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_acc_state)
        acc_labels[M*sizeQ + 1] = 1
        push!(pimdp_states, (-1, dfa_acc_state))
    end

    sink_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_sink_state)
        sink_labels[end] = 1
        push!(pimdp_states, (-1, dfa_sink_state))
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, nothing, nothing, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)

    # here is where we get transition probabilities from one state to another
    pbar = ProgressBar(total=(num_regions+1)*sizeQ*sizeA)
    p_min_vec = zeros(sizeA, num_regions+1)
    p_max_vec = zeros(sizeA, num_regions+1)

    # pre-load all bounds and store in new array?
    all_means = Dict()
    all_sigs = Dict()
    for mode in 1:(num_modes::Int)
        mean_bounds = numpy.load(global_exp_dir*"/mean_data_$mode" * "_$refinement.npy")
        sig_bounds = numpy.load(global_exp_dir*"/sig_data_$mode" * "_$refinement.npy")
        all_means[mode] = mean_bounds
        all_sigs[mode] = sig_bounds
    end


    # will need to create a props file with the accepting state as "goal"
    props_file = filename * ".props"
    open(props_file, "w") do f
        pimdp_acc_labels = pimdp.accepting_labels
        acc_state = findall(>(0.), pimdp_acc_labels[:])[1]  # just one state by design
        @printf(f, "label \"goal\" = (s=%d);\n", acc_state-1)
        @printf(f, "Pmaxmin=? [ F \"goal\" ]\n")
    end

    # now write the prism module for the pimdp
    pm_file = filename * ".pm"
    open(pm_file, "w") do f

        @printf(f, "mdp\n\n")
        @printf(f, "module pimdp\n\n")

        state_num = length(pimdp.states)
        @printf(f, "\ts : [0..%d]; \n\n", state_num-1)

        for i in 1:(num_regions::Int)
            p_min_vec .= 0.0
            p_max_vec .= 0.0
            for q in states_sans_ends
                qp_test = delta_(q, dfa, imdp.labels[i])  # which dfa state it transitions to
                i_pimdp = (i-1)*sizeQ + dfa_num_map[q]   # this is the pimdp state number
                for mode in 1:(num_modes::Int)

                    mean_bounds = all_means[mode]
                    sig_bounds = all_sigs[mode]
                    sum_up = Atomic{Float64}(0)
                    if sum(p_max_vec[mode, :]) < 1
                        # calculate transitions
                        if any([matching_label(label_fn[i], skip_) for skip_ in skip_labels])
                            # don't actually care about the transitions on this region either, just put 1's to itself
                            p_min_vec[mode, i] = 1.0
                            p_max_vec[mode, i] = 1.0
                        else
                            # find transition probabilities to
                            Threads.@threads for j in 1:(num_regions+1::Int)
                                post = extents[j, :, :]
                                p_min = 1.0
                                p_max = 1.0
                                for dim in 1:(num_dims::Int)
                                    if p_max == 0.0
                                        continue
                                    end

                                    # bounds on the mean of the image of the pre-region
                                    lower_mean = mean_bounds[i, dim, 1]
                                    upper_mean = mean_bounds[i, dim, 2]

                                    lower_sigma = sig_bounds[i, dim, 1] + dyn_noise[dim]
                                    upper_sigma = sig_bounds[i, dim, 2] + dyn_noise[dim]  # this is a std deviation

                                    post_bounds = post[dim, :]
                                    post_low = post_bounds[1]
                                    post_up = post_bounds[2]
                                    post_mean = post_low + (post_up - post_low) / 2.0

                                    if (lower_mean < post_mean) && (upper_mean > post_mean)
                                        # image contains the center of the post region
                                        middle_mean = post_mean
                                    elseif upper_mean < post_mean
                                        # upper bound is closer to the center of the post
                                        middle_mean = upper_mean
                                    else
                                        # lower bound is closer to the center of the post
                                        middle_mean = lower_mean
                                    end

                                    p_max *= max(prob_via_erf(post_up, post_low, middle_mean, upper_sigma),
                                                 prob_via_erf(post_up, post_low, middle_mean, lower_sigma))

                                    # min prob will be one of the four combos of largest/smallest mean/sigma
                                    p_min *= minimum([prob_via_erf(post_up, post_low, lower_mean, lower_sigma),
                                                      prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                                                      prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                                                      prob_via_erf(post_up, post_low, upper_mean, lower_sigma)])

                                end
                                p_min = round(p_min; digits=5)
                                p_max = round(p_max; digits=5)

                                if j == num_regions + 1
                                    p_min_ = p_min
                                    p_min = 1.0 - p_max
                                    p_max = 1.0 - p_min_
                                end
                                atomic_add!(sum_up, p_max)

                                p_min_vec[mode, j] = p_min
                                p_max_vec[mode, j] = p_max
                            end
                            if sum_up[] < 1
                                if sum_up[] > .999
                                    # this is a numerical error, adjust the upper bounds to be equal to 1
                                    p_max_vec[mode, :] ./ sum_up[]
                                else
                                    @info "upper bound is bad, $(sum_up[]), $i, $mode"
                                    exit()
                                end
                            end
                        end
                    end

                    # here I have the transition probabilities from state "i" under action "mode" and dfa state "q"
                    if qp_test == dfa_acc_state
                        @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);\n", mode-1, i_pimdp-1, 1.0, 1.0, (M*sizeQ))
#                         min_val = max(sum(p_min_vec[mode, :]), 1e-9)
#                         if min_val == 1e-9
#                             @printf(f, "\t[a%d] s=%d -> [%.9f, %.5f]:(s'=%d);\n", mode-1, i_pimdp-1, min_val, sum(p_max_vec[mode, :]), (M*sizeQ))
#                         else
#                             @printf(f, "\t[a%d] s=%d -> [%.5f, %.5f]:(s'=%d);\n", mode-1, i_pimdp-1, min_val, sum(p_max_vec[mode, :]), (M*sizeQ))
#                         end
                    elseif qp_test == dfa_sink_state
                        @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);\n", mode-1, i_pimdp-1, 1.0, 1.0, (M*sizeQ)+addon-1)
#                         min_val = max(sum(p_min_vec[mode, :]), 1e-9)
#                         if min_val == 1e-9
#                             @printf(f, "\t[a%d] s=%d -> [%.9f, %.5f]:(s'=%d);\n", mode-1, i_pimdp-1, min_val, sum(p_max_vec[mode, :]), (M*sizeQ)+addon-1)
#                         else
#                             @printf(f, "\t[a%d] s=%d -> [%.5f, %.5f]:(s'=%d);\n", mode-1, i_pimdp-1, min_val, sum(p_max_vec[mode, :]), (M*sizeQ)+addon-1)
#                         end
                    else
                        ij = findall(>(0.), p_max_vec[mode, :])
                        @printf(f, "\t[a%d] s=%d -> ", mode-1, i_pimdp-1)
                        qp_use = dfa_num_map[qp_test]
                        for s in ij
                            col_idx = (s-1)*sizeQ + qp_use
                            min_val = max(p_min_vec[mode, s], 1e-9)
                            if min_val == 1e-9
                                @printf(f, "[%.9f, %.5f]:(s'=%d)", min_val, p_max_vec[mode, s], col_idx-1)
                            else
                                @printf(f, "[%.5f, %.5f]:(s'=%d)", min_val, p_max_vec[mode, s], col_idx-1)
                            end
                            if s < ij[end]
                                @printf(f, " + ")
                            else
                                @printf(f, ";\n")
                            end
                        end
                    end
                    update(pbar)
                end
            end
        end

        # self transitions outside of the defined space
        # do the same file writing for this state
        for q in states_sans_ends
            qp_test = delta_(q, dfa, imdp.labels[num_regions+1])  # which dfa state it transitions to
            i_pimdp = (num_regions)*sizeQ + dfa_num_map[q]   # this is the pimdp state number
            for mode in 1:(num_modes::Int)
                if qp_test == dfa_acc_state
                    @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);\n", mode-1, i_pimdp-1, 1.0, 1.0, (M*sizeQ))
                elseif qp_test == dfa_sink_state
                    @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);\n", mode-1, i_pimdp-1, 1.0, 1.0, (M*sizeQ)+addon-1)
                else
                    qp_use = dfa_num_map[qp_test]
                    col_idx = (num_regions)*sizeQ + qp_use
                    @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);\n", mode-1, i_pimdp-1, 1.0, 1.0, col_idx-1)
                end
                update(pbar)
            end
        end

        if !isnothing(dfa_acc_state)
            for mode in 1:(num_modes::Int)
                @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);", mode-1, (M*sizeQ), 1.0, 1.0, (M*sizeQ))
                if !isnothing(dfa_sink_state) || mode < num_modes
                    @printf(f, "\n")
                end
            end
        end

        if !isnothing(dfa_sink_state)
            for mode in 1:(num_modes::Int)
                @printf(f, "\t[a%d] s=%d -> [%.1f, %.1f]:(s'=%d);", mode-1, (M*sizeQ)+addon-1, 1.0, 1.0, (M*sizeQ)+addon-1)
                if mode < num_modes
                    @printf(f, "\n")
                end
            end
        end

        @printf(f, "\nendmodule\n")

    end

    return pimdp
end


function pimdp_from_prism_res(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims,
                                   label_fn, skip_labels, dfa, imdp, filename, res)
    # this one is going to avoid making a sparse array and directly construct the pimdp file

    # define basic pimdp aspects from states and dfa
    dfa_states = sort(dfa["states"])
    states_sans_ends = dfa_states
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)
    addon = 0
    if !isnothing(dfa_acc_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_acc_state, states_sans_ends)
    end

    if !isnothing(dfa_sink_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_sink_state, states_sans_ends)
    end

    dfa_num_map = Dict()
    for i in 1:length(states_sans_ends)
        dfa_num_map[states_sans_ends[i]] = i
    end

    # Transition matrix size will be Nx|Q| -- or the original transition matrix permuted with the number of states in DFA
    N = (num_regions + 1) * num_modes
    M = num_regions + 1

    pimdp_states = []

    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    for s in imdp.states
        for q in states_sans_ends
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end


    acc_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_acc_state)
        acc_labels[M*sizeQ + 1] = 1
        push!(pimdp_states, (-1, dfa_acc_state))
    end

    sink_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_sink_state)
        sink_labels[end] = 1
        push!(pimdp_states, (-1, dfa_sink_state))
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, nothing, nothing, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)

    # here is where we get transition probabilities from one state to another
    pbar = ProgressBar(total=(num_regions+1)*sizeQ)
    p_min_vec = zeros(num_regions+1)
    p_max_vec = zeros(num_regions+1)

    # pre-load all bounds and store in new array?
    all_means = Dict()
    all_sigs = Dict()
    for mode in 1:(num_modes::Int)
        mean_bounds = numpy.load(global_exp_dir*"/mean_data_$mode" * "_$refinement.npy")
        sig_bounds = numpy.load(global_exp_dir*"/sig_data_$mode" * "_$refinement.npy")
        all_means[mode] = mean_bounds
        all_sigs[mode] = sig_bounds
    end

    # this is actually super wasteful since it calculates the transitions for each dfa state ...
    open(filename, "w") do f
        state_num = length(pimdp.states)
        action_num = length(pimdp.actions)

        @printf(f, "%d \n", state_num)
        @printf(f, "%d \n", length(pimdp.actions))

        # Get number of accepting states from the labels vector
        pimdp_acc_labels = pimdp.accepting_labels
        acc_states = findall(>(0.), pimdp_acc_labels[:])

        @printf(f, "%d \n", sum(pimdp_acc_labels))
        [@printf(f, "%d ", acc_state-1) for acc_state in acc_states]
        @printf(f, "\n")

        for i in 1:(num_regions::Int)
            for q in states_sans_ends
                p_min_vec .= 0.0
                p_max_vec .= 0.0
                qp_test = delta_(q, dfa, imdp.labels[i])  # which dfa state it transitions to
                i_pimdp = (i-1)*sizeQ + dfa_num_map[q]   # this is the pimdp state number

                mode = res[i_pimdp]
                if mode == 0
                    # this state can't reach the goal, so it doesn't really matter, just give it a self transition
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, 0, i_pimdp-1, 1.0, 1.0)
                    update(pbar)
                else
                    mean_bounds = all_means[mode]
                    sig_bounds = all_sigs[mode]
                    sum_up = Atomic{Float64}(0)

                    # calculate transitions
                    if any([matching_label(label_fn[i], skip_) for skip_ in skip_labels])
                        # don't actually care about the transitions on this region either, just put 1's to itself
                        p_min_vec[i] = 1.0
                        p_max_vec[i] = 1.0
                    else
                        # find transition probabilities to
                        Threads.@threads for j in 1:(num_regions+1::Int)
                            post = extents[j, :, :]
                            p_min = 1.0
                            p_max = 1.0
                            for dim in 1:(num_dims::Int)
                                if p_max == 0.0
                                    continue
                                end

                                # bounds on the mean of the image of the pre-region
                                lower_mean = mean_bounds[i, dim, 1]
                                upper_mean = mean_bounds[i, dim, 2]

                                lower_sigma = sig_bounds[i, dim, 1] + dyn_noise[dim]
                                upper_sigma = sig_bounds[i, dim, 2] + dyn_noise[dim]  # this is a std deviation

                                post_bounds = post[dim, :]
                                post_low = post_bounds[1]
                                post_up = post_bounds[2]
                                post_mean = post_low + (post_up - post_low) / 2.0

                                if (lower_mean < post_mean) && (upper_mean > post_mean)
                                    # image contains the center of the post region
                                    middle_mean = post_mean
                                elseif upper_mean < post_mean
                                    # upper bound is closer to the center of the post
                                    middle_mean = upper_mean
                                else
                                    # lower bound is closer to the center of the post
                                    middle_mean = lower_mean
                                end

                                p_max *= max(prob_via_erf(post_up, post_low, middle_mean, upper_sigma),
                                             prob_via_erf(post_up, post_low, middle_mean, lower_sigma))

                                # min prob will be one of the four combos of largest/smallest mean/sigma
                                p_min *= minimum([prob_via_erf(post_up, post_low, lower_mean, lower_sigma),
                                                  prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                                                  prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                                                  prob_via_erf(post_up, post_low, upper_mean, lower_sigma)])

                            end
                            p_min = round(p_min; digits=5)
                            p_max = round(p_max; digits=5)

                            if j == num_regions + 1
                                p_min_ = p_min
                                p_min = 1.0 - p_max
                                p_max = 1.0 - p_min_
                            end
                            atomic_add!(sum_up, p_max)

                            p_min_vec[j] = p_min
                            p_max_vec[j] = p_max
                        end
                        if sum_up[] < 1
                            if sum_up[] > .999
                                # this is a numerical error, adjust the upper bounds to be equal to 1
                                p_max_vec ./ sum_up[]
                            else
                                @info "upper bound is bad, $(sum_up[]), $i, $mode"
                                exit()
                            end
                        end
                    end

                    # here I have the transition probabilities from state "i" under action "mode" and dfa state "q"
                    if qp_test == dfa_acc_state
#                         @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ), sum(p_min_vec), sum(p_max_vec))
                        @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ), 1.0, 1.0)
                    elseif qp_test == dfa_sink_state
#                         @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ)+addon-1, sum(p_min_vec), sum(p_max_vec))
                        @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ)+addon-1, 1.0, 1.0)
                    else
                        ij = findall(>(0.), p_max_vec[:])
                        qp_use = dfa_num_map[qp_test]
                        for s in ij
                            col_idx = (s-1)*sizeQ + qp_use
                            @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, col_idx-1, p_min_vec[s], p_max_vec[s])
                        end
                    end

                    update(pbar)
                end
            end
        end

        # self transitions outside of the defined space
        # do the same file writing for this state
        for q in states_sans_ends
            qp_test = delta_(q, dfa, imdp.labels[num_regions+1])  # which dfa state it transitions to
            i_pimdp = (num_regions)*sizeQ + dfa_num_map[q]   # this is the pimdp state number
            for mode in 1:(num_modes::Int)
                if qp_test == dfa_acc_state
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ), 1.0, 1.0)
                elseif qp_test == dfa_sink_state
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, (M*sizeQ)+addon-1, 1.0, 1.0)
                else
                    qp_use = dfa_num_map[qp_test]
                    col_idx = (num_regions)*sizeQ + qp_use
                    @printf(f, "%d %d %d %f %f\n", i_pimdp-1, mode-1, col_idx-1, 1.0, 1.0)
                end
                update(pbar)
            end
        end

        if !isnothing(dfa_acc_state)
            for mode in 1:(num_modes::Int)
                @printf(f, "%d %d %d %f %f", (M*sizeQ), mode-1, (M*sizeQ), 1.0, 1.0)
                if !isnothing(dfa_sink_state) || mode < num_modes
                    @printf(f, "\n")
                end
            end
        end

        if !isnothing(dfa_sink_state)
            for mode in 1:(num_modes::Int)
                @printf(f, "%d %d %d %f %f", (M*sizeQ)+addon-1, mode-1, (M*sizeQ)+addon-1, 1.0, 1.0)
                if mode < num_modes
                    @printf(f, "\n")
                end
            end
        end

    end

    return pimdp
end

#======================================================================================================================#
#  PIMDP/IMDP construction
#======================================================================================================================#


function label_states(labels, extents, unsafe_label, num_dims, num_regions)
    state_labels = []
    for idx in 1:(num_regions+1::Int)
        extent = extents[idx, :, :]
        if idx == num_regions+1
            append!(state_labels, [unsafe_label])
            continue
        end
        possible_labels = []
        for label in keys(labels)
            # does this extent fit in any labels
            in_ranges = false
            ranges = labels[label]
            for sub_range in ranges
                in_sub_range = true
                if isnothing(sub_range)
                    in_ranges = false
                    break
                end
                for dim in 1:(num_dims::Int)

                    if !(extent[dim, 1] >= sub_range[dim][1] && extent[dim, 2] <= sub_range[dim][2])
                        in_sub_range = false
                        break
                    end
                end
                if in_sub_range
                    in_ranges = true
                    break
                end
            end

            if in_ranges
                append!(possible_labels, [label])
            else
                append!(possible_labels, ["!" * label])
            end
        end
        extent_label = ""
        for idx_ in 1:length(possible_labels)
            label = possible_labels[idx_]
            if idx_ > 1
                extent_label *= '∧' * label
            else
                extent_label *= label
            end
        end
        append!(state_labels, [extent_label])
    end

    return state_labels

end


function fast_pimdp(dfa, imdp, num_regions)
    dfa_states = sort(dfa["states"])
    states_sans_ends = dfa_states
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)
    addon = 0
    if !isnothing(dfa_acc_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_acc_state, states_sans_ends)
    end

    if !isnothing(dfa_sink_state)
        sizeQ -= 1
        addon += 1
        filter!(x -> x != dfa_sink_state, states_sans_ends)
    end

    dfa_num_map = Dict()
    for i in 1:length(states_sans_ends)
        dfa_num_map[states_sans_ends[i]] = i
    end

    M = num_regions + 1

    pimdp_states = []
    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    for s in imdp.states
        for q in states_sans_ends
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end

    acc_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_acc_state)
        acc_labels[M*sizeQ + 1] = 1
        push!(pimdp_states, (-1, dfa_acc_state))
    end

    sink_labels = zeros(1, M*sizeQ + addon)
    if !isnothing(dfa_sink_state)
        sink_labels[end] = 1
        push!(pimdp_states, (-1, dfa_sink_state))
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, nothing, nothing, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)
    return pimdp

end


function delta_(q, dfa, label)
    # returns the dfa state that can be transitioned to from q under the label
    trans = dfa["trans"]

    labels = []
    for relation in trans
        test_label = relation[2]
        output = relation[3]
        if relation[1] == dfa["accept"]
            # make sure you can't fail once you win?
            test_label = "true"
            output = relation[1]
        end
        if relation[1] == q && matches(test_label, label)
            return output
        end
    end
end


function matches(test_label, compare_label)
    if test_label == "true"
        # any observation satisfies true
        return true
    end
    separated_test = Set(split(test_label, '∧'))
    separated_compare = Set(split(compare_label, '∧'))
    if issubset(separated_test, separated_compare)
        return true
    end
    if issubset(separated_compare, separated_test)
        return true
    end
    return false
end


function find_q(q, labels)
    q_return = []
    for (idx, label) in enumerate(labels)
        if matches(label, q)
            append!(q_return, [idx])
        end
    end
    return q_return
end


#======================================================================================================================#
#  PIMDP save and synthesis calls
#======================================================================================================================#


function validate_pimdp(pimdp)
    idx = 1
    for (minrow, maxrow) in zip(eachrow(pimdp.Pmin), eachrow(pimdp.Pmax))
        if sum(maxrow) < 1
            @info "bad max sum from pimdp idx $idx"
            @assert sum(maxrow) >= 1
        end
        if sum(minrow) > 1
            @assert sum(minrow) <= 1
        end
        idx += 1
    end
end


function write_pimdp_to_file(pimdp, filename)
    open(filename, "w") do f
        state_num = length(pimdp.states)
        action_num = length(pimdp.actions)

        @printf(f, "%d \n", state_num)
        @printf(f, "%d \n", length(pimdp.actions))

        # Get number of accepting states from the labels vector
        pimdp_acc_labels = pimdp.accepting_labels
        acc_states = findall(>(0.), pimdp_acc_labels[:])

        @printf(f, "%d \n", sum(pimdp_acc_labels))
        [@printf(f, "%d ", acc_state-1) for acc_state in acc_states]
        @printf(f, "\n")

        pimdp_sink_labels = pimdp.sink_labels
        sink_states = findall(>(0.), pimdp_sink_labels[:])

        for (i, sq) in enumerate(pimdp.states)

            for action in pimdp.actions
                row_idx = (i-1)*action_num + action
                ij = findall(>(0.), pimdp.Pmax[row_idx, :])
                for j=ij
                    @printf(f, "%d %d %d %f %f", i-1, action-1, j-1, pimdp.Pmin[row_idx, j], pimdp.Pmax[row_idx, j])
                    if (i < state_num || j < ij[end] || action < action_num)
                        @printf(f, "\n")
                    end
                end
            end
        end
    end
end


function run_prism(global_exp_dir, prism_filepath, refinement, pimdp)
    exe_path = "/home/robert/Desktop/prism-4.8-linux64-x86/bin/prism"  #TODO, figure out how to call this in a better way
    pm_file = prism_filepath * ".pm"
    prop_file = prism_filepath * ".props"
    strat_file = global_exp_dir * "/prism_strat_$(refinement).txt"

    @info "Running prism to generate strategy"
    prism_info = read(`$exe_path $pm_file $prop_file -exportstrat $strat_file`, String)

    dst_dir = dirname(prism_filepath)
    open("$dst_dir/prism-info-$refinement.txt", "w") do f
        print(f, prism_info)
    end

    @info "Loading prism strategy"
    res = zeros(Int, length(pimdp.states))
    open(strat_file, "r") do f
        for line in readlines(f)
            temp = split(line, ")")
            state = parse(Int, temp[1][2:end]) + 1  # ignore the ( and get the rest)
            action = parse(Int, temp[2][3:end]) + 1  # ignore the :a and get the rest
            res[state] = action
        end
    end
    # if an action is still 0, that means this pimdp state cannot reach the goal
    return res
end


function run_synthesis(imdp_file, k, refinement, EXPERIMENT_DIR; ep=1e-6, mode1="maximize", mode2="pessimistic")
    exe_path = EXPERIMENT_DIR * "/synthesis"  # Assumes that this program is in the experiement dir
    @assert isfile(imdp_file)
    res = read(`$exe_path $mode1 $mode2 $k $ep $imdp_file`, String)
    dst_dir = dirname(imdp_file)
    open("$dst_dir/synthesis-result-$refinement.txt", "w") do f
        print(f, res)
    end
    res_mat = res_to_numbers(res)
    return res_mat
end


function res_to_numbers(res_string)
    filter_res = replace(res_string, "\n"=>" ")
    res_split = split(filter_res)
    num_rows = Int(length(res_split)/4)

    res_mat = zeros(num_rows, 4)
    for i=1:num_rows
        res_mat[i, 1] = parse(Int, res_split[(i-1)*4+1])+1.
        res_mat[i, 2] = parse(Int, res_split[(i-1)*4+2])+1.
        res_mat[i, 3] = parse(Float64, res_split[(i-1)*4+3])
        res_mat[i, 4] = parse(Float64, res_split[(i-1)*4+4])
    end

    return res_mat
end


#======================================================================================================================#
#  IMDP save and pimdp dot graph
#======================================================================================================================#

function write_imdp_to_file_bounded(imdp, Qyes, Qno, filename)
    open(filename, "w") do f
        state_num = length(imdp.states)
        action_num =length(imdp.actions)
        @printf(f, "%d \n", state_num)
        @debug "Length actions: " length(imdp.actions)
        @printf(f, "%d \n", length(imdp.actions))
        # Get number of accepting states from the labels vector
        acc_states = Qyes
        @printf(f, "%d \n", length(acc_states))
        [@printf(f, "%d ", acc_state-1) for acc_state in acc_states]
        @printf(f, "\n")
        sink_states = Qno

        for i=1:state_num
            if isnothing(sink_states) || !(i∈sink_states)
                for action in imdp.actions
                    row_idx = (i-1)*action_num + action
                    ij = findall(>(0.), imdp.Pmax[(i-1)*action_num + action, :])
                    # Something about if the upper bound is less than one? Perhaps for numerical issues?
                    @debug action, i
                    psum = sum(imdp.Pmax[row_idx, :])
                    psum >= 1 ? nothing : throw(AssertionError("Bad max sum: $psum"))
                    for j=ij
                        @printf(f, "%d %d %d %f %f", i-1, action-1, j-1, imdp.Pmin[row_idx, j], imdp.Pmax[row_idx, j])
                        if (i < state_num || j < ij[end] || action < action_num)
                            @printf(f, "\n")
                        end
                    end
                end
            else
                @printf(f, "%d %d %d %f %f", i-1, 0, i-1, 1.0, 1.0)
                if i<state_num
                    @printf(f, "\n")
                end
            end
        end
    end
end


function create_dot_graph(imdp, filename)
    open(filename, "w") do f
        println(f, "digraph G {")
        println(f, "  rankdir=LR\n  node [shape=\"circle\"]\n  fontname=\"Lato\"\n  node [fontname=\"Lato\"]\n  edge [fontname=\"Lato\"]")
        println(f, "  size=\"8.2,8.2\" node[style=filled,fillcolor=\"#FDEDD3\"] edge[arrowhead=vee, arrowsize=.7]")

        # Initial State
        @printf(f, "  I [label=\"\", style=invis, width=0]\n  I -> %d\n", 1)

        state_num = length(imdp.states)
        for s in 1:state_num
            state = imdp.states[s][1]
            @printf(f, "  %d [label=<q<SUB>%d</SUB>>, xlabel=<%s>]\n", s, s, imdp.labels[state])

            for action in imdp.actions
                row_idx = (s-1)*length(imdp.actions) + action

                for idx in findall(>(0.), imdp.Pmax[row_idx, :])
                    state_p = imdp.states[idx][1]
                    @printf(f, "  %d -> %d [label=<a<SUB>%d</SUB>: %0.2f-%0.2f >]\n", s, idx, action, imdp.Pmin[row_idx,idx], imdp.Pmax[row_idx,idx])
                end
            end
        end
        println(f, "}")
    end
end
