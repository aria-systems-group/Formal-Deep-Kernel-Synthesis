
@everywhere using SpecialFunctions
@everywhere function prob_via_erf(lb, la, mean, sigma)
    # Pr(la <= X <= lb) when X ~ N(mean, sigma)
    return 0.5 * (erf((lb - mean) / (sqrt(2) * sigma)) - erf((la - mean) / (sqrt(2) * sigma)))
end

using SparseArrays
using SharedArrays
using Printf
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


function imdp_probs(extents, dyn_noise, global_exp_dir, refinement, num_modes, num_regions, num_dims)

#     @info "Doing something weird"

    minPrs = spzeros((num_regions+1)*num_modes, num_regions+1)
    maxPrs = spzeros((num_regions+1)*num_modes, num_regions+1)

    convert(SharedArray, extents)
    for mode in 1:(num_modes::Int)
        mean_bounds = numpy.load(global_exp_dir*"/mean_data_$mode" * "_$refinement.npy")
        sig_bounds = numpy.load(global_exp_dir*"/sig_data_$mode" * "_$refinement.npy")
        convert(SharedArray, mean_bounds)
        convert(SharedArray, sig_bounds)
        for i in 1:(num_regions::Int)
            pre = extents[i,:,:]
            row_idx = (i - 1)*num_modes + mode  # should start at 1 and end at num_regions*num_modes
            p_min_vec = SharedArray(zeros(num_regions+1))
            p_max_vec = SharedArray(zeros(num_regions+1))
            # find transition probabilities to
            @sync @distributed for j in 1:(num_regions+1::Int)
                # threads are good here because it's low computation complexity
                post = extents[j, :, :]
                p_min = 1.
                p_max = 1.
                for dim in 1:(num_dims::Int)
                    if p_max == 0
                        continue
                    end

                    # bounds on the mean of the image of the pre-region
                    lower_mean = mean_bounds[i, dim, 1]
                    upper_mean = mean_bounds[i, dim, 2]

                    lower_sigma = sig_bounds[i, dim, 1]
                    upper_sigma = sig_bounds[i, dim, 2] + sqrt(dyn_noise[dim])  # this is a std deviation

                    post_bounds = post[dim, :]
                    post_low = post_bounds[1]
                    post_up = post_bounds[2]
                    post_mean = post_low + (post_up - post_low) / 2.0

                    if lower_mean > post_up
                        # post entirely to the right of the pre
                        # max prob is with the lower mean bound (closest to post region)
                        p_max *= max(prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                                     prob_via_erf(post_up, post_low, lower_mean, lower_sigma))

                        # min prob is with the upper mean bound (far from post region)
                        p_min *= min(prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                                     prob_via_erf(post_up, post_low, upper_mean, lower_sigma))
                    elseif upper_mean < post_low
                        # post entirely to the left of the pre
                        # max prob is with the upper mean bound (closest to post region)
                        p_max *= max(prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                                     prob_via_erf(post_up, post_low, upper_mean, lower_sigma))

                        # min prob is with the lower mean bound (far from post region)
                        p_min *= min(prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                                     prob_via_erf(post_up, post_low, lower_mean, lower_sigma))
                    else
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

                        p_max *= prob_via_erf(post_up, post_low, middle_mean, lower_sigma)

                        # min prob will be one of the four combos of largest/smallest mean/sigma
                        p_min *= minimum([prob_via_erf(post_up, post_low, lower_mean, lower_sigma),
                                      prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                                      prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                                      prob_via_erf(post_up, post_low, upper_mean, lower_sigma)])

                    end

                    if p_min < 1e-4
                        p_min = 0.
                    end
                    if p_max < 1e-4
                        p_max = 0.
                    end
                end

                if j == num_regions + 1
                    p_min_ = p_min
                    p_min = 1 - p_max
                    p_max = 1 - p_min_
                end


                if p_min < 1e-4
                    p_min = 0.
                end
                if p_max < 1e-4
                    p_max = 0.
                end

                p_min_vec[j] = p_min
                p_max_vec[j] = p_max

#                 minPrs[row_idx, j] = p_min
#                 maxPrs[row_idx, j] = p_max

            end

            minPrs[row_idx, :] = p_min_vec
            maxPrs[row_idx, :] = p_max_vec
        end

        # self transitions outside of the defined space
        row_idx = num_regions*num_modes + mode
        minPrs[row_idx, num_regions+1] = 1.
        maxPrs[row_idx, num_regions+1] = 1.

    end

    for (minrow, maxrow) in zip(eachrow(minPrs), eachrow(maxPrs))
        @assert sum(maxrow) >= 1.
        @assert sum(minrow) <= 1.
    end

    return minPrs, maxPrs
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


function fast_pimdp(dfa, imdp)
    dfa_states = sort(dfa["states"])
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)
    N, M = size(imdp.Pmin)

    pimdp_states = []
    pimdp_actions = imdp.actions

    for s in imdp.states
        for q in dfa_states
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end

    acc_labels = zeros(1, M*sizeQ)
    if !isnothing(dfa_acc_state)
        acc_labels[dfa_acc_state:sizeQ:M*sizeQ] .= 1
    end

    sink_labels = zeros(1, M*sizeQ)
    if !isnothing(dfa_sink_state)
        sink_labels[dfa_sink_state:sizeQ:M*sizeQ] .= 1
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, nothing, nothing, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)
    return pimdp
end


function construct_DFA_IMDP_product_mkII(dfa, imdp)

    dfa_states = sort(dfa["states"])
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)

    # Transition matrix size will be Nx|Q| -- or the original transition matrix permutated with the number of states in DFA
    Pmin = imdp.Pmin
    Pmax = imdp.Pmax
    N, M = size(Pmin)
    Pmin_new = spzeros(N*sizeQ, M*sizeQ)
    Pmax_new = spzeros(N*sizeQ, M*sizeQ)

    pimdp_states = []
    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    pimdp_trans_states = []

    for s in imdp.states
        for q in dfa_states
            new_state = (s, q)
            push!(pimdp_states, new_state)
            if (q == dfa_acc_state) || (q == dfa_sink_state)
                # accepting/sink states have self transitions for LTLf
                for a in pimdp_actions
                    row_idx = (s-1)*sizeQ*sizeA + (q-1)*sizeA + a
                    col_idx = (s-1)*sizeQ + q
                    Pmin_new[row_idx, col_idx] = 1.0
                    Pmax_new[row_idx, col_idx] = 1.0
                end
            else
                push!(pimdp_trans_states, new_state)
            end
        end
    end

    for sq in pimdp_trans_states
        qp_test = delta_(sq[2], dfa, imdp.labels[sq[1]])
        for a in pimdp_actions
            # new transition matrix row
            row_idx = (sq[1]-1)*sizeQ*sizeA + (sq[2]-1)*sizeA + a
            ij = findall(>(0.), Pmax[(sq[1]-1)*sizeA + a, :])
            for s in ij
                sqp = (s, qp_test)
                # Get the corresponding entry of the transition interval matrices
                # new transition matrix column
                col_idx = (sqp[1]-1)*sizeQ + sqp[2]
                if (qp_test == dfa_acc_state) || (qp_test == dfa_sink_state)
                    # transition to only one sink/accept state
                    col_idx = qp_test
                    Pmin_new[row_idx, col_idx] += Pmin[(sq[1]-1)*sizeA + a, sqp[1]]
                    Pmax_new[row_idx, col_idx] += Pmax[(sq[1]-1)*sizeA + a, sqp[1]]
                else
                    Pmin_new[row_idx, col_idx] = Pmin[(sq[1]-1)*sizeA + a, sqp[1]]
                    Pmax_new[row_idx, col_idx] = Pmax[(sq[1]-1)*sizeA + a, sqp[1]]
                end
            end
        end
    end

    acc_labels = zeros(1, M*sizeQ)
    if !isnothing(dfa_acc_state)
        acc_labels[dfa_acc_state] = 1
    end

    sink_labels = zeros(1, M*sizeQ)
    if !isnothing(dfa_sink_state)
        sink_labels[dfa_sink_state] = 1
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, Pmin_new, Pmax_new, imdp.labels, acc_labels, sink_labels,
                       imdp.extents)
    return pimdp
end


function construct_DFA_IMDP_product(dfa, imdp)

    dfa_states = sort(dfa["states"])
    dfa_acc_state = dfa["accept"]
    dfa_sink_state = dfa["sink"]

    sizeQ = length(dfa_states)

    # Transition matrix size will be Nx|Q| -- or the original transition matrix permutated with the number of states in DFA
    Pmin = imdp.Pmin
    Pmax = imdp.Pmax
    N, M = size(Pmin)
    Pmin_new = spzeros(N*sizeQ, M*sizeQ)
    Pmax_new = spzeros(N*sizeQ, M*sizeQ)

    pimdp_states = []
    pimdp_actions = imdp.actions
    sizeA = length(pimdp_actions)

    for s in imdp.states
        for q in dfa_states
            new_state = (s, q)
            push!(pimdp_states, new_state)
        end
    end

    for sq in pimdp_states
        qp_test = delta_(sq[2], dfa, imdp.labels[sq[1]])
        for a in pimdp_actions
            # new transition matrix row
            row_idx = (sq[1]-1)*sizeQ*sizeA + (sq[2]-1)*sizeA + a
            for s in imdp.states
                sqp = (s, qp_test)
                # Get the corresponding entry of the transition interval matrices
                if (sq[2] == dfa_acc_state && sqp[2] == dfa_acc_state) || (sq[2] == dfa_sink_state && sqp[2] == dfa_sink_state)
                    col_idx = (sq[1]-1)*sizeQ + sq[2]
                    Pmin_new[row_idx, col_idx] = 1.0
                    Pmax_new[row_idx, col_idx] = 1.0
                else
                    # new transition matrix column
                    col_idx = (sqp[1]-1)*sizeQ + sqp[2]
                    Pmin_new[row_idx, col_idx] = Pmin[(sq[1]-1)*sizeA + a, sqp[1]]
                    Pmax_new[row_idx, col_idx] = Pmax[(sq[1]-1)*sizeA + a, sqp[1]]
                end
            end
        end
    end

    acc_labels = zeros(1, M*sizeQ)
    if !isnothing(dfa_acc_state)
        acc_labels[dfa_acc_state:sizeQ:M*sizeQ] .= 1
    end

    sink_labels = zeros(1, M*sizeQ)
    if !isnothing(dfa_sink_state)
        sink_labels[dfa_sink_state:sizeQ:M*sizeQ] .= 1
    end

    pimdp = PIMDPModel(pimdp_states, imdp.actions, Pmin_new, Pmax_new, imdp.labels, acc_labels, sink_labels,
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
    for (minrow, maxrow) in zip(eachrow(pimdp.Pmin), eachrow(pimdp.Pmax))
        @assert sum(maxrow) >= 1
        @assert sum(minrow) <= 1
    end
end


function write_pimdp_to_file_mkII(pimdp, filename)
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
                psum = sum(pimdp.Pmax[row_idx, :])
                psum >= 1 ? nothing : throw(AssertionError("Bad max sum: $psum, $sq"))
                psum = sum(pimdp.Pmin[row_idx, :])
                psum <= 1 ? nothing : throw(AssertionError("Bad min sum: $psum"))
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

        for i=1:state_num
            if isnothing(sink_states) || !(i∈sink_states)
                for action in pimdp.actions
                    row_idx = (i-1)*action_num + action
                    ij = findall(>(0.), pimdp.Pmax[row_idx, :])
                    psum = sum(pimdp.Pmax[row_idx, :])
                    psum >= 1 ? nothing : throw(AssertionError("Bad max sum: $psum"))
                    psum = sum(pimdp.Pmin[row_idx, :])
                    psum <= 1 ? nothing : throw(AssertionError("Bad min sum: $psum"))
                    for j=ij
                        @printf(f, "%d %d %d %f %f", i-1, action-1, j-1, pimdp.Pmin[row_idx, j], pimdp.Pmax[row_idx, j])
                        if (i < state_num || j < ij[end] || action < action_num)
                            @printf(f, "\n")
                        end
                    end
                end
            else
                [@printf(f, "%d %d %d %f %f\n", i-1, j, i-1, 1.0, 1.0) for j=0:action_num-1]
            end
        end
    end
end


function run_imdp_synthesis(imdp_file, k, refinement; ep=1e-6, mode1="maximize", mode2="pessimistic")
    exe_path = "/usr/local/bin/synthesis"  # Assumes that this program is on the user's path
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