
using Base.Threads
using LinearAlgebra
using SpecialFunctions
using PyCall
@pyimport numpy


function prob_via_erf(lb, la, mean, sigma)
    # Pr(la <= X <= lb) when X ~ N(mean, sigma)
    return 0.5 * (erf((lb - mean) / (sqrt(2) * sigma)) - erf((la - mean) / (sqrt(2) * sigma)))
end


function refine_check(res, q_question, n_best, num_dfa_states, p_action_diff; dfa_init_state=1)
    # this returns the n_best regions to refine by assessing the difference between upper and lower probability of
    # satisfying as well as outgoing transition probability

    indVmin = res[:, 3]
    indVmax = res[:, 4]

    maxPrs = indVmax[dfa_init_state:num_dfa_states:end]
    minPrs = indVmin[dfa_init_state:num_dfa_states:end]

    # determine which states to refine
    theta = zeros(length(q_question))

    for (idx, s) in enumerate(q_question)
        sat_prob = (maxPrs[s] - minPrs[s])

        i_pimdp = (s-1)*num_dfa_states + dfa_init_state
        p_actions = p_action_diff[i_pimdp]

        theta[idx] = p_actions * sat_prob
    end

    n_best = min(n_best, length(q_question))
    refine_idx = sortperm(theta)[1:n_best]
    refine_regions = []
    for i in refine_idx
        append!(refine_regions, [q_question[i]])
    end

    refine_regions = sort(refine_regions)
    return refine_regions
end


function refinement_algorithm(refine_states, extents, modes, num_dims, global_dir_name, nn_bounds_dir, refinement;
                              threshold=1e-5, use_regular_gp=false)

    # load prior info
    linear_transforms = numpy.load(nn_bounds_dir * "/linear_trans_m_1_$(refinement).npy")
    linear_bias = numpy.load(nn_bounds_dir * "/linear_trans_b_1_$(refinement).npy")
    linear_bounds = numpy.load(nn_bounds_dir*"/linear_bounds_1_$(refinement).npy")
    for mode in 2:num_modes
        linear_transforms = cat(linear_transforms, numpy.load(nn_bounds_dir * "/linear_trans_m_$(mode)_$(refinement).npy"), dims=6)
        linear_bias = cat(linear_bias, numpy.load(nn_bounds_dir * "/linear_trans_b_$(mode)_$(refinement).npy"), dims=5)
        linear_bounds = cat(linear_bounds, numpy.load(nn_bounds_dir * "/linear_bounds_$(mode)_$(refinement).npy"), dims=4)
    end

    # TODO, need to figure out how to do this in parallel
    keep_states = []
    for i in 1:size(extents)[1]-1
        if i in refine_states
            continue
        else
            push!(keep_states, i)
        end
    end

    kept = length(keep_states)
    new_extents = nothing
    new_transforms = []
    new_bias = []
    new_linear_bounds = []
    num_added = 0
    for idx in refine_states
        # find which dimensions have the largest growth
        refine_dim = dim_checker(extents[idx, :, :], linear_transforms, modes, idx, use_regular_gp)
        # split the extent along that dimensions
        new_regions = extent_splitter(extents[idx, :, :], refine_dim, threshold)
        if isnothing(new_extents)
            new_extents = new_regions
        else
            new_extents = vcat(new_extents, new_regions)
        end

        # get the NN posterior for the new regions
        repeats = size(new_regions)[1]
        for sub_idx in 1:repeats
            temp = size(linear_transforms)
            added_transform = reshape(linear_transforms[idx,:,:,:,:,:], (1, temp[2], temp[3], temp[4] ,temp[5], temp[6]))
            new_transforms = cat(new_transforms, added_transform, dims=1)

            temp = size(linear_bias)
            added_bias = reshape(linear_bias[idx,:,:,:,:], (1, temp[2], temp[3], temp[4] ,temp[5]))
            new_bias = cat(new_bias, added_bias, dims=1)

            # use this transform to get bounds with the vertices of the new extents
            temp = new_posts_fnc(new_regions[sub_idx, :, :], linear_transforms, linear_bias, modes, idx, num_dims)
            new_linear_bounds = cat(new_linear_bounds, temp, dims=1)
            num_added += 1
        end
    end

    specific_extents = collect((kept):(kept+num_added-1))
    @info "Added $num_added regions"

    # now re-save files for refinement+1
    new_extents = vcat(extents[keep_states, :, :], new_extents)
    domain = reshape(extents[end, :, :], (1, num_dims, 2))
    new_extents = vcat(new_extents, domain)
    numpy.save(global_exp_dir * "/extents_$(refinement+1)", new_extents)

    new_transforms = vcat(linear_transforms[keep_states, :, :, :, :, :], new_transforms)

    new_bias = vcat(linear_bias[keep_states, :, :, :, :], new_bias)

    new_linear_bounds = vcat(linear_bounds[keep_states, :, :, :], new_linear_bounds)

    additional_array = zeros(num_added, num_dims, 2)

    for mode in modes
        numpy.save(nn_bounds_dir * "/linear_trans_m_$(mode)_$(refinement+1)", new_transforms[:,:,:,:,:,mode])
        numpy.save(nn_bounds_dir * "/linear_trans_b_$(mode)_$(refinement+1)", new_bias[:,:,:,:,mode])
        numpy.save(nn_bounds_dir * "/linear_bounds_$(mode)_$(refinement+1)", new_linear_bounds[:,:,:,mode])

        mean_bound = numpy.load(global_exp_dir*"/mean_data_$(mode)_$refinement.npy")
        mean_bound = vcat(mean_bound[keep_states, :, :], additional_array)
        numpy.save(global_exp_dir*"/mean_data_$(mode)_$(refinement+1)", mean_bound)

        sig_bound = numpy.load(global_exp_dir*"/sig_data_$(mode)_$refinement.npy")
        sig_bound = vcat(sig_bound[keep_states, :, :], additional_array)
        numpy.save(global_exp_dir*"/sig_data_$(mode)_$(refinement+1)", sig_bound)

        for dim in 1:num_dims
            dim_region_filename = nn_bounds_dir * "/linear_bounds_$(mode)_1_$dim"
            numpy.save(dim_region_filename*"_these_indices_$(refinement+1)", specific_extents)
        end
    end

end


function new_posts_fnc(region, linear_transforms, linear_bias, modes, idx, dims)
    x_ranges = [region[k,:] for k in 1:(size(region)[1])]
    vertices = [[vert...] for vert in Base.product(x_ranges...)]

    new_posts = nothing
    for mode in modes
        lA = linear_transforms[idx,1,1,:,:,mode]
        uA = linear_transforms[idx,2,1,:,:,mode]
        l_bias = linear_bias[idx,1,1,:,mode]
        u_bias = linear_bias[idx,2,1,:,mode]

        v_low = nothing
        v_up = nothing
        for vertex in vertices
            l_out = lA * vertex + l_bias
            u_out = uA * vertex + u_bias
            if isnothing(v_low)
                v_low = l_out
                v_up = u_out
            else
                for dim in 1:dims
                    v_low[dim] = min(v_low[dim], l_out[dim])
                    v_up[dim] = max(v_up[dim], u_out[dim])
                end
            end
        end

        if isnothing(new_posts)
            new_posts = vcat(transpose(v_low), transpose(v_up))
        else
            new_posts = cat(new_posts, vcat(transpose(v_low), transpose(v_up)), dims=3)
        end
    end

    test = size(new_posts)
    reshaping = [1]
    for i in 1:length(test)
        push!(reshaping, test[i])
    end

    return reshape(new_posts, reshaping...)

end


function dim_checker(region, linear_transforms, modes, idx, use_regular_gp)
    x_ranges = [region[k,:] for k in 1:(size(region)[1])]
    vertices = [[vert...] for vert in Base.product(x_ranges...)]

    xi_max = 0
    max_dim = nothing
    for mode in modes
        if use_regular_gp
            lA = 1
            uA = 1
        else
            lA = linear_transforms[idx,1,1,:,:,mode]
            uA = linear_transforms[idx,2,1,:,:,mode]
        end

        v_low = []
        v_up = []
        for vertex in vertices
            l_out = lA * vertex
            u_out = uA * vertex
            append!(v_low, [l_out])
            append!(v_up, [u_out])
        end

        checked_pairs = []
        for (idx1, v1) in enumerate(vertices)
            for (idx2, v2) in enumerate(vertices)
                if v1 == v2
                    continue
                end

                if (v2, v1) in checked_pairs
                    # don't check pairs of vertices again
                    continue
                end

                matching_dims = findall(in(v1), v2)
                if length(matching_dims) == 0
                    continue
                end

                append!(checked_pairs, [(v1, v2)])

                vertex_norm = norm(v1 - v2)
                upper_norm = norm(v_up[idx1] - v_low[idx2])
                lower_norm = norm(v_low[idx1] - v_up[idx2])

                xi_a = max(upper_norm / vertex_norm, lower_norm / vertex_norm)
                if xi_a > xi_max
                    xi_max = xi_a
                    max_dim = matching_dims
                end
            end
        end
    end

    return max_dim
end


function extent_splitter(extent, refine_dims, threshold)

    num_dims = size(extent)[1]
    grid_size = []
    num_new = 1
    for dim in 1:num_dims
        dx = extent[dim, 2] - extent[dim, 1]
        if dim in refine_dims
            dx /= 2.0
            num_new *= 2
        end
        if dx < threshold
            dx *= 2.0  # can't split this dimension
            num_new /= 2
        end
        push!(grid_size, dx)
    end

    # now I have the grid size, figure out how to split the space according to that grid
    dim_ranges = [collect(extent[dim, 1]:grid_size[dim]:extent[dim, 2]) for dim in 1:num_dims]
    temp = [[[dim_ranges[dim][i], dim_ranges[dim][i+1]] for i in 1:(length(dim_ranges[dim]) - 1)] for dim in 1:num_dims]
    state_extents = (Base.product(temp...))

    discrete_sets = zeros(num_new, num_dims, 2)
    for (i, state) in enumerate(state_extents)
        for j in 1:size(extent)[1]
            discrete_sets[i, j, :] = state[j]
        end
    end
    return discrete_sets[:, :, :]


end