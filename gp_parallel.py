from crown_scripts import *
from space_discretize_scripts import discretize_space, discretize_space_list
import multiprocessing as mp


############################################################################################
#  Transition probabilities
############################################################################################


def parallel_erf(extents, mode_info, num_dims, pre_idx, mode, sigs, threads=8):
    num_extents = len(extents)

    mean_info = mode_info[0][pre_idx]
    sig_info = mode_info[1][pre_idx]

    check_idx = get_reasonable_extents(extents, mean_info, sig_info, num_dims, sigs)
    check_idx.append(num_extents - 1)

    pool = mp.Pool(threads)
    results = pool.starmap(erf_transitions, [(num_extents, num_dims, sigs, mean_info, sig_info,
                                              post_idx, extents[post_idx]) for post_idx in check_idx])
    pool.close()

    max_out = np.zeros(num_extents)
    min_out = np.zeros(num_extents)
    res_ = False
    for idx, res in enumerate(results):
        max_out[check_idx[idx]] = res[1]
        min_out[check_idx[idx]] = res[0]
        if res[2]:
            res_ = True

    if res_:
        print(f'index {pre_idx} and mode {mode} has large mean bounds.')

    return min_out, max_out


def get_reasonable_extents(extents, mean_info, sig_info, num_dims, sigs):
    possible_intersect = {}
    intersect_list = []
    what_sig = 3  # be within 3 standard deviations of the mean
    for dim in range(num_dims):
        possible_intersect[dim] = []
        upper_sig = sig_info[dim][1]  # this is a std deviation
        upper_sig += sigs[dim]

        lower_bound = mean_info[dim][0] - what_sig * upper_sig
        upper_bound = mean_info[dim][1] + what_sig * upper_sig
        for idx, region in enumerate(extents):
            if region[dim][1] > lower_bound and region[dim][0] < upper_bound:
                # if the 3*sigma lower bound is less than the higher value of the region and the 3*sigma upper
                # bound is greater than the lower bound of the region then these might have probabilities
                # higher than like 0.001
                possible_intersect[dim].extend([idx])
        intersect_list.append(list(set(possible_intersect[dim])))

    intersects_ = list(set.intersection(*map(set, intersect_list)))

    return intersects_


def erf_transitions(num_extents, num_dims, sigs, mean_info, sig_info, post_idx, post_region):
    # check if these two regions intersect
    p_min = 1
    p_max = 1
    large_mean = False
    for dim in range(num_dims):
        lower_mean = mean_info[dim][0]
        upper_mean = mean_info[dim][1]

        upper_sigma = sig_info[dim][1] + sigs[dim]  # this is a std deviation
        lower_sigma = sig_info[dim][0] + sigs[dim]
        if lower_sigma is None:
            lower_sigma = upper_sigma * .7
        if lower_sigma > upper_sigma:
            lower_sigma = upper_sigma * .7

        post_bounds = post_region[dim]
        post_low = post_bounds[0]
        post_up = post_bounds[1]
        post_mean = post_low + (post_up - post_low) / 2

        # check to see transition probability to this region
        if lower_mean > post_up:
            # post entirely to the right of the pre
            # max prob is with the lowest mean
            p_max *= max(prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                         prob_via_erf(post_up, post_low, lower_mean, lower_sigma))

            # min prob is with the largest mean
            p_min *= min(prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                         prob_via_erf(post_up, post_low, upper_mean, lower_sigma))
        elif upper_mean < post_low:
            # post entirely to the left of the pre
            # max prob is with the largest mean
            p_max *= max(prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                         prob_via_erf(post_up, post_low, upper_mean, lower_sigma))

            # min prob is with the lowest mean
            p_min *= min(prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                         prob_via_erf(post_up, post_low, lower_mean, lower_sigma))
        else:
            # post intersects with pre
            # find point on post mean that is closest to the center of the pre with the smallest sigma

            if (lower_mean < post_mean) and (upper_mean > post_mean):
                middle_mean = post_mean
            elif lower_mean < post_mean:
                middle_mean = upper_mean
            else:
                middle_mean = lower_mean

            p_max *= prob_via_erf(post_up, post_low, middle_mean, lower_sigma)

            # min prob will be one of the four combos of largest/smallest mean/sigma
            p_min *= min([prob_via_erf(post_up, post_low, lower_mean, lower_sigma),
                          prob_via_erf(post_up, post_low, upper_mean, upper_sigma),
                          prob_via_erf(post_up, post_low, lower_mean, upper_sigma),
                          prob_via_erf(post_up, post_low, upper_mean, lower_sigma)])

    if post_idx == num_extents - 1:
        p_min_ = p_min
        p_min = 1 - p_max
        p_max = 1 - p_min_

    if p_min < 1e-4:
        p_min = 0
    if p_max < 1e-4:
        p_max = 0

    return [p_min, p_max, large_mean]


def prob_via_erf(lb, la, mean, sigma):
    # Pr(la <= X <= lb) when X ~ N(mean, sigma)
    return 0.5 * (math.erf((lb - mean) / (math.sqrt(2) * sigma)) - math.erf((la - mean) / (math.sqrt(2) * sigma)))


############################################################################################
#  DKL posteriors
############################################################################################


def run_dkl_in_parallel_just_bounds(extents, mode, nn_out_dim, crown_dir, experiment_dir, linear_bounds_info,
                                    linear_transform_m, linear_transform_b, threads=8, use_regular_gp=False):
    # This function does parallel calls to CROWN to get bounds on the NN output over input regions
    extent_len = len(extents)
    if not use_regular_gp:
        # this calls the Crown scripts in parallel
        os.chdir(crown_dir)

        dim_ = 0
        pool = mp.Pool(threads)
        results = pool.starmap(run_dkl_crown_parallel, [(extents[idx], crown_dir, nn_out_dim, mode, dim_,
                                                         idx) for idx in range(extent_len)])
        pool.close()

        # store the results
        os.chdir(experiment_dir)
        for index, lin_trans in enumerate(results):
            saved_vals = np.array([lin_trans[4].astype(np.float64), lin_trans[5].astype(np.float64)])
            linear_bounds_info[index] = saved_vals
            saved_m = np.array([lin_trans[0], lin_trans[1]])
            linear_transform_m[index] = saved_m
            saved_b = np.array([lin_trans[2], lin_trans[3]])
            linear_transform_b[index] = saved_b

    else:
        for index, region in enumerate(extents):
            x_min = [k[0] for k in list(region)]
            x_max = [k[1] for k in list(region)]
            saved_vals = [np.array(x_min).astype(np.float64), np.array(x_max).astype(np.float64)]
            linear_bounds_info[index] = saved_vals

    return linear_bounds_info, linear_transform_m, linear_transform_b


def refinement_algorithm(refine_states, region_data, extents, modes, crown_dir, nn_out_dim, global_exp_dir,
                         experiment_dir, nn_bounds_dir, refinement, threshold=1e-5, threads=8, use_regular_gp=False):

    # for each region in the refinement list, check which dimension causes the largest expansion across modes
    print("Starting refinement procedure")
    pool = mp.Pool(threads)
    dim_list = pool.starmap(dim_checker, [(extents[idx], region_data, modes, idx) for idx in refine_states])

    # now have which dimensions each region needs to be split across, pull out those extents, split them and rerun crown
    crown_regions = pool.starmap(get_new_regions, [(extents[idx], dim_list[i], threshold)
                                                   for i, idx in enumerate(refine_states)])

    pool.close()

    new_crown_regions = []
    for i in crown_regions:
        new_crown_regions.extend(i)

    # update extents
    extents = extents.tolist()
    domain = extents.pop()
    for idx in reversed(refine_states):
        extents.pop(idx)

    start_idx = len(extents)
    num_new = len(new_crown_regions)

    # add new extents to array and save
    extents.extend(new_crown_regions)
    extents.append(domain)
    filename = global_exp_dir + f"/extents_{refinement+1}"
    np.save(filename, np.array(extents))

    # these are the new indices to get bound for
    specific_extents = np.array([i for i in range(start_idx, start_idx + num_new)])

    lin_bounds = [[] for _ in range(num_new)]
    linear_trans_m = [[] for _ in range(num_new)]
    linear_trans_b = [[] for _ in range(num_new)]

    # nn_out_dim is the dimension of input as well for our cases
    mean_extension = np.zeros([num_new, nn_out_dim, 2]).tolist()
    sig_extension = np.zeros([num_new, nn_out_dim, 2]).tolist()
    for mode in modes:
        lin_bounds, linear_trans_m, linear_trans_b = run_dkl_in_parallel_just_bounds(new_crown_regions, mode,
                                                                                     nn_out_dim, crown_dir,
                                                                                     experiment_dir, lin_bounds,
                                                                                     linear_trans_m, linear_trans_b,
                                                                                     threads=threads,
                                                                                     use_regular_gp=use_regular_gp)
        filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_{refinement}.npy"
        lin_bounds_old = np.load(filename)
        lin_bounds_old = lin_bounds_old.tolist()
        lin_bounds_old.extend(lin_bounds)

        filename = nn_bounds_dir + f"/linear_trans_m_{mode + 1}_{refinement}.npy"
        lin_trans_m_old = np.load(filename)
        lin_trans_m_old = lin_trans_m_old.tolist()
        lin_trans_m_old.extend(linear_trans_m)

        filename = nn_bounds_dir + f"/linear_trans_b_{mode + 1}_{refinement}.npy"
        lin_trans_b_old = np.load(filename)
        lin_trans_b_old = lin_trans_b_old.tolist()
        lin_trans_b_old.extend(linear_trans_b)

        mean_bounds = region_data[mode][0]
        mean_bounds = mean_bounds.tolist()
        mean_bounds.extend(mean_extension)

        sig_bounds = region_data[mode][1]
        sig_bounds = sig_bounds.tolist()
        sig_bounds.extend(sig_extension)

        # remove the refined indices, also save a file listing what regions need updated
        # also need to modify mean and sig bound structures to be the appropriate size
        for idx in reversed(refine_states):
            # TODO, make this modular for local gps?
            # TODO, figure out how to modify transition probabilities efficiently
            lin_bounds_old.pop(idx)
            lin_trans_m_old.pop(idx)
            lin_trans_b_old.pop(idx)
            mean_bounds.pop(idx)
            sig_bounds.pop(idx)

        # save the bounds for new regions and adjusted mean/sig arrays
        np.save(nn_bounds_dir + f"/linear_bounds_{mode + 1}_{refinement+1}", np.array(lin_bounds_old))
        np.save(nn_bounds_dir + f"/linear_trans_m_{mode + 1}_{refinement+1}", np.array(lin_trans_m_old))
        np.save(nn_bounds_dir + f"/linear_trans_b_{mode + 1}_{refinement+1}", np.array(lin_trans_b_old))
        np.save(global_exp_dir + f"/mean_data_{mode + 1}_{refinement+1}", np.array(mean_bounds))
        np.save(global_exp_dir + f"/sig_data_{mode + 1}_{refinement+1}", np.array(sig_bounds))

        # save the extents to refine
        for dim in range(nn_out_dim):
            sub_idx = 0
            dim_region_filename = nn_bounds_dir + f"/linear_bounds_{mode + 1}_{sub_idx + 1}_{dim + 1}"
            np.save(dim_region_filename+f"_these_indices_{refinement+1}", specific_extents)


def dim_checker(region, region_data, modes, idx):
    x_ranges = [region[k] for k in range(len(region))]
    vertices = list(itertools.product(*x_ranges))

    xi_max = 0
    max_dim = None
    for mode in modes:
        lin_transform = region_data[mode][2][idx]
        lA = lin_transform[0]
        uA = lin_transform[1]
        l_bias = lin_transform[2]
        u_bias = lin_transform[3]

        v_low = []
        v_up = []
        for vertex in vertices:
            x_input = np.array(vertex)
            l_out = lA @ x_input + l_bias
            u_out = uA @ x_input + u_bias
            v_low.append(l_out.tolist()[0])
            v_up.append(u_out.tolist()[0])

        checked_pairs = []
        for idx_1, v1 in enumerate(vertices):
            for idx_2, v2 in enumerate(vertices):
                if idx_1 == idx_2:
                    continue

                if (v2, v1) in checked_pairs:
                    continue

                matching_dims = np.where(np.array(v1) == np.array(v2))[0].tolist()
                if len(matching_dims) == 0:
                    continue

                checked_pairs.append((v1, v2))

                vertex_norm = np.linalg.norm(np.array(v1) - np.array(v2), ord=2, axis=0)
                upper_norm = np.linalg.norm(np.array(v_up[idx_1]) - np.array(v_low[idx_2]), ord=2, axis=0)
                lower_norm = np.linalg.norm(np.array(v_low[idx_1]) - np.array(v_up[idx_2]), ord=2, axis=0)

                xi_a = max(upper_norm / vertex_norm, lower_norm / vertex_norm)
                if xi_a > xi_max:
                    xi_max = xi_a
                    max_dim = matching_dims

    return max_dim


def get_new_regions(old_region, dim_idx, threshold):
    region = {k: old_region[k] for k in range(len(old_region))}
    grid_size = {}
    for k in range(len(old_region)):
        side = old_region[k]
        grid_len = (side[1] - side[0])
        split = 1.0
        if k in dim_idx:
            split = 2.0
        grid_size[k] = max(grid_len / split, threshold)
    new_regions = discretize_space_list(region, grid_size, include_space=False)

    return new_regions

############################################################################################
#  Support functions
############################################################################################


def se_(sig2, l2, x1, x2):
    dx = x1 - x2
    return sig2 * np.exp(-0.5 * np.dot(dx, dx) / l2)


def create_kernel_mat(sig2, l2, noise, X, n_obs):
    # return a square mat k(X,X)
    K_mat = noise * np.identity(n_obs)

    for x1_idx in range(n_obs):
        x1 = X[:, x1_idx]
        for x2_idx in range(n_obs):
            x2 = X[:, x2_idx]
            K_mat[x1_idx, x2_idx] += se_(sig2, l2, x1, x2)

    return K_mat


def extents_in_region(extents, region):
    # returns a list of indices of extents that are inside the region
    valid = []
    for extent_idx, extent in enumerate(extents):
        in_region = True
        for idx, k in enumerate(list(extent)):
            if k[0] < region[idx][0] or k[1] > region[idx][1]:
                # not in this dimension of the region
                in_region = False
                break
        if in_region:
            valid.append(extent_idx)

    return valid


def convert_to_list(region):
    dims = len(region)
    rl = {}
    for dim in range(dims):
        rl[dim] = region[dim]

    return rl
