from import_script import *
from crown_scripts import *
import multiprocessing as mp
from space_discretize_scripts import *

from juliacall import Main as jl

# try:
#     jl.seval("using PosteriorBounds")
#
#     compute_mean_bounds_jl = jl.seval("PosteriorBounds.compute_μ_bounds_bnb_tmp")
#     compute_sig_bounds_jl = jl.seval("PosteriorBounds.compute_σ_bounds_bnb_tmp")
#     theta_vectors = jl.seval("PosteriorBounds.theta_vectors")
#     scale_cKinv = jl.seval("PosteriorBounds.scale_cK_inv")
# except:
jl.seval("using Pkg")
Pkg_add = jl.seval("Pkg.add")
Pkg_add(url="https://github.com/aria-systems-group/PosteriorBounds.jl")
jl.seval("using PosteriorBounds")

compute_mean_bounds_jl = jl.seval("PosteriorBounds.compute_μ_bounds_bnb_tmp")
compute_sig_bounds_jl = jl.seval("PosteriorBounds.compute_σ_bounds_bnb_tmp")
theta_vectors = jl.seval("PosteriorBounds.theta_vectors")
scale_cKinv = jl.seval("PosteriorBounds.scale_cK_inv")


############################################################################################
#  Transition probabilities
############################################################################################

def parallel_erf(extents, mode_info, num_dims, pre_idx, mode, threads=8):
    num_extents = len(extents)

    mean_info = mode_info[0][pre_idx]
    sig_info = mode_info[1][pre_idx]

    check_idx = get_reasonable_extents(extents, mean_info, sig_info, num_dims)
    check_idx.append(num_extents-1)

    pool = mp.Pool(threads)
    results = pool.starmap(erf_transitions, [(num_extents, num_dims, mean_info, sig_info,
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


def get_reasonable_extents(extents, mean_info, sig_info, num_dims):
    possible_intersect = {}
    intersect_list = []
    what_sig = 3  # be within 3 standard deviations of the mean
    for dim in range(num_dims):
        possible_intersect[dim] = []
        upper_sig = sig_info[dim][1]  # this is a std deviation

        lower_bound = mean_info[dim][0] - what_sig*upper_sig
        upper_bound = mean_info[dim][1] + what_sig*upper_sig
        for idx, region in enumerate(extents):
            if region[dim][1] > lower_bound and region[dim][0] < upper_bound:
                # if the 3*sigma lower bound is less than the higher value of the region and the 3*sigma upper
                # bound is greater than the lower bound of the region then these might have probabilities
                # higher than like 0.001
                possible_intersect[dim].extend([idx])
        intersect_list.append(list(set(possible_intersect[dim])))

    intersects_ = list(set.intersection(*map(set, intersect_list)))

    return intersects_


def erf_transitions(num_extents, num_dims, mean_info, sig_info, post_idx, post_region):

    # check if these two regions intersect
    p_min = 1
    p_max = 1
    large_mean = False
    for dim in range(num_dims):
        lower_mean = mean_info[dim][0]
        upper_mean = mean_info[dim][1]
        if abs(upper_mean - lower_mean) > .35:
            large_mean = True

        upper_sigma = sig_info[dim][1]  # this is a std deviation

        lower_sigma = sig_info[dim][0]
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


def dkl_posts(dkl_by_dim, x_data, y_data, mode, extents, nn_out_dim, crown_dir, experiment_dir, threads=8):
    # returns a list organized as [mean bounds, sig bounds, NN linear transform]
    # mean bounds is a list of tuples of tuples [((dim_upper, dim_lower) for each dim) for each region], same for sig
    # NN linear transform is a list of tuples of tuples [((lA, uA, l_bias, u_bias, bounds_min, bounds_max) for each dim) for each region]

    num_regions = len(extents) - 1
    num_dims = len(dkl_by_dim)
    n_obs = max(np.shape(y_data))

    linear_transform_info = [[[] for dim in range(num_dims)] for idx in range(num_regions)]
    mean_bound = [[[None, None] for dim in range(num_dims)] for idx in range(num_regions)]
    sig_bound = [[[None, None] for dim in range(num_dims)] for idx in range(num_regions)]

    # first get crown posts for each dim
    # dkl_by_dim is a list [[model, likelihood] for each dim]

    for dim in range(num_dims):
        linear_transform_info = run_dkl_in_parallel(extents, mode, dim, nn_out_dim, crown_dir, experiment_dir,
                                                    linear_transform_info, threads=threads)

        # setup some intermediate values
        # pass x_train through the NN portion to get the kernel inputs
        dim_gp = dkl_by_dim[dim]
        model = dim_gp[0]
        model.cpu()  # unfortunately needs to be on cpu to access values
        nn_portion = model.feature_extractor
        with torch.no_grad():
            kernel_inputs = nn_portion.forward(x_data)

        noise = model.likelihood.noise.item()
        noise_mat = noise * np.identity(np.shape(kernel_inputs)[0])

        covar_module = model.covar_module
        kernel_mat = covar_module(kernel_inputs)
        kernel_mat = kernel_mat.evaluate()
        K_ = kernel_mat.detach().numpy() + noise_mat
        # enforce perfect symmetry, it is very close but causes errors when computing sig bounds
        K_ = (K_ + K_.transpose()) / 2.
        K_inv = np.linalg.inv(K_)  # only need to do this once per dim, yay

        y_dim = np.reshape(y_data[:, dim], (n_obs,))
        alpha_vec = K_inv @ y_dim  # TODO, store this for refinement?
        length_scale = model.covar_module.base_kernel.lengthscale.item()
        output_scale = model.covar_module.outputscale.item()

        # convert to julia input structure
        x_gp = np.array(np.transpose(kernel_inputs.detach().numpy())).astype(np.float64)
        K_a = np.array(K_)
        K_inv_a = np.array(K_inv)
        alpha = np.array(alpha_vec)
        out_2 = output_scale**2.
        len_2 = length_scale**2.
        theta_vec, theta_vec_2 = theta_vectors(x_gp, len_2)
        K_inv_scaled = scale_cKinv(K_a, out_2, noise)

        # Get bounds on the mean function
        # TODO, figure out why the julia call doesn't work in pool, still fast without it though
        mean_bound = bound_gp_from_nn_jl(x_gp, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                         theta_vec_2, theta_vec, dim, mean_bound, max_flag=False)
        mean_bound = bound_gp_from_nn_jl(x_gp, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                         theta_vec_2, theta_vec, dim, mean_bound, max_flag=True)

        # get upper bounds on variance
        sig_bound = bound_sig_from_nn_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                         theta_vec_2, theta_vec, K_inv_scaled, dim, sig_bound)
        # sig_bound = bound_sig_from_nn_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, linear_transform_info,
        #                                  theta_vec_2, theta_vec, K_inv_scaled, dim, sig_bound, min_flag=True)

    return [mean_bound, sig_bound, linear_transform_info]


def dkl_posts_fixed_nn(dkl_by_dim, x_data, y_data, mode, extents, nn_out_dim, crown_dir, experiment_dir, threads=8):
    # returns a list organized as [mean bounds, sig bounds, NN linear transform]
    # mean bounds is a list of tuples of tuples [((dim_upper, dim_lower) for each dim) for each region], same for sig
    # NN linear transform is a list of tuples of tuples [((lA, uA, l_bias, u_bias, bounds_min, bounds_max) for each dim) for each region]

    num_regions = len(extents) - 1
    num_dims = len(dkl_by_dim)
    n_obs = max(np.shape(y_data))

    linear_transform_info = [[[] for dim in range(num_dims)] for idx in range(num_regions)]
    mean_bound = [[[None, None] for dim in range(num_dims)] for idx in range(num_regions)]
    sig_bound = [[[None, None] for dim in range(num_dims)] for idx in range(num_regions)]

    # first get crown posts for each dim
    # dkl_by_dim is a list [[model, likelihood] for each dim]

    linear_transform_info = run_dkl_in_parallel(extents, mode, range(num_dims), nn_out_dim, crown_dir, experiment_dir,
                                                linear_transform_info, threads=threads)
    print('Finished bounding the NN portion')
    for dim in range(num_dims):

        # setup some intermediate values
        # pass x_train through the NN portion to get the kernel inputs
        dim_gp = dkl_by_dim[dim]
        model = dim_gp[0]
        model.cpu()  # unfortunately needs to be on cpu to access values
        nn_portion = model.feature_extractor
        with torch.no_grad():
            kernel_inputs = nn_portion.forward(x_data)

        noise = model.likelihood.noise.item()
        noise_mat = noise * np.identity(np.shape(kernel_inputs)[0])

        covar_module = model.covar_module
        kernel_mat = covar_module(kernel_inputs)
        kernel_mat = kernel_mat.evaluate()
        K_ = kernel_mat.detach().numpy() + noise_mat
        # enforce perfect symmetry, it is very close but causes errors when computing sig bounds
        K_ = (K_ + K_.transpose()) / 2.
        K_inv = np.linalg.inv(K_)  # only need to do this once per dim, yay

        y_dim = np.reshape(y_data[:, dim], (n_obs,))
        alpha_vec = K_inv @ y_dim  # TODO, store this for refinement?
        length_scale = model.covar_module.base_kernel.lengthscale.item()
        output_scale = model.covar_module.outputscale.item()

        # convert to julia input structure
        x_gp = np.array(np.transpose(kernel_inputs.detach().numpy())).astype(np.float64)
        K_a = np.array(K_)
        K_inv_a = np.array(K_inv)
        alpha = np.array(alpha_vec)
        out_2 = output_scale**2.
        len_2 = length_scale**2.
        theta_vec, theta_vec_2 = theta_vectors(x_gp, len_2)
        K_inv_scaled = scale_cKinv(K_a, out_2, noise)

        # Get bounds on the mean function
        # TODO, figure out why the julia call doesn't work in pool, still fast without it though
        mean_bound = bound_gp_from_nn_jl(x_gp, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                         theta_vec_2, theta_vec, dim, mean_bound, max_flag=False)
        mean_bound = bound_gp_from_nn_jl(x_gp, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                         theta_vec_2, theta_vec, dim, mean_bound, max_flag=True)

        # get upper bounds on variance
        sig_bound = bound_sig_from_nn_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                         theta_vec_2, theta_vec, K_inv_scaled, dim, sig_bound)
        # sig_bound = bound_sig_from_nn_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, linear_transform_info,
        #                                  theta_vec_2, theta_vec, K_inv_scaled, dim, sig_bound, min_flag=True)

    return [mean_bound, sig_bound, linear_transform_info]


def local_dkl_posts_fixed_nn(dkl_by_dim, mode, extents, region_info, nn_out_dim, crown_dir, experiment_dir, threads=8):
    # returns a list organized as [mean bounds, sig bounds, NN linear transform]
    # mean bounds is a list of tuples of tuples [((dim_upper, dim_lower) for each dim) for each region], same for sig
    # NN linear transform is a list of tuples of tuples [((lA, uA, l_bias, u_bias, bounds_min, bounds_max) for each dim) for each region]

    num_regions = len(extents) - 1
    num_sub_regions = len(region_info)
    num_dims = len(dkl_by_dim[0])

    linear_transform_info = [[[] for dim in range(num_dims)] for idx in range(num_regions)]
    mean_bound = [[[None, None] for dim in range(num_dims)] for idx in range(num_regions)]
    sig_bound = [[[None, None] for dim in range(num_dims)] for idx in range(num_regions)]

    # first get crown posts for each dim
    # dkl_by_dim is a list [[model, likelihood] for each dim]

    linear_transform_info = run_dkl_in_parallel(extents, mode, range(num_dims), nn_out_dim, crown_dir, experiment_dir,
                                                linear_transform_info, threads=threads)

    print('Finished bounding the NN portion')
    for sub_idx in range(num_sub_regions):
        region = region_info[sub_idx][2]
        x_data = torch.tensor(np.transpose(region_info[sub_idx][0]), dtype=torch.float32)
        y_data = np.transpose(region_info[sub_idx][1])
        n_obs = max(np.shape(y_data))

        # check bounds over sub_region, this will converge for sigma bounds almost surely
        rl = convert_to_list(region)
        os.chdir(crown_dir)
        res = run_dkl_crown_parallel(rl, crown_dir, nn_out_dim, mode, 0, num_regions)
        os.chdir(experiment_dir)
        x_L = np.array(res[4]).astype(np.float64)
        x_U = np.array(res[5]).astype(np.float64)

        # identify which extents are in this region and their indices
        specific_extents = extents_in_region(extents, region)

        for dim in range(num_dims):

            # setup some intermediate values
            # pass x_train through the NN portion to get the kernel inputs
            dim_gp = dkl_by_dim[sub_idx][dim]
            model = dim_gp[0]
            model.cpu()  # unfortunately needs to be on cpu to access values
            nn_portion = model.feature_extractor
            with torch.no_grad():
                kernel_inputs = nn_portion.forward(x_data)

            noise = model.likelihood.noise.item()
            noise_mat = noise * np.identity(np.shape(kernel_inputs)[0])

            covar_module = model.covar_module
            kernel_mat = covar_module(kernel_inputs)
            kernel_mat = kernel_mat.evaluate()
            K_ = kernel_mat.detach().numpy() + noise_mat
            # enforce perfect symmetry, it is very close but causes errors when computing sig bounds
            K_ = (K_ + K_.transpose()) / 2.
            K_inv = np.linalg.inv(K_)  # only need to do this once per dim, yay

            y_dim = np.reshape(y_data[:, dim], (n_obs,))
            alpha_vec = K_inv @ y_dim  # TODO, store this for refinement?
            length_scale = model.covar_module.base_kernel.lengthscale.item()
            output_scale = model.covar_module.outputscale.item()

            # convert to julia input structure
            x_gp = np.array(np.transpose(kernel_inputs.detach().numpy())).astype(np.float64)
            K_a = np.array(K_)
            K_inv_a = np.array(K_inv)
            alpha = np.array(alpha_vec)
            out_2 = output_scale**2.
            len_2 = length_scale**2.
            theta_vec, theta_vec_2 = theta_vectors(x_gp, len_2)
            K_inv_scaled = scale_cKinv(K_a, out_2, noise)

            # Get bounds on the mean function
            # TODO, figure out why the julia call doesn't work in pool, still pretty fast without it though
            mean_bound = bound_local_gp_from_nn_jl(x_gp, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                                   theta_vec_2, theta_vec, dim, mean_bound, specific_extents,
                                                   max_flag=False)
            mean_bound = bound_local_gp_from_nn_jl(x_gp, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                                   theta_vec_2, theta_vec, dim, mean_bound, specific_extents,
                                                   max_flag=True)

            # get bounds on variance
            worst_case_sig = compute_sig_bounds_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, x_L, x_U, theta_vec_2,
                                                   theta_vec, K_inv_scaled, max_iterations=500)
            worst_sig = worst_case_sig[2]
            print(f"Worst case sigma for sub-region {sub_idx+1} and dim {dim} is {worst_sig}")
            sig_bound = bound_local_sig_from_nn_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, linear_transform_info,
                                                   theta_vec_2, theta_vec, K_inv_scaled, dim, sig_bound,
                                                   specific_extents, worst_case=worst_sig)

            # sig_bound = bound_local_sig_from_nn_jl(x_gp, K_a, K_inv_a, alpha, out_2, len_2, linear_transform_info,
            #                                        theta_vec_2, theta_vec, K_inv_scaled, dim, sig_bound,
            #                                        specific_extents, min_flag=True)
            # print('Bounded min sig')

        print(f'Finished sub-region {sub_idx+1} of {num_sub_regions}')

    return [mean_bound, sig_bound, linear_transform_info]


def run_dkl_in_parallel(extents, mode, dim, nn_out_dim, crown_dir, experiment_dir, linear_transform_info, 
                        threads=8):

    # set up region_eps
    extent_len = len(extents) - 1

    os.chdir(crown_dir)

    if type(dim) is range:
        dim_ = 0
    else:
        dim_ = dim

    pool = mp.Pool(threads)
    results = pool.starmap(run_dkl_crown_parallel, [(extents[idx], crown_dir, nn_out_dim, mode, dim_,
                           idx) for idx in range(extent_len)])
    pool.close()

    os.chdir(experiment_dir)
    if type(dim) is range:
        for dims in dim:
            for index, lin_trans in enumerate(results):
                linear_transform_info[index][dims] = lin_trans
    else:
        for index, lin_trans in enumerate(results):
            linear_transform_info[index][dim] = lin_trans

    return linear_transform_info

############################################################################################
#  Julia Call portions
############################################################################################


def bound_gp_from_nn_jl(x_gp, K_inv, alpha, output_scale, length_scale, linear_transform, theta_vec_2, theta_vec, dim,
                        mean_bound, max_flag=False):

    num_regions = np.shape(linear_transform)[0]

    if max_flag is False:
        # lower bound on mean
        # TODO, figure out how to parallelize
        for idx in range(num_regions):
            mean_info = compute_mean_bounds_jl(x_gp, K_inv, alpha, output_scale, length_scale,
                                               np.array(linear_transform[idx][dim][4]).astype(np.float64),
                                               np.array(linear_transform[idx][dim][5]).astype(np.float64),
                                               theta_vec_2, theta_vec)
            mean_bound[idx][dim][0] = mean_info[1]
            if abs(mean_info[1] - mean_info[2]) > 0.1:
                print(f'Upper mean info is {mean_info} at region {idx} and dim {dim}')

    else:
        # upper bound on mean
        for idx in range(num_regions):
            mean_info = compute_mean_bounds_jl(x_gp, K_inv, -alpha, output_scale, length_scale,
                                               np.array(linear_transform[idx][dim][4]).astype(np.float64),
                                               np.array(linear_transform[idx][dim][5]).astype(np.float64),
                                               theta_vec_2, theta_vec)
            mean_bound[idx][dim][1] = -mean_info[1]
            if abs(mean_info[1] - mean_info[2]) > 0.1:
                print(f'Lower mean info is {mean_info} at region {idx} and dim {dim}')

    return mean_bound


def bound_sig_from_nn_jl(x_gp, K_, K_inv, alpha, output_scale, length_scale, linear_transform, theta_vec_2, theta_vec,
                         K_inv_scaled, dim, sig_bound, min_flag=False):

    num_regions = np.shape(linear_transform)[0]
    mi = 500
    if min_flag:
        mi = 10

    # upper bound on sigma
    # TODO, figure out how to parallelize
    for idx in range(num_regions):
        sig_info = compute_sig_bounds_jl(x_gp, K_, K_inv, alpha, output_scale, length_scale,
                                         np.array(linear_transform[idx][dim][4]).astype(np.float64),
                                         np.array(linear_transform[idx][dim][5]).astype(np.float64),
                                         theta_vec_2, theta_vec, K_inv_scaled, max_iterations=mi, min_flag=min_flag)
        sig_ = sig_info[2]  # this is a std deviation
        if min_flag:
            sig_ = sig_info[1]

        if sig_ > .4 and not min_flag:
            print(f"Region {idx} has a sigma info {sig_info} in dim {dim}")
            sig_ = min(max(sig_info[1], 0.4), sig_info[2])

        if min_flag:
            sig_bound[idx][dim][0] = sig_
        else:
            sig_bound[idx][dim][1] = sig_

    return sig_bound


def bound_local_gp_from_nn_jl(x_gp, K_inv, alpha, output_scale, length_scale, linear_transform, theta_vec_2, theta_vec,
                              dim, mean_bound, specific_extents, max_flag=False):

    if max_flag is False:
        # lower bound on mean
        # TODO, figure out how to parallelize
        for idx in specific_extents:
            mean_info = compute_mean_bounds_jl(x_gp, K_inv, alpha, output_scale, length_scale,
                                               np.array(linear_transform[idx][dim][4]).astype(np.float64),
                                               np.array(linear_transform[idx][dim][5]).astype(np.float64),
                                               theta_vec_2, theta_vec)
            mean_bound[idx][dim][0] = mean_info[1]

    else:
        # upper bound on mean
        for idx in specific_extents:
            mean_info = compute_mean_bounds_jl(x_gp, K_inv, -alpha, output_scale, length_scale,
                                               np.array(linear_transform[idx][dim][4]).astype(np.float64),
                                               np.array(linear_transform[idx][dim][5]).astype(np.float64),
                                               theta_vec_2, theta_vec)
            mean_bound[idx][dim][1] = -mean_info[1]

    return mean_bound


def bound_local_sig_from_nn_jl(x_gp, K_, K_inv, alpha, output_scale, length_scale, linear_transform, theta_vec_2,
                               theta_vec, K_inv_scaled, dim, sig_bound, specific_extents, min_flag=False, worst_case=0.3):

    # upper bound on sigma
    mi = 500
    if min_flag:
        mi = 5

    for idx in specific_extents:
        x_L = np.array(linear_transform[idx][dim][4]).astype(np.float64)
        x_U = np.array(linear_transform[idx][dim][5]).astype(np.float64)

        sig_info = compute_sig_bounds_jl(x_gp, K_, K_inv, alpha, output_scale, length_scale, x_L, x_U, theta_vec_2,
                                         theta_vec, K_inv_scaled, max_iterations=mi, min_flag=min_flag)
        sig_ = sig_info[2]  # this is a std deviation
        if sig_ > worst_case:
            print(f"Region {idx} has sig bounds {[sig_info[1], sig_info[2]]} in dim {dim}")
            sig_ = worst_case

        if min_flag:
            sig_bound[idx][dim][0] = sig_
        else:
            sig_bound[idx][dim][1] = sig_

    return sig_bound

############################################################################################
#  Support functions
############################################################################################


def extents_in_region(extents, region):
    # returns a list of indices of extents that are inside the region
    keys = list(extents[0])
    valid = []
    for extent_idx, extent in enumerate(extents):
        in_region = True
        for idx, k in enumerate(keys):
            if extent[k][0] < region[idx][0] or extent[k][1] > region[idx][1]:
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

