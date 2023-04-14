using Distributed

args = ARGS
addprocs(parse(Int, args[1]))

@everywhere using PosteriorBounds
using PyCall
using SharedArrays
@everywhere include("quad_prog_bnb.jl")
@pyimport numpy

EXPERIMENT_DIR = @__DIR__
experiment_type = "/deep_kernel_synthesis"
exp_dir = EXPERIMENT_DIR * experiment_type

experiment_number = parse(Int, args[2])

if experiment_number == 3
    global_dir_name = "sys_3d"
elseif experiment_number == 3.5
    global_dir_name = "sys_3d_gp"
end

global_exp_dir = exp_dir * "/" * global_dir_name
general_data = numpy.load(global_exp_dir * "/general_info.npy")
nn_bounds_dir = global_exp_dir * "/nn_bounds"

num_modes = general_data[1]
num_dims = general_data[2]
num_sub_regions = general_data[3]
num_regions = general_data[4]

# =====================================================================================
# 3. Use NN posteriors as input spaces for the GP and get bounds on that
# =====================================================================================

# once the new method is in place, test mode 1, sub_region 1, dim 1, index 233+1

mean_bound = SharedArray{Float64}(num_modes, num_regions, num_dims, 2)
sig_bound = SharedArray{Float64}(num_modes, num_regions, num_dims, 2)
for mode in 1:num_modes
    linear_bounds = numpy.load(nn_bounds_dir*"/linear_bounds_$mode.npy")

    mode_runtime = @elapsed begin
        for sub_idx in 1:(num_sub_regions::Int)
            for dim in 1:(num_dims::Int)
                # load all the data for this mode
                dim_region_filename = nn_bounds_dir * "/linear_bounds_$mode" * "_$sub_idx" * "_$dim"

                specific_extents = numpy.load(dim_region_filename*"_these_indices.npy")  # need to add 1 to this
                x_gp = numpy.load(dim_region_filename*"_x_gp.npy")
                theta_vec = numpy.load(dim_region_filename*"_theta_vec.npy")
                theta_vec_2 = numpy.load(dim_region_filename*"_theta_vec_2.npy")
                K = numpy.load(dim_region_filename*"_K.npy")
                K_inv = numpy.load(dim_region_filename*"_K_inv.npy")
                alpha = numpy.load(dim_region_filename*"_alpha.npy")
                K_inv_scaled = numpy.load(dim_region_filename*"_K_inv_s.npy")
                kernel_info = numpy.load(dim_region_filename*"_kernel.npy")
                out2 = kernel_info[1]
                l2 = kernel_info[2]

                noise = K[1,1] - out2
                cK_inv_scaled = PosteriorBounds.scale_cK_inv(K, out2, noise)

                # parallelize getting mean bounds and variance bounds
                @sync @distributed for idx in specific_extents
                    x_L = linear_bounds[idx+1, dim, 1, :]
                    x_U = linear_bounds[idx+1, dim, 2, :]

                    # get lower mean bounds
                    mean_info_l = PosteriorBounds.compute_μ_bounds_bnb_tmp(x_gp, K_inv, alpha, out2, l2,
                                                                           x_L, x_U, theta_vec_2, theta_vec)

                    # get upper mean bounds, negating alpha is the same as
                    mean_info_u = PosteriorBounds.compute_μ_bounds_bnb_tmp(x_gp, K_inv, -alpha, out2, l2,
                                                                           x_L, x_U, theta_vec_2, theta_vec)

                    mean_bound[mode, idx+1, dim, 1] = mean_info_l[2]
                    mean_bound[mode, idx+1, dim, 2] = -mean_info_u[2]

                    # get upper bounds on variance
                    sig_info = PosteriorBounds.compute_σ_bounds_bnb_tmp(x_gp, K, K_inv, alpha, out2, l2, x_L, x_U,
                                                                        theta_vec_2, theta_vec, cK_inv_scaled;
                                                                        max_iterations=20, bound_epsilon=1e-3)

                    sig_ = sqrt(sig_info[3])  # this is a std deviation
                    sig_low = sqrt(sig_info[2])
                    if abs(sig_-sig_low) > sqrt(1e-3)
                        @info "Fast method didn't converge, using the quadratic program (mode $mode, sub_region $sub_idx, dim $dim, index $idx)"
                        # this means it didn't converge properly, use expensive quadratic program to find solution
                        m = size(x_gp, 2) # n_obs
                        n = size(x_gp, 1) # dims
                        gp = PosteriorBounds.PosteriorGP(
                                n, m, x_gp, K, Matrix{Float64}(undef, m, m),
                                PosteriorBounds.UpperTriangular(zeros(m, m)),
                                K_inv, alpha, PosteriorBounds.SEKernel(out2, l2)
                            )
                        PosteriorBounds.compute_factors!(gp)
                        outputs = sigma_bnb(gp, x_gp, m, n, out2, x_L, x_U, theta_vec, K_inv_scaled;
                                            max_iterations=20, bound_epsilon=1e-2)

                        sig_ = outputs[3]
                    end

                    sig_bound[mode, idx+1, dim, 2] = sig_
                    sig_bound[mode, idx+1, dim, 1] = 0.7*sig_

                end
            end
        end
    end
    @info "Finished getting bounds for mode $mode in $mode_runtime seconds"
end

# save all the data for use in IMDP generation next
numpy.save(global_exp_dir*"/mean_data", mean_bound)
numpy.save(global_exp_dir*"/sig_data", sig_bound)