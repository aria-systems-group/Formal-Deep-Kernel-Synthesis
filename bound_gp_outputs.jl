# using Distributed
using SparseArrays

@everywhere using PosteriorBounds
using PyCall
using SharedArrays
@everywhere include("quad_prog_bnb.jl")
@pyimport numpy

@everywhere function matching_label(test_label, compare_label)
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


function bound_gp(num_regions, num_modes, num_dims, refinement, global_exp_dir, reuse_regions, label_fn, skip_labels,
                  use_personal)
    nn_bounds_dir = global_exp_dir * "/nn_bounds"
    gp_bounds_dir = global_exp_dir  # * "/gp_bounds"

    if !isdir(gp_bounds_dir)
        mkdir(gp_bounds_dir)
    end
    for mode in 1:num_modes
        if isfile(gp_bounds_dir*"/complete_$mode" * "_$refinement.npy") && reuse_regions
            @info "Reusing prior bounds for mode $mode"
            continue
        end

        if refinement > 0
            mean_bound = numpy.load(global_exp_dir*"/mean_data_$mode" * "_$refinement.npy")
            mean_bound = convert(SharedArray, mean_bound)
            sig_bound = numpy.load(global_exp_dir*"/sig_data_$mode" * "_$refinement.npy")
            sig_bound = convert(SharedArray, sig_bound)
        else
            mean_bound = SharedArray(zeros(num_regions, num_dims, 2))
            sig_bound = SharedArray(zeros(num_regions, num_dims, 2))
        end

        linear_bounds = numpy.load(nn_bounds_dir*"/linear_bounds_$(mode)_$(refinement).npy")
        convert(SharedArray, linear_bounds)

        mode_runtime = @elapsed begin
            for dim in 1:(num_dims::Int)
                dim_sig = dyn_noise[dim]
                # load all the data for this mode
                dim_region_filename = nn_bounds_dir * "/linear_bounds_$(mode)_1_$dim"

                specific_extents = numpy.load(dim_region_filename*"_these_indices_$refinement.npy")  # need to add 1 to this
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

                m = size(x_gp, 2) # n_obs
                n = size(x_gp, 1) # dims
                gp_neg = PosteriorBounds.PosteriorGP(n, m, x_gp, K, Matrix{Float64}(undef, m, m),
                        PosteriorBounds.UpperTriangular(zeros(m,m)), K_inv, -alpha,
                        PosteriorBounds.SEKernel(out2, l2))
                gp = PosteriorBounds.PosteriorGP(n, m, x_gp, K, Matrix{Float64}(undef, m, m),
                        PosteriorBounds.UpperTriangular(zeros(m,m)), K_inv, alpha,
                        PosteriorBounds.SEKernel(out2, l2))
                PosteriorBounds.compute_factors!(gp)

                # parallelize getting mean bounds and variance bounds
                @sync @distributed for idx in specific_extents
                    # need distributed here because it is significant computation, threading is inefficient
                    if use_personal == 1
                        x_L = linear_bounds[idx+1, 1, dim]
                        x_U = linear_bounds[idx+1, 2, dim]
                    else
                        x_L = linear_bounds[idx+1, 1, :]
                        x_U = linear_bounds[idx+1, 2, :]
                    end

                    if any([matching_label(label_fn[idx+1], skip_) for skip_ in skip_labels])
                        # don't actually care about the bounds on this region, just put 1s and 0s
                        mean_bound[idx+1, dim, 1] = 0.0
                        mean_bound[idx+1, dim, 2] = 1.0
                        sig_bound[idx+1, dim, 1] = 0.0
                        sig_bound[idx+1, dim, 2] = 1.0
                    else

                        # get lower mean bounds
                        mean_info_l = PosteriorBounds.compute_μ_bounds_bnb(gp, x_L, x_U, theta_vec_2,
                                                                           theta_vec; max_iterations=100,
                                                                           bound_epsilon=1e-3, max_flag=false,
                                                                           prealloc=nothing)

                        # get upper mean bounds, negating alpha allows for the "min" to be the -max
                        mean_info_u = PosteriorBounds.compute_μ_bounds_bnb(gp_neg, x_L, x_U, theta_vec_2,
                                                                           theta_vec; max_iterations=100,
                                                                           bound_epsilon=1e-3, max_flag=false,
                                                                           prealloc=nothing)

                        if mean_info_l[2] == mean_info_u[2]
                            @info "something is wrong with $idx, $mode, $dim. Mean is identical"
                        end

                        mean_bound[idx+1, dim, 1] = mean_info_l[2]
                        mean_bound[idx+1, dim, 2] = -mean_info_u[2]

                        # get upper bounds on variance
                        sig_check = sig_bound[idx+1, dim, 2]
                        if refinement > 0 && sig_check <= 0*sqrt(dim_sig)
                            # if previous sigma bounds were already small, don't waste time finding better ones
                            sig_upper = sig_check
                            sig_lower = sig_bound[idx+1, dim, 1]
                        else
                            sig_info = PosteriorBounds.compute_σ_bounds(gp, x_L, x_U, theta_vec_2, theta_vec,
                                                                        cK_inv_scaled; max_iterations=20,
                                                                        bound_epsilon=1e-3, min_flag=false,
                                                                        prealloc=nothing)

                            sig_upper = sqrt(sig_info[2])  # this is a std deviation
                            sig_low = sqrt(sig_info[3])
                            if abs(sig_upper-sig_low) > sqrt(1e-3)
                                # this means it didn't converge properly, use expensive quadratic program to find solution
                                outputs = sigma_bnb(gp, x_gp, m, n, out2, x_L, x_U, theta_vec, K_inv_scaled;
                                                    max_iterations=20, bound_epsilon=sqrt(1e-3))

                                sig_upper = outputs[3]  # this is a std deviation
                            end
                            # TODO, figure out why this takes forever
                            # sig_info = PosteriorBounds.compute_σ_bounds(gp, x_L, x_U, theta_vec_2, theta_vec,
                            #                                             cK_inv_scaled; max_iterations=10,
                            #                                             bound_epsilon=1e-2, min_flag=true,
                            #                                             prealloc=nothing)
                            sig_lower = min(sig_low, sig_upper)
                        end

                        sig_bound[idx+1, dim, 1] = sig_lower
                        sig_bound[idx+1, dim, 2] = sig_upper
                    end
                end
            end
        end
        @info "Calculated bounds for mode $mode in $mode_runtime seconds"
        # save data
        numpy.save(gp_bounds_dir*"/mean_data_$mode" * "_$refinement", mean_bound)
        numpy.save(gp_bounds_dir*"/sig_data_$mode" * "_$refinement", sig_bound)
        numpy.save(gp_bounds_dir*"/complete_$mode" * "_$refinement", 1)
    end

end
