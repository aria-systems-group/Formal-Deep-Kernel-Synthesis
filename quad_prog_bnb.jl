using PosteriorBounds
using JuMP, Ipopt


function sigma_bnb(gp, x_train, m, n, sigma_prior, x_L, x_U, theta_vec, cK_inv_scaled;
                   max_iterations=10, bound_epsilon=1e-2, min_flag=false)

    z_i_vec = zeros(Float64, m, 2)
    dx_L = zeros(Float64, n)
    dx_U = zeros(Float64, n)
    B = zeros(Float64, 2*m, n+m)
    r_L = zeros(Float64, m)
    r_U = zeros(Float64, m)
    a_vec = zeros(Float64, 2*m)
    x_star = zeros(Float64, n)
    sig_post = zeros(Float64, 1,1)

    x_best = nothing
    lbest = -Inf
    ubest = Inf

    minmax_factor = min_flag ? -1 : 1
    candidates = [(x_L, x_U, minmax_factor*Inf)]
    iterations = 0

    Q = [zeros(Float64, n, m+n); zeros(Float64, m, n) cK_inv_scaled]
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "sb"=>"yes"))
    set_silent(model)
    @variable(model, x[i=1:size(Q,1)])
    @objective(model, Min, x'*Q*x)

    split_regions = nothing
    x_avg = zeros(gp.dim)
    while !isempty(candidates) && iterations < max_iterations
        new_candidates = []
        for extent in candidates

            # Skip candidates according to current best bound
            if min_flag
                if extent[3] > lbest
                    continue
                end
            else
                if extent[3] < lbest
                    continue
                end
            end

            if isnothing(split_regions)
                split_regions = PosteriorBounds.split_region!(extent[1], extent[2], x_avg)
            else
                split_regions = PosteriorBounds.split_region!(extent[1], extent[2], x_avg, new_regions=split_regions)
            end

            for pair in split_regions

                x_ub1, ub1 = compute_bounds_sqe_quad_prog(model, x, x_train, m, n, sigma_prior, pair[1], pair[2], theta_vec,
                                                         cK_inv_scaled, B, dx_L, dx_U, z_i_vec, r_L, r_U, a_vec)
                x_star[:] = x_ub1[:]
                lb1 = PosteriorBounds.compute_σ2!(sig_post, gp, hcat(x_star))
                lb1 = sqrt(lb1)

                if lb1 >= lbest
                    lbest = lb1
                    ubest = ub1
                    x_best = x_ub1
                    push!(new_candidates, (pair[1], pair[2], lbest))
                elseif ub1 > lbest
                    push!(new_candidates, (pair[1], pair[2], lb1))
                end
            end
        end

        if PosteriorBounds.norm(ubest - lbest) < bound_epsilon
            break
        end
        candidates = new_candidates
        iterations += 1
    end

    return x_best, lbest, ubest
end


function compute_bounds_sqe_quad_prog(model, var, x_train, m, n, sigma_prior, x_L, x_U, theta_vec, cK_inv_scaled,
                                      B::Matrix{Float64}, dx_L::Vector{Float64}, dx_U::Vector{Float64},
                                      z_i_vector::Matrix{Float64}, r_L::Vector{Float64}, r_U::Vector{Float64},
                                      a_vec::Vector{Float64})

    for idx=1:m
        z_i_vector[idx, :] .= PosteriorBounds.compute_z_intervals(x_train[:, idx], x_L, x_U, theta_vec, n, dx_L, dx_U)
    end

    for ii=1:m  # n_obs
        r_L[ii] = exp(-z_i_vector[ii,2])
        r_U[ii] = exp(-z_i_vector[ii,1])
        a_i_L, b_i_L, a_i_U, b_i_U = concave_bounds(r_L[ii], r_U[ii])

        a_ij_L_sum = 0.
        a_ij_U_sum = 0.

        for jj = 1:n  # n_dims
            x_t = x_train[jj, ii]
            v_t = theta_vec[jj]
            if x_L[jj] < x_U[jj]
                a_ij_L, b_ij_L, a_ij_U, b_ij_U = convex_bounds(x_L[jj],x_U[jj], x_t, v_t)
            else
                a_ij_L = v_t * (x_L[jj] - x_t)^2.
                a_ij_U = a_ij_L
                b_ij_L = 0.
                b_ij_U = 0.
            end

            B[(2*ii)-1, jj] = b_ij_L
            B[(2*ii), jj] = -b_ij_U
            a_ij_L_sum += a_ij_L
            a_ij_U_sum += a_ij_U
        end

        a_vec[(2*ii)-1] = a_i_L + a_ij_L_sum
        a_vec[(2*ii)] = -a_i_U - a_ij_U_sum
        B[(2*ii)-1, ii+n] = b_i_L
        B[(2*ii), ii+n] = -b_i_U
    end

    # get upper bound on variance
    x_opt, ub, exitFlag = quadprog(model, var; A=B, b=-a_vec, lb=[x_L;r_L], ub=[x_U;r_U])

    sigma_u = sigma_prior*(1 - ub)
    x_sigma = x_opt[1:n]
    return x_sigma, sqrt(sigma_u)
end


function concave_bounds(x_l, x_u)
    # in the concave part of the function, the lower bound is given by
    # the line that connects the two function extremes
    f_l = log(x_l)
    f_u = log(x_u)
    a_lower, b_lower = line_through_points(x_l, f_l, x_u, f_u)

    # the upper bound is the tangent line through the mid point
    x_c = (x_l + x_u) / 2.
    df_c = 1. / x_c
    f_c = log(x_c)

    a_upper, b_upper = line_through_points_ang_coeff(x_c, f_c, df_c)

    return a_lower, b_lower, a_upper, b_upper
end


function line_through_points(x_l, f_l, x_u, f_u)
    b = (f_l - f_u) / (x_l - x_u)
    a = f_l - b*x_l
    return a, b
end


function line_through_points_ang_coeff(x_c, f_c, df_c)
    b = df_c
    a = f_c - b*x_c
    return a, b
end


function convex_bounds(x_l, x_u, x_t, v_t)

    x_c = (x_l + x_u) / 2.
    f_c = v_t * (x_c - x_t)^2.
    df_c = 2. * v_t * (x_c - x_t)
    a_lower, b_lower = line_through_points_ang_coeff(x_c, f_c, df_c)

    f_l = v_t * (x_l - x_t)^2.
    f_u = v_t * (x_u - x_t)^2.
    a_upper, b_upper = line_through_points(x_l, f_l, x_u, f_u)

    return a_lower, b_lower, a_upper, b_upper
end


function quadprog(model, x;
                       A   =  zeros(Float64, (0, 0)),
                       b   =  zeros(Float64, 0),
                       lb  = -Inf*ones(0),
                       ub  =  Inf*ones(0))

    # DESCRIPTION:
    # min  x' * Q * x
    # s.t. A   * x <= b
    #      lb <= x <= ub

    # remove previous constraints and add new ones
    c = all_constraints(model; include_variable_in_set_constraints=true)
    for c_ in c
        delete(model, c_)
    end
    @constraint(model, A*x <= b)
    @constraint(model, lb .<= x .<= ub)

    optimize!(model)
    x_opt = JuMP.value.(x)
    return x_opt, objective_value(model), termination_status(model)
end
