
from import_script import *
from generic_fnc import DynModelNetRelu3, DynModelNetRelu2, DynModelNetRelu1, DynModelNetTanh1, DynModelNetTanh3
from crown_scripts import setup_crown_yaml_dkl, run_dkl_crown_parallel
from space_discretize_scripts import separate_data_into_regions_new, discretize_space_list
import multiprocessing as mp


# =============================================================================================================
#  GP classes
# =============================================================================================================

def loss_function_smooth(unknown_dyn_model, y_data, x_data):
    y_predict = unknown_dyn_model(x_data)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(y_predict, y_data)

    # temp = torch.autograd.grad(y_predict, x_data, grad_outputs=torch.ones_like(y_predict), retain_graph=True)
    # grad_loss = torch.div(torch.sum(torch.abs(temp[0])), len(x_data)*3)
    #
    # loss += grad_loss
    return loss


def train_feature_extractor(all_data, mode, use_relu, num_layers, network_dims, random_seed=20,
                            epochs=10000, lr=1e-3, use_scaling=False):

    if use_relu:
        if num_layers == 3:
            net_model_to_use = DynModelNetRelu3
        elif num_layers == 2:
            net_model_to_use = DynModelNetRelu2
        else:
            net_model_to_use = DynModelNetRelu1
    else:
        if num_layers == 1:
            net_model_to_use = DynModelNetTanh1
        else:
            net_model_to_use = DynModelNetTanh3

    d = network_dims[0]
    width_1 = network_dims[1]
    width_2 = network_dims[2]
    out_dim = network_dims[3]

    x_train = all_data[mode][0]
    y_train = all_data[mode][1]
    n_samples = int(np.shape(x_train)[1]/15.)

    scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(random_seed)
    global feature_extractor
    feature_extractor = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)
    feature_extractor.to(device)
    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
    y_data = torch.tensor(np.transpose(y_train), dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr)
    print("Training Neural Network on generated data for mode {}...".format(mode+1))
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        # get minibatch data, ~ 15% of the data
        perm = torch.randperm(y_data.size(0))
        idx = perm[:n_samples]
        y_mini = y_data[idx]
        x_mini = x_data[idx]

        if use_scaling:
            if out_dim == 5:
                # y_mini = scaling_fnc(x_mini)
                # y_mini = scaling_fnc(y_mini, dx=10., dy=10., dz=4.)
                y_mini = scaling_fnc(y_mini, dx=4., dy=4.)
            if out_dim == 3:
                y_mini = scaling_fnc(y_mini, dx=10., dy=2.)
            else:
                y_mini = scale_to_bounds(y_mini)

        # Compute and print loss.
        loss = loss_function_smooth(feature_extractor, y_mini, x_mini)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    feature_extractor.to(torch.device('cpu'))


def scaling_fnc(y_in, dx=4., dy=4., dz=1.):

    M, N = np.shape(y_in)
    # scale each dimension, rather than the whole input
    y_out = torch.clone(y_in)
    idx = 0
    for n in range(N):
        if idx == 0:
            scale = dx
        elif idx == 1:
            scale = dy
        else:
            scale = dz
        y_out[:, n] = y_in[:, n]/scale

    return y_out


def set_feature_extractor(use_relu, num_layers, network_dims, random_seed=20):
    # define neural network layers
    if use_relu:
        if num_layers == 3:
            net_model_to_use = DynModelNetRelu3
        elif num_layers == 2:
            net_model_to_use = DynModelNetRelu2
        else:
            net_model_to_use = DynModelNetRelu1
    else:
        if num_layers == 1:
            net_model_to_use = DynModelNetTanh1
        else:
            net_model_to_use = DynModelNetTanh3

    d = network_dims[0]
    width_1 = network_dims[1]
    width_2 = network_dims[2]
    out_dim = network_dims[3]

    global feature_extractor
    torch.manual_seed(random_seed)
    feature_extractor = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)


def reset_feature_extractor(fe):
    global feature_extractor
    feature_extractor = fe


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        lengthscale_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(DeepKernelGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        lengthscale_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# =============================================================================================================
# Learning functions
# =============================================================================================================


# def deep_kernel_fixed_nn_local(all_data, mode, keys, use_reLU, num_layers, network_dims, crown_dir, global_dir_name,
#                                grid_size, domain, training_iter=40, lr=0.01, process_noise=None, random_seed=11,
#                                epochs=10000, nn_lr=1e-3, use_regular_gp=False, use_scaling=False):
#     # this function trains either a standard se_kernel GP (if use_regular_gp=True) or a deep kernel model
#     # The deep kernel model will use a pre-trained fixed NN while optimizing the kernel parameters
#     # The models are trained on local data sets to reduce computational load in the future
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     alpha = 1e-2
#     if process_noise is not None:
#         alpha = process_noise["sig"]
#
#     if not use_regular_gp:
#         # train the NN with all the data
#         train_feature_extractor(all_data, mode, use_reLU, num_layers, network_dims, random_seed=random_seed,
#                                 epochs=epochs, lr=nn_lr, use_scaling=use_scaling)
#
#     print('Optimizing GP parameters\n')
#
#     domain_list = discretize_space_list(domain, grid_size, include_space=False)
#     region_data = [None, None, domain_list]
#     x_train = all_data[mode][0]
#     y_train = all_data[mode][1]
#     region_data[0] = x_train
#     region_data[1] = y_train
#
#     y_train = np.transpose(y_train)
#     x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
#     n_obs = np.shape(y_train)[0]
#
#     gp_by_dim = []
#     for dim, key in enumerate(keys):
#
#         y_data = torch.tensor(np.reshape(y_train[:, dim], (n_obs,)), dtype=torch.float32).to(device)
#
#         likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
#         if use_regular_gp:
#             model = ExactGPModel(x_data, y_data, likelihood)
#         else:
#             model = DeepKernelGP(x_data, y_data, likelihood)
#
#         if torch.cuda.is_available():
#             model = model.cuda()
#             likelihood = likelihood.cuda()
#
#         if process_noise["dist"] == "multi_norm":
#             alpha_ = alpha[dim]
#             if "theta_dim" in list(process_noise):
#                 if dim in process_noise["theta_dim"]:
#                     # for some reason there are a lot of errors if the noise is seeded low for theta dynamics
#                     if len(domain) == 3:
#                         alpha_ = max(alpha[dim], 0.0075)
#                         alpha_ = max(alpha[dim], 0.01)
#                     else:
#                         alpha_ = max(alpha[dim], 0.005)
#                         alpha_ = max(alpha[dim], 0.01)
#         else:
#             alpha_ = alpha
#         hypers = {'likelihood.noise_covar.noise': torch.tensor(alpha_), }
#
#         model.initialize(**hypers)
#         model.train()
#         likelihood.train()
#
#         # NOTE, this does not optimize the NN feature extractor, uses the pre-trained version
#         optimizer = torch.optim.Adam([{'params': model.covar_module.parameters()},
#                                       {'params': model.mean_module.parameters()},
#                                       {'params': model.likelihood.parameters()},
#                                       ], lr=lr)
#
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#         for i in range(training_iter):
#             # Zero gradients from previous iteration
#             optimizer.zero_grad()
#             # Output from model
#             output = model(x_data)
#
#             # Calc loss and backprop gradients
#             loss = -mll(output, y_data)
#
#             loss.backward()
#             if i % training_iter == training_iter-1:
#                 print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' %
#                       (i + 1, training_iter, loss.item(),
#                        model.covar_module.base_kernel.lengthscale.item()))
#
#             optimizer.step()
#
#         if not use_regular_gp and global_dir_name is not None:
#             setup_crown_yaml_dkl(network_dims, mode, dim, crown_dir, global_dir_name, use_reLU, num_layers)
#             torch_file_name = "/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)
#             nn_to_save = model.feature_extractor
#             torch.save(nn_to_save.state_dict(), crown_dir + "/models" + torch_file_name)
#
#         model.eval()
#         likelihood.eval()
#         print("")
#         gp_by_dim.append([model, likelihood])
#
#     return gp_by_dim, region_data


def deep_kernel_fixed_nn_local(all_data, mode, keys, use_reLU, num_layers, network_dims, crown_dir, global_dir_name,
                               grid_size, domain, training_iter=40, lr=0.01, process_noise=None, random_seed=11,
                               epochs=10000, nn_lr=1e-3, use_regular_gp=False, use_scaling=False):
    # this function trains either a standard se_kernel GP (if use_regular_gp=True) or a deep kernel model
    # The deep kernel model will use a pre-trained fixed NN while optimizing the kernel parameters
    # The models are trained on local data sets to reduce computational load in the future

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = 1e-2
    if process_noise is not None:
        alpha = process_noise["sig"]

    if not use_regular_gp:
        # train the NN with all the data
        train_feature_extractor(all_data, mode, use_reLU, num_layers, network_dims, random_seed=random_seed,
                                epochs=epochs, lr=nn_lr, use_scaling=use_scaling)

    # Splitting data into desired regions for localization
    local_regions = discretize_space_list(domain, grid_size, include_space=False)
    data_by_region = separate_data_into_regions_new(local_regions, all_data[mode])

    print('Optimizing GP parameters\n')
    gp_by_region = []
    region_info = []
    for idx, region in enumerate(local_regions):
        region_data = [None, None, region]
        x_train = data_by_region[idx][0]
        y_train = data_by_region[idx][1]
        region_data[0] = x_train
        region_data[1] = y_train

        y_train = np.transpose(y_train)
        x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
        n_obs = np.shape(y_train)[0]

        gp_by_dim = []
        for dim, key in enumerate(keys):

            y_data = torch.tensor(np.reshape(y_train[:, dim], (n_obs,)), dtype=torch.float32).to(device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            if use_regular_gp:
                model = ExactGPModel(x_data, y_data, likelihood)
            else:
                model = DeepKernelGP(x_data, y_data, likelihood)

            if torch.cuda.is_available():
                model = model.cuda()
                likelihood = likelihood.cuda()

            if process_noise["dist"] == "multi_norm":
                alpha_ = alpha[dim]
                if "theta_dim" in list(process_noise):
                    if dim in process_noise["theta_dim"]:
                        # for some reason there are a lot of errors if the noise is seeded low for theta dynamics
                        if len(domain) == 3:
                            alpha_ = max(alpha[dim], 0.0075)
                            alpha_ = max(alpha[dim], 0.01)
                        else:
                            alpha_ = max(alpha[dim], 0.005)
                            alpha_ = max(alpha[dim], 0.01)
            else:
                alpha_ = alpha
            hypers = {'likelihood.noise_covar.noise': torch.tensor(alpha_), }

            model.initialize(**hypers)
            model.train()
            likelihood.train()

            # NOTE, this does not optimize the NN feature extractor, uses the pre-trained version
            optimizer = torch.optim.Adam([{'params': model.covar_module.parameters()},
                                          {'params': model.mean_module.parameters()},
                                          {'params': model.likelihood.parameters()},
                                          ], lr=lr)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(x_data)

                # Calc loss and backprop gradients
                loss = -mll(output, y_data)

                loss.backward()
                if i % training_iter == training_iter-1:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' %
                          (i + 1, training_iter, loss.item(),
                           model.covar_module.base_kernel.lengthscale.item()))

                optimizer.step()

            if not use_regular_gp and global_dir_name is not None:
                setup_crown_yaml_dkl(network_dims, mode, dim, crown_dir, global_dir_name, use_reLU, num_layers)
                torch_file_name = "/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)
                nn_to_save = model.feature_extractor
                torch.save(nn_to_save.state_dict(), crown_dir + "/models" + torch_file_name)

            model.eval()
            likelihood.eval()
            print("")
            gp_by_dim.append([model, likelihood])

        gp_by_region.append(gp_by_dim)
        region_info.append(region_data)
        print(f"finished sub-region {idx+1} of {len(local_regions)}\n")

    return gp_by_region, region_info


def dkl_save(file_name_base, dkl_models, mode, use_local_gp):
    mode_models = dkl_models[mode]
    if use_local_gp:
        num_regions = len(mode_models)
        for region in range(num_regions):
            region_model = mode_models[region]
            num_dims = len(region_model)
            for dim in range(num_dims):
                dim_model = region_model[dim]
                model = dim_model[0]
                file_name = file_name_base + f'_{mode}_{dim}_{region}.pth'

                state_dict = model.state_dict()
                torch.save(state_dict, file_name)
                time.sleep(0.1)
    else:
        num_dims = len(mode_models)
        for dim in range(num_dims):
            dim_model = mode_models[dim]
            model = dim_model[0]
            file_name = file_name_base + f'_{mode}_{dim}.pth'

            state_dict = model.state_dict()
            torch.save(state_dict, file_name)
            time.sleep(0.1)


def dkl_load(file_name_base, dkl_models, all_data, use_reLU, num_layers, network_dims):
    num_models = len(dkl_models)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for mode in range(num_models):
        file_name = file_name_base + f'_{mode}_0.pth'
        if not os.path.exists(file_name):
            break

        x_train = all_data[mode][0]
        y_train = np.transpose(all_data[mode][1])

        x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
        n_obs = max(np.shape(y_train))
        num_dims = min(np.shape(x_train))

        gp_by_dim = []
        for dim in range(num_dims):
            file_name = file_name_base + f'_{mode}_{dim}.pth'
            print(f'Loading DKL model for mode {mode} and dim {dim + 1}')

            y_data = torch.tensor(np.reshape(y_train[:, dim], (n_obs,)), dtype=torch.float32).to(device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            set_feature_extractor(use_reLU, num_layers, network_dims, 0)
            model = DeepKernelGP(x_data, y_data, likelihood)

            state_dict = torch.load(file_name)
            model.load_state_dict(state_dict)

            gp_by_dim.append([model, likelihood])
            time.sleep(1)

        dkl_models[mode] = gp_by_dim

    return dkl_models


############################################################################################
#  DKL posteriors
############################################################################################


def run_dkl_in_parallel_just_bounds(extents, mode, nn_out_dim, crown_dir, global_dir, exp_dir, linear_bounds_info,
                                    linear_transform_m, linear_transform_b, threads=8, use_regular_gp=False,
                                    merged=None):
    # This function does parallel calls to CROWN to get bounds on the NN output over input regions
    extent_len = len(extents)
    if not use_regular_gp:
        # this calls the Crown scripts in parallel
        os.chdir(crown_dir)

        dim_ = 0
        pool = mp.Pool(threads)
        results = pool.starmap(run_dkl_crown_parallel, [(extents[idx], crown_dir, global_dir, nn_out_dim, mode, dim_,
                                                         idx, merged) for idx in range(extent_len)])
        pool.close()

        # store the results
        os.chdir(exp_dir)
        for index, lin_trans in enumerate(results):
            saved_vals = np.array([lin_trans[4].astype(np.float64), lin_trans[5].astype(np.float64)])
            linear_bounds_info[index] = saved_vals
            saved_m = np.array([lin_trans[0], lin_trans[1]])
            linear_transform_m[index] = saved_m
            saved_b = np.array([lin_trans[2], lin_trans[3]])
            linear_transform_b[index] = saved_b

    else:
        transform_ = np.eye(nn_out_dim)
        transform_ = transform_.reshape(1, nn_out_dim, nn_out_dim)

        bias_ = np.zeros(1, nn_out_dim)
        for index, region in enumerate(extents):
            x_min = [k[0] for k in list(region)]
            x_max = [k[1] for k in list(region)]
            saved_vals = [np.array(x_min).astype(np.float64), np.array(x_max).astype(np.float64)]
            linear_bounds_info[index] = saved_vals

            saved_m = np.array([transform_, transform_])
            linear_transform_m[index] = saved_m

            saved_b = np.array([bias_, bias_])
            linear_transform_b[index] = saved_b

    return linear_bounds_info, linear_transform_m, linear_transform_b


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
