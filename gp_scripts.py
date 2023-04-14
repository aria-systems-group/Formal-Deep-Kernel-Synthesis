
from import_script import *
from gp_parallel import *
from generic_fnc import DynModelNetRelu3, DynModelNetRelu2, DynModelNetRelu1, DynModelNetTanh1, DynModelNetTanh3
from crown_scripts import setup_crown_yaml_dkl
from space_discretize_scripts import separate_data_into_regions_new, discretize_space_list


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


def train_feature_extractor(all_data, mode, use_relu, num_layers, network_dims, random_seed=20, good_loss=0.1,
                            epochs=10000, lr=1e-3):

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
    n_samples = int(np.shape(x_train)[1]/10.)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(random_seed)
    global feature_extractor
    feature_extractor = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)
    feature_extractor.to(device)
    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
    y_data = torch.tensor(np.transpose(y_train), dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    print("Training Neural Network on generated data for mode {}...".format(mode))
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        # y_predict = feature_extractor(x_data)
        # loss = loss_fn(y_predict, y_data)
        # get minibatch data, ~ 10% of the data
        perm = torch.randperm(y_data.size(0))
        idx = perm[:n_samples]
        y_mini = y_data[idx]
        x_mini = x_data[idx]
        # Compute and print loss.
        loss = loss_function_smooth(feature_extractor, y_mini, x_mini)

        # if t % 500 == 499:
        #     print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if loss.item() < good_loss:
        #     # print(t, loss.item())
        #     break

    feature_extractor.to(torch.device('cpu'))


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


class IMDPModel:
    def __init__(self, states, actions, pmin, pmax, labels, extents):
        self.states = states
        self.actions = actions
        self.Pmin = pmin
        self.Pmax = pmax
        self.labels = labels
        self.extents = extents


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

def deep_kernel_fixed_nn_local(all_data, mode, keys, use_reLU, num_layers, network_dims, crown_dir, global_dir_name,
                               grid_size, domain, training_iter=40, lr=0.01, process_noise=None, random_seed=11,
                               epochs=10000, nn_lr=1e-3, use_regular_gp=False):
    # this function trains either a standard se_kernel GP (is use_regular_gp=True) or a deep kernel model
    # The deep kernel model will use a pre-trained fixed NN while optimizing the kernel parameters
    # The models are trained on local data sets to reduce computational load in the future

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = 1e-2
    if process_noise is not None:
        alpha = process_noise["sig"]

    if not use_regular_gp:
        # train the NN with all the data
        train_feature_extractor(all_data, mode, use_reLU, num_layers, network_dims, random_seed=random_seed, good_loss=0.1,
                                epochs=epochs, lr=nn_lr)

    # Splitting data into desired regions for localization
    local_regions = discretize_space_list(domain, grid_size)
    local_regions.pop()
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
                alpha_ = max(alpha[dim], 5e-3)  # for some reason there are a lot of errors if the noise is seeded low
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
            final_loss = 9e9
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(x_data)

                # Calc loss and backprop gradients
                loss = -mll(output, y_data)

                loss.backward()
                final_loss = loss.item()
                if i % training_iter == training_iter-1:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' %
                          (i + 1, training_iter, loss.item(),
                           model.covar_module.base_kernel.lengthscale.item()))

                optimizer.step()

            if final_loss > -1.:
                # keep training for another 100 iterations
                print('Continuing training on this dimension...')
                for i in range(200):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(x_data)

                    # Calc loss and backprop gradients
                    loss = -mll(output, y_data)

                    loss.backward()
                    if i % 200 == 199:
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' %
                              (i + 1, 200, loss.item(),
                               model.covar_module.base_kernel.lengthscale.item()))

                    optimizer.step()

            if not use_regular_gp:
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


def deep_kernel_with_fixed_nn(all_data, mode, keys, use_reLU, num_layers, network_dims, crown_dir, global_dir_name,
                              training_iter=40, lr=0.01, process_noise=None, random_seed=11, epochs=10000, nn_lr=1e-3):
    x_train = all_data[mode][0]
    y_train = np.transpose(all_data[mode][1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)

    alpha = 1e-3
    if process_noise is not None:
        alpha = process_noise["sig"]

    hypers = {'likelihood.noise_covar.noise': torch.tensor(alpha), }

    n_obs = max(np.shape(y_train))
    gp_by_dim = []

    train_feature_extractor(all_data, mode, use_reLU, num_layers, network_dims, random_seed=random_seed, good_loss=0.1,
                            epochs=epochs, lr=nn_lr)
    print('Optimizing GP parameters\n')
    for dim, key in enumerate(keys):

        y_data = torch.tensor(np.reshape(y_train[:, dim], (n_obs,)), dtype=torch.float32).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model = DeepKernelGP(x_data, y_data, likelihood)
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        model.initialize(**hypers)
        model.train()
        likelihood.train()

        # NOTE, this does not optimize the NN feature extractor, just use a pre-trained version
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

        setup_crown_yaml_dkl(network_dims, mode, dim, crown_dir, global_dir_name, use_reLU, num_layers)
        torch_file_name = "/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)
        nn_to_save = model.feature_extractor
        torch.save(nn_to_save.state_dict(), crown_dir + "/models" + torch_file_name)

        model.eval()
        likelihood.eval()
        print("")

        gp_by_dim.append([model, likelihood])

    return gp_by_dim


def deep_kernel_learn(all_data, mode, keys, use_reLU, num_layers, network_dims, crown_dir, global_dir_name,
                      training_iter=40, lr=0.01, process_noise=None, random_seed=11):

    # define training data
    x_train = all_data[mode][0]
    y_train = np.transpose(all_data[mode][1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)

    alpha = 1e-3
    if process_noise is not None:
        alpha = process_noise["sig"]

    hypers = {'likelihood.noise_covar.noise': torch.tensor(alpha), }

    n_obs = max(np.shape(y_train))
    gp_by_dim = []
    for dim, key in enumerate(keys):

        y_data = torch.tensor(np.reshape(y_train[:, dim], (n_obs,)), dtype=torch.float32).to(device)

        set_feature_extractor(use_reLU, num_layers, network_dims, random_seed + dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = DeepKernelGP(x_data, y_data, likelihood)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        model.initialize(**hypers)
        model.train()
        likelihood.train()
        if use_DKL:
            optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters()},
                                          {'params': model.covar_module.parameters()},
                                          {'params': model.mean_module.parameters()},
                                          {'params': model.likelihood.parameters()},
                                          ], lr=lr)
        else:
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
            if use_toeplitz is not True:
                if i % 100 == 99:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' %
                          (i + 1, training_iter, loss.item(),
                           model.covar_module.base_kernel.lengthscale.item()))
            else:
                if i % 10 == 9:
                    print('Iter %d/%d - Loss: %.3f' %
                          (i + 1, training_iter, loss.item()))

            optimizer.step()

        setup_crown_yaml_dkl(network_dims, mode, dim, crown_dir, global_dir_name, use_reLU, num_layers)
        torch_file_name = "/unknown_dyn_model_mode_{}_dim_{}_experiment_{}.pt".format(mode, dim, global_dir_name)
        nn_to_save = model.feature_extractor
        torch.save(nn_to_save.state_dict(), crown_dir + "/models" + torch_file_name)

        model.eval()
        likelihood.eval()
        print("")

        gp_by_dim.append([model, likelihood])

    return gp_by_dim


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


# =============================================================================================================
# Posterior Predictions
# =============================================================================================================

def dict_save(file_name, dict_to_save):
    file_ = open(file_name, "wb")
    # write the python object (dict) to pickle file
    pickle.dump(dict_to_save, file_)
    # close file
    file_.close()


def generate_trans_par_dkl_subsections(extents, region_data,  modes, threads, file_name):

    num_extents = len(extents)
    num_modes = len(modes)
    num_dims = len(region_data[0][0][0])

    min_prob = None
    max_prob = None

    max_extents = 10000
    can_test = int(max_extents/num_modes)
    can_test = min(num_extents-1, can_test)

    num_left = num_extents*num_modes  # how many state/action pairs need evaluated
    while num_left > num_modes:

        insert_idx = 0
        for pre_idx in range(can_test):
            for mode in modes:
                min_, max_ = parallel_erf(extents, region_data[mode], num_dims, pre_idx, mode, threads=threads)
                if pre_idx == 0 and mode == 0:
                    min_prob_csr = csr_matrix(min_)
                    max_prob_csr = csr_matrix(max_)
                else:
                    min_prob_csr = vstack((min_prob_csr, csr_matrix(min_)), format='csr')
                    max_prob_csr = vstack((max_prob_csr, csr_matrix(max_)), format='csr')
                insert_idx += 1

        if min_prob is None:
            min_prob = min_prob_csr
            max_prob = max_prob_csr
        else:
            min_prob = vstack((min_prob, min_prob_csr), format='csr')  # this takes almost no space as a csr mat
            max_prob = vstack((max_prob, max_prob_csr), format='csr')

        probs = {"min": min_prob, "max": max_prob}
        dict_save(file_name, probs)
        del probs

        num_left -= insert_idx
        print(f'Have {num_left-num_modes} transition posteriors to find.')
        if num_left < can_test * num_modes:
            can_test = int(num_left/num_modes) - 1

    self_trans = np.zeros(num_extents)
    self_trans[num_extents - 1] = 1
    sparse_trans = csr_matrix(self_trans)
    for mode in modes:
        min_prob = vstack((min_prob, sparse_trans), format='csr')  # this takes almost no space as a csr mat
        max_prob = vstack((max_prob, sparse_trans), format='csr')

    return min_prob, max_prob


def generate_trans_par_dkl(extents, region_data,  modes, threads):

    num_extents = len(extents)
    num_dims = len(region_data[0][0][0])

    bar = progressbar.ProgressBar(maxval=num_extents * len(modes)).start()
    # alive_ = alive_bar(num_extents * len(modes), bar='bubbles')

    insert_idx = 0
    for pre_idx in range(num_extents - 1):
        for mode in modes:
            bar.update(insert_idx)
            # alive_()
            min_, max_ = parallel_erf(extents, region_data[mode], num_dims, pre_idx, mode, threads=threads)
            if np.sum(max_) < 1 or np.sum(min_) > 1:
                print(f'Something is wrong at index {insert_idx}')
                exit()

            if pre_idx == 0 and mode == 0:
                min_prob = csr_matrix(min_)
                max_prob = csr_matrix(max_)
            else:
                min_prob = vstack((min_prob, csr_matrix(min_)), format='csr')
                max_prob = vstack((max_prob, csr_matrix(max_)), format='csr')
            insert_idx += 1

    self_trans = np.zeros(num_extents)
    self_trans[num_extents-1] = 1
    sparse_trans = csr_matrix(self_trans)

    for mode in modes:
        bar.update(insert_idx)
        # alive_()
        min_prob = vstack((min_prob, sparse_trans), format='csr')
        max_prob = vstack((max_prob, sparse_trans), format='csr')
        insert_idx += 1

    return min_prob, max_prob


# =============================================================================================================
#  IMDP synthesis
# =============================================================================================================

def delta_(q, dfa, label):
    # TODO
    labels = []
    for relation in dfa["trans"]:
        if relation[0] == q and relation[1] == label:
            labels.append(relation[2])

    return labels


def construct_pimdp(dfa, imdp:IMDPModel):
    # TODO
    # this assumes one accepting state and one sink state
    states = dfa["states"]
    size_A = len(states)
    sink_state = dfa["sink"]
    accept_state = dfa["accept"]

    states_sans_accept = states.copy()
    states_sans_accept.remove(accept_state)
    states_sans_accept.remove(sink_state)

    # qinit = delta(s, L(s))

    N, M = np.shape(imdp.Pmin)
    Pmin_new = lil_matrix(np.zeros([N*size_A, M*size_A]))
    Pmax_new = lil_matrix(np.zeros([N*size_A, M*size_A]))

    pimdp_states = []
    pimdp_actions = imdp.actions
    n_actions = len(pimdp_actions)

    new_labels = np.zeros([1, M*size_A])
    idx = 0
    for s in imdp.states:
        for q in states:
            if q == accept_state:
                new_labels[idx] = 1
            pimdp_states.append((s, q))
            idx += 1
            
    for sq in pimdp_states:
        for a in pimdp_actions:
            row_idx = sq[0]*size_A*n_actions + (sq[1]-1)*n_actions + a
            row_min = np.zeros([1, M*size_A])
            row_max = np.zeros([1, M*size_A])
            q_prime = delta_(sq[1], dfa, imdp.labels[sq[0]])
            for sqp in pimdp_states:
                # transition from (x, q) -(a)-> (x', q') if x -(a)-> x' and q' = del(q, L(x))
                if q_prime == sqp[1]:
                    if (sq[1] == sink_state and sqp[1] == sink_state) or \
                            (sq[1] == accept_state and sqp[1] == accept_state):
                        col_idx = sq[0]*size_A + sq[1]
                        Pmin_new[row_idx, col_idx] = 1.
                        Pmax_new[row_idx, col_idx] = 1.
                    else:
                        col_idx = sqp[0]*size_A + sqp[1]
                        Pmin_new[row_idx, col_idx] = imdp.Pmin[(sq[0]*n_actions) + a, sqp[0]]
                        Pmax_new[row_idx, col_idx] = imdp.Pmax[(sq[0]*n_actions) + a, sqp[0]]

    Pmin_new = Pmin_new.tocsr(copy=True)
    Pmax_new = Pmax_new.tocsr(copy=True)

    return None


def label_states(labels, extents, unsafe_label):
    state_labels = []
    dims = len(extents[0])
    for idx, extent in enumerate(extents):
        if idx == len(extents) - 1:
            state_labels.append(unsafe_label)
            continue
        possible_labels = []
        for label in labels:
            # does this extent fit in any labels
            in_ranges = True
            ranges = labels[label]
            for sub_range in ranges:
                in_sub_range = True
                if sub_range is None:
                    in_ranges = False
                    break
                for dim in range(dims):
                    if not (extent[dim][0] >= sub_range[dim][0] and extent[dim][1] <= sub_range[dim][1]):
                        in_sub_range = False
                        break
                if not in_sub_range:
                    in_ranges = False
                    break

            if in_ranges:
                possible_labels.append(label)
            else:
                possible_labels.append("!" + label)
        extent_label = ""
        for idx_, label in enumerate(possible_labels):
            if idx_ > 0:
                extent_label += '∧' + label
            else:
                extent_label += label
        state_labels.append(extent_label)

    return state_labels


def satisfies(label, phi):
    separated_label = label.split('∧')
    for sub in separated_label:
        if sub == phi:
            return True
    return False


def find_q_yes(phi, labels):
    qyes = []

    for idx, label in enumerate(labels):
        if satisfies(label, phi):
            qyes.append(idx)

    return qyes


def find_q_no(phi1, phi2, labels):
    qno = []
    for idx, label in enumerate(labels):
        if not (satisfies(label, phi1) or satisfies(label, phi2)):
            qno.append(idx)

    return qno


def bounded_until(imdp, phi1, phi2, k, imdp_filepath, synthesis_flag=False):

    # Get the Qyes and Qno states
    Qyes = find_q_yes(phi2, imdp.labels)
    Qno = find_q_no(phi1, phi2, imdp.labels)
    Qno.append(len(imdp.states))  # for leaving the space

    # Write them to the file
    print('Writing IMDP to file')
    write_imdp_to_file_bounded(imdp, Qyes, Qno, imdp_filepath)
    # Do verification or synthesis
    if synthesis_flag:
        print('Running Synthesis')
        result_mat = run_imdp_synthesis(imdp_filepath, k, mode1="maximize")
    else:
        print('Running Verification')
        result_mat = run_imdp_synthesis(imdp_filepath, k, mode1="minimize")
    # Done
    return result_mat


def write_imdp_to_file_bounded(imdp:IMDPModel, Qyes, Qno, filename):
    file_ = open(filename, "w")
    state_num = len(imdp.states) + 1
    action_num = len(imdp.actions)

    file_.write(f"{state_num}\n")
    file_.write(f"{action_num}\n")

    num_acc = len(Qyes)
    file_.write(f"{num_acc}\n")
    [file_.write(f"{acc_state} ") for acc_state in Qyes]
    file_.write(f"\n")

    for i in range(state_num):
        if i not in Qno:
            for action in imdp.actions:
                row_idx = i*action_num + action
                pmax = imdp.Pmax[row_idx].toarray()[0]
                pmin = imdp.Pmin[row_idx].toarray()[0]
                ij = [j for j, v in enumerate(pmax) if v > 0.0001]
                for j in ij:
                    file_.write(f"{i} {action} {j} {pmin[j]} {pmax[j]}")
                    if (i < state_num-1) or (j != ij[len(ij)-1]) or (action < action_num-1):
                        file_.write(f"\n")
        else:
            file_.write(f"{i} {0} {i} {1.} {1.}")
            if i < state_num-1:
                file_.write(f"\n")
    file_.close()


def run_imdp_synthesis(imdp_file, k, ep=1e-6, mode1="maximize", mode2="pessimistic"):
    exe_path = "/usr/local/bin/synthesis"  # Assumes that this program is on the user's path
    res = subprocess.run([exe_path, mode1, mode2, str(k), str(ep), imdp_file], capture_output=True, text=True)
    res_mat = res_to_numbers(res.stdout)
    return res_mat


def res_to_numbers(res):
    res_filter = res.replace("\n", " ")
    res_split = res_filter.split(" ")
    num_rows = int(len(res_split)/4)

    res_mat = np.zeros([num_rows, 4])
    for i in range(num_rows):
        res_mat[i][0] = int(res_split[i * 4 + 1])
        res_mat[i][1] = int(res_split[i * 4 + 2])
        res_mat[i][2] = float(res_split[i * 4 + 3])
        res_mat[i][3] = float(res_split[i * 4 + 4])

    return res_mat

# =============================================================================================================
#  Plot IMDP Synthesis Results
# =============================================================================================================


def plot_verification_results(res_mat, imdp, global_exp_dir, k, region_labels, num_dfa_states=1, min_threshold=0.8, prob_plots=False, lazy_flag=False,
                              plot_tag="", trajectories=None, dfa_init_state=1):
    extents = imdp.extents

    imdp_dir = global_exp_dir + f"/imdp_{k}"
    if not os.path.isdir(imdp_dir):
        os.mkdir(imdp_dir)

    extent_len = len(extents) - 1
    domain = extents[extent_len]

    fig, ax = plt.subplots(1, 1)

    x_length = domain[0][1] - domain[0][0]
    y_length = domain[1][1] - domain[1][0]
    fig.set_size_inches(x_length + 1, y_length + 1)

    Domain_Polygon = Polygon([(domain[0][0], domain[1][0]), (domain[0][0], domain[1][1]),
                              (domain[0][1], domain[1][1]), (domain[0][1], domain[1][0])])
    x_X, y_X = Domain_Polygon.exterior.xy
    ax.fill(x_X, y_X, alpha=0.6, fc='r', ec='k')

    # figure out how to plot from the res mat
    sat_regions = []
    unsat_regions = []
    maybe_regions = []
    for i in range(extent_len):
        max_prob = res_mat[i][3]
        min_prob = res_mat[i][2]

        if min_prob >= min_threshold:
            # yay this region satisfied
            sat_regions.append(i)
        elif max_prob < min_threshold:
            unsat_regions.append(i)
        else:
            maybe_regions.append(i)

    print(f"Qyes = {len(sat_regions)}, Qno = {len(unsat_regions)}, Q? = {len(maybe_regions)}")

    # don't need to plot unsat, plot maybe in light red
    plotted_list = []
    for idx in maybe_regions:
        region = extents[idx]
        x_dims = (region[0], region[1])
        if x_dims in plotted_list:
            continue
        plotted_list.append(x_dims)

        region_polygon = Polygon([(region[0][0], region[1][0]), (region[0][0], region[1][1]),
                                  (region[0][1], region[1][1]), (region[0][1], region[1][0])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='w', ec='None')
        ax.fill(x_R, y_R, alpha=0.2, fc='r', ec='None')

    # plot yes in green
    plotted_list = []
    for idx in sat_regions:
        region = extents[idx]

        x_dims = (region[0], region[1])
        if x_dims in plotted_list:
            continue
        plotted_list.append(x_dims)

        region_polygon = Polygon([(region[0][0], region[1][0]), (region[0][0], region[1][1]),
                                  (region[0][1], region[1][1]), (region[0][1], region[1][0])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='w', ec='None')
        ax.fill(x_R, y_R, alpha=0.2, fc='g', ec='None')

    goal_set = region_labels['a']
    unsafe_set = region_labels['b']

    for goal in goal_set:
        area = goal
        region_polygon = Polygon(
            [(area[0][0], area[1][0]), (area[0][1], area[1][0]),
             (area[0][1], area[1][1]), (area[0][0], area[1][1])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='None', ec='k')

    for obs in unsafe_set:
        area = obs
        if area is None:
            continue
        region_polygon = Polygon(
            [(area[0][0], area[1][0]), (area[0][1], area[1][0]),
             (area[0][1], area[1][1]), (area[0][0], area[1][1])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='k', ec='k')

    plt.xlabel('$x_{1}$', fontdict={"size": 15})
    plt.ylabel('$x_{2}$', fontdict={"size": 15})
    plt.show(block=False)
    plt.savefig(imdp_dir + f'/synthesis_results_{k}.png')

    return None

