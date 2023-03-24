
from import_script import *
from dynamics_script import *
from mpl_toolkits.mplot3d import axes3d

# ======================================================================
# 1. Define NN structure and box dataset for crown
# ======================================================================


class DynModelNetTanh3(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetTanh3, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.tanh = nn.Tanh()
        self.layer_4 = nn.Linear(width_2, out_dim)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.tanh(out)
        out = self.layer_2(out)
        out = self.tanh(out)
        out = self.layer_3(out)
        out = self.tanh(out)
        out = self.layer_4(out)
        return out


class DynModelNetTanh1(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetTanh1, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, out_dim)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.Tanh(out)
        out = self.layer_2(out)
        return out


class DynModelNetRelu3(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetRelu3, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_3 = nn.Linear(width_2, width_2)
        self.layer_4 = nn.Linear(width_2, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        out = self.ReLU(out)
        out = self.layer_3(out)
        out = self.ReLU(out)
        out = self.layer_4(out)
        return out


class DynModelNetRelu2(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetRelu2, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, width_2)
        self.layer_4 = nn.Linear(width_2, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        out = self.ReLU(out)
        out = self.layer_4(out)
        return out


class DynModelNetRelu1(nn.Module):
    def __init__(self, d=1, width_1=1, width_2=1, out_dim=1):
        super(DynModelNetRelu1, self).__init__()
        self.layer_1 = nn.Linear(d, width_1)
        self.layer_2 = nn.Linear(width_1, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        return out


def simple_box_data(x_max, x_min):
    dim = len(x_max)
    X = torch.tensor([[(x_max[i] - x_min[i]) / dim + x_min[i] for i in range(dim)]]).float()
    labels = torch.tensor([0]).long()
    data_max = torch.tensor([[x_max[i] for i in range(dim)]]).reshape(1, -1)
    data_min = torch.tensor([[x_min[i] for i in range(dim)]]).reshape(1, -1)
    eps = None
    return X, labels, data_max, data_min, eps


def train_nn_model(global_exp_dir, network_dims, all_data, mode, global_dir_name, num_layers, use_relu, loss_fun,
                   reuse_regions=False, random_seed=11, lr=None, train_lim=None, good_loss=None, add_noise=None):
    if good_loss is None:
        good_loss = 0.05
    if train_lim is None:
        train_lim = 10000
    if lr is None:
        lr = 1e-3

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

    torch_file_name = "/unknown_dyn_model_mode_{}_experiment_{}.pt".format(mode, global_dir_name)

    if reuse_regions and os.path.isdir(global_exp_dir) and os.path.exists(global_exp_dir + "/" + torch_file_name):
        print("Loading Neural Network for mode {} from saved data...".format(mode))
        unknown_dyn_model = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)
        unknown_dyn_model.load_state_dict(torch.load(global_exp_dir + "/" + torch_file_name))
    else:
        x_train = all_data[mode]["x_train"]
        y_train = all_data[mode]["y_train"]

        if add_noise is not None:
            data_num = np.max(np.shape(y_train))
            sig = add_noise["sig"]
            noise = np.random.uniform(-sig, sig, size=(out_dim, data_num))
            y_train += noise

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_seed)
        unknown_dyn_model = net_model_to_use(d=d, width_1=width_1, width_2=width_2, out_dim=out_dim)
        unknown_dyn_model.to(device)
        x_data = torch.tensor(np.transpose(x_train), dtype=torch.float32, requires_grad=True).to(device)
        y_data = torch.tensor(np.transpose(y_train), dtype=torch.float32).to(device)
        learning_rate = lr
        epochs = train_lim
        # optimizer = torch.optim.RMSprop(unknown_dyn_model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(unknown_dyn_model.parameters(), lr=learning_rate)
        print("Training Neural Network on generated data for mode {}...".format(mode))
        for t in range(epochs):
            # Forward pass: compute predicted y by passing x to the model.
            
            # Compute and print loss.
            loss = loss_fun(unknown_dyn_model, y_data, x_data)
            if t % 500 == 499:
                print(t, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() < good_loss:
                print(t, loss.item())
                break

    device = torch.device('cpu')
    unknown_dyn_model.to(device)
    return unknown_dyn_model


def loss_function_standard(unknown_dyn_model, y_data, x_data):
    y_predict = unknown_dyn_model(x_data)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(y_predict, y_data)

    return loss


def loss_function_smooth(unknown_dyn_model, y_data, x_data):
    y_predict = unknown_dyn_model(x_data)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(y_predict, y_data)

    temp = torch.autograd.grad(y_predict, x_data, grad_outputs=torch.ones_like(y_predict), retain_graph=True)
    grad_loss = torch.div(torch.sum(torch.abs(temp[0])), len(x_data)*3)

    loss += grad_loss
    return loss


def loss_function_theta(unknown_dyn_model, y_data, x_data):
    y_predict = unknown_dyn_model(x_data)

    scale_tensor = torch.tensor([1., 1., 20.], dtype=torch.float32).to('cuda')

    y_predict = torch.mul(y_predict, scale_tensor)
    y_data = torch.mul(y_data, scale_tensor)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(y_predict, y_data)

    return loss


def find_data_in_subdomain(x_data, y_data, test_domain):
    keys = list(test_domain)
    num_data = max(np.shape(x_data))
    d = min(np.shape(x_data))

    intersects_list = []
    for data_idx in range(num_data):
        intersects_ = True
        for i, k in enumerate(keys):
            if (test_domain[k][0] <= x_data[i][data_idx]) and (x_data[i][data_idx] <= test_domain[k][1]):
                continue
            else:
                intersects_ = False
                break
        if intersects_:
            intersects_list.append(data_idx)

    num_samples = len(intersects_list)
    print(f'Found {num_samples} data points in the sub domain.')
    x1 = [x_data[idx][intersects_list] for idx in range(d)]
    x_predictor = torch.tensor(np.reshape(x1, [num_samples, d]), dtype=torch.float32)

    y1 = {key: torch.tensor(np.reshape(y_data[idx][intersects_list], (num_samples,)), dtype=torch.float32) for idx, key in enumerate(keys)}

    return [x_predictor, y1]


####################################
# Plotting scripts
####################################

def plot_synthesis(policy, region_labels, extents, extents_polys, global_exp_dir, refinement, product_transitions, dfa,
                   specification, unknown_modes_list, known_fnc=None, process_dist=None, random_seed=9248, x0=None, plot_=False):

    refinement_dir = global_exp_dir + "/refinement_{}".format(refinement)
    if not os.path.isdir(refinement_dir):
        os.mkdir(refinement_dir)

    extent_len = len(extents) - 1
    domain = extents[extent_len]

    fig, ax = plt.subplots(1, 1)

    x_length = domain["x1"][1] - domain["x1"][0]
    y_length = domain["x2"][1] - domain["x2"][0]
    fig.set_size_inches(x_length + 1, y_length + 1)

    Domain_Polygon = Polygon([(domain["x1"][0], domain["x2"][0]), (domain["x1"][0], domain["x2"][1]),
                              (domain["x1"][1], domain["x2"][1]), (domain["x1"][1], domain["x2"][0])])
    x_X, y_X = Domain_Polygon.exterior.xy
    ax.fill(x_X, y_X, alpha=0.6, fc='r', ec='k')

    sat_regions = []
    for state in list(product_transitions):
        if state[1] == 1:
            if policy[state] is not None:
                sat_regions.append(state[0])

    plotted_list = []
    for idx in sat_regions:
        region_area = extents[idx]

        x_dims = (region_area["x1"], region_area["x2"])
        if x_dims in plotted_list:
            continue
        plotted_list.append(x_dims)

        region_polygon = Polygon(
            [(region_area["x1"][0], region_area["x2"][0]), (region_area["x1"][1], region_area["x2"][0]),
             (region_area["x1"][1], region_area["x2"][1]), (region_area["x1"][0], region_area["x2"][1])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='w', ec='None')
        ax.fill(x_R, y_R, alpha=0.2, fc='g', ec='None')

    goal_regions = []

    goal_set = region_labels['goal']
    unsafe_set = region_labels['obs']

    for goal in goal_set:
        area = goal
        region_polygon = Polygon(
            [(area["x1"][0], area["x2"][0]), (area["x1"][1], area["x2"][0]),
             (area["x1"][1], area["x2"][1]), (area["x1"][0], area["x2"][1])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='None', ec='k')

    for goal in goal_set:
        # this index should be based on the order regions are labeled, i.e. idx 1 should be the post of region 1
        goal_area = goal
        if goal_area is None:
            continue
        b_lims = [goal_area[k] for k in list(goal_area)]
        goal_polytope = pc.box2poly(b_lims)
        goal_b = goal_polytope.b
        for idx in range(len(extents) - 1):
            region_area = extents[idx]
            b_lims = [region_area[k] for k in list(region_area)]
            region_polytope = pc.box2poly(b_lims)
            region_b = region_polytope.b
            intersects_ = all(region_b <= goal_b)
            if intersects_:
                goal_regions.append(idx)

    for obs in unsafe_set:
        area = obs
        if area is None:
            continue
        region_polygon = Polygon(
            [(area["x1"][0], area["x2"][0]), (area["x1"][1], area["x2"][0]),
             (area["x1"][1], area["x2"][1]), (area["x1"][0], area["x2"][1])])
        x_R, y_R = region_polygon.exterior.xy
        ax.fill(x_R, y_R, alpha=1, fc='k', ec='k')

    plt.xlabel('$x_{1}$', fontdict={"size": 15})
    plt.ylabel('$x_{2}$', fontdict={"size": 15})
    plt.show(block=False)
    plt.savefig(refinement_dir + '/synthesis_results_' + specification + '.png')

    # make sure not to plot in goal
    safe_regions = list(set(sat_regions) - set(goal_regions))
    np.random.seed(random_seed)

    if x0 is not None:
        for key in x0:
            x_start = x0[key]
            x_new = x_start
            x1 = [x_start[0]]
            x2 = [x_start[1]]

            for region_idx in range(extent_len):
                poly = extents_polys[region_idx]
                if x_new in poly:
                    init_state = (region_idx, 1)
                    break

            # simulate evolution
            step = 0
            while True:
                control_mode = policy[init_state]

                x_new = f(x_new, unknown_modes_list[control_mode], known_fnc,
                            process_dist=process_dist)
                step += 1

                # x_traj.extend([x_new])
                x1.append(x_new[0])
                x2.append(x_new[1])

                next_states = list(product_transitions[init_state][control_mode]["to"])

                for state in next_states:
                    region_idx = state[0]
                    poly = extents_polys[region_idx]
                    if x_new in poly:
                        init_state = state
                        break

                control_mode = policy[init_state]
                next_states = list(product_transitions[init_state][control_mode]["to"])
                if next_states == [(0, dfa["accept"])]:
                    # this means label of current state is accepting, doing a control policy may cause us to exit accept
                    break

            ax.plot(x1, x2, linestyle='--', marker='o', color='b', lw=2, alpha=1)

        plt.show(block=False)
        plt.savefig(refinement_dir + '/final_results.png')

    # test 5 possible starts
    if len(safe_regions) > 0 and plot_:
        # print('More than just the goal region satisfies the specification.')
        for random_start in range(1):
            safe_region_idx = np.random.randint(len(safe_regions), size=1)[0]
            region_idx = safe_regions[safe_region_idx]
            init_region = extents[region_idx]
            x_start = [np.random.uniform(init_region[k][0], init_region[k][1], 1).tolist()[0] for k in list(init_region)]

            init_state = (region_idx, 1)

            # generate a traj from x_start
            x_new = x_start
            x1 = [x_start[0]]
            x2 = [x_start[1]]
            while True:
                control_mode = policy[init_state]

                x_new = f(x_new, unknown_modes_list[control_mode], known_fnc,
                        process_dist=process_dist)

                # x_traj.extend([x_new])
                x1.append(x_new[0])
                x2.append(x_new[1])

                next_states = list(product_transitions[init_state][control_mode]["to"])

                for state in next_states:
                    region_idx = state[0]
                    poly = extents_polys[region_idx]
                    if x_new in poly:
                        init_state = state
                        break

                control_mode = policy[init_state]
                next_states = list(product_transitions[init_state][control_mode]["to"])
                if next_states == [(0, dfa["accept"])]:
                    # this means label of current state is accepting, doing a control policy may cause us to exit accept
                    break

            ax.plot(x1, x2, linestyle='--', marker='o', color='b', lw=2, alpha=1)

        plt.show(block=False)
        plt.savefig(refinement_dir + '/synthesis_traj_' + specification + '.png')


def plot_vector_fields(domain, grid_size, global_exp_dir):

    # Domain_Polygon = Polygon([(domain["x1"][0], domain["x2"][0]), (domain["x1"][0], domain["x2"][1]),
    #                           (domain["x1"][1], domain["x2"][1]), (domain["x1"][1], domain["x2"][0])])
    # x_X, y_X = Domain_Polygon.exterior.xy

    refinement_dir = global_exp_dir + "/eps_figs"
    if not os.path.isdir(refinement_dir):
            os.mkdir(refinement_dir)

    x, y, z = np.meshgrid(np.arange(domain["x1"][0], domain["x1"][1] + grid_size["x1"], grid_size["x1"]),
                          np.arange(domain["x2"][0], domain["x2"][1] + grid_size["x2"], grid_size["x2"]),
                          np.arange(domain["x3"][0]/2, domain["x3"][1]/2 + grid_size["x3"], grid_size["x3"]))


    x_length = domain["x1"][1] + grid_size["x1"] - domain["x1"][0] + 1.5
    y_length = domain["x2"][1] + grid_size["x2"] - domain["x2"][0] + 1.5

    u_vel = 10
    omega = 5
    Ts = 0.1
    phi_list = [-.45, -.3, -.15, 0, .15, .3, .45]

    for phi in phi_list:
        u = Ts * u_vel * np.cos(z)
        v = Ts * u_vel * np.sin(z)
        w = (phi - z) * Ts * omega

        fig = plt.figure(figsize=(x_length,y_length))
        ax = fig.gca(projection='3d', adjustable='box')
        ax.quiver(x, y, z, u, v, w, length=0.5, color='k')
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')

        plt.show(block=False)
        plt.savefig(refinement_dir + f'/vector_field_phi_{phi:0.3f}.png', dpi=300)

    return None


def plot_eps(error_est_dict, modes, global_exp_dir):

    refinement_dir = global_exp_dir + "/eps_figs"
    if not os.path.isdir(refinement_dir):
        os.mkdir(refinement_dir)

    better_eps = error_est_dict["better_eps"]
    bt_et = error_est_dict["bt_et"]
    Lips_data = error_est_dict["Lips_data"]

    for mode in modes:
        eps_dict = better_eps[mode]
        bt = bt_et[mode]["bt"]
        et = bt_et[mode]["et"]
        idx_list = list(eps_dict)

        x_vals = []
        eps_vals = []
        bt_vals = []
        et_vals = []
        for idx in idx_list:
            x_vals.append(idx)
            eps_vals.append(eps_dict[idx])
            bt_vals.append(bt[idx])
            et_vals.append(et[idx])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        fig.set_size_inches(7, 7)

        ax1.plot(x_vals, eps_vals, linestyle='None', marker='o', color='b', lw=2, alpha=1)

        ax1.set_title(f'Epsilon with L = {Lips_data[mode]:0.4f}')

        ax2.plot(x_vals, bt_vals, linestyle='None', marker='o', color='b', lw=2, alpha=1)
        ax2.set_title('b_T')

        ax3.plot(x_vals, et_vals, linestyle='None', marker='o', color='b', lw=2, alpha=1)
        ax3.set_title('e_T')

        plt.show(block=False)
        plt.savefig(refinement_dir + f'/eps_plot_mode_{mode}.png')


# ======================================================================
# 4. Save and load data functions
# ======================================================================

def dict_save(file_name, dict_to_save):
    file_ = open(file_name, "wb")
    # write the python object (dict) to pickle file
    pickle.dump(dict_to_save, file_)
    # close file
    file_.close()


def img_save(file_name, fig):
    file_ = open(file_name, "wb")
    # write the python object (fig) to pickle file
    pickle.dump(fig, file_)
    # close file
    file_.close()


def dict_load(file_name):
    file_ = open(file_name, "rb")
    return pickle.load(file_)

