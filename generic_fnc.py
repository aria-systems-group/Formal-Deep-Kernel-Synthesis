
from dynamics_script import *

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


def get_grid_info(X, grid_len):
    large_grid = {k: X[k][1] - X[k][0] for k in list(X)}  # essentially a placeholder
    if len(X) < 3:
        grid_size = {k: grid_len for k in list(X)}
    elif len(X) == 3:
        grid_size = {"x1": 0.125, "x2": 0.125, "x3": 0.05}
        grid_size = {"x1": 0.2, "x2": 0.2, "x3": 0.05}
    elif len(X) == 5:
        grid_size = {"x1": 0.25, "x2": 0.25, "x3": 0.2, "x4": 0.2, "x5": 0.2}
        # grid_size = {"x1": 0.125, "x2": 0.125, "x3": 0.2, "x4": 0.2, "x5": 0.2}
    else:
        print("You need to define a grid size for this dimensionality")
        exit()

    return grid_size, large_grid

# ======================================================================
#  Save and load data functions
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

