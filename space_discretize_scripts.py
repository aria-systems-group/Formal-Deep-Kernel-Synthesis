from import_script import *

# ======================================================================
# 3. Region creation functions
# ======================================================================


def discretize_space(space, grid_size):
    extents = []
    boundaries = {}
    dim_ranges = {}
    for k in list(space):
        boundaries[k] = {"up": None, "low": None}
        x = space[k]
        x_d = np.arange(x[0], x[1] + grid_size[k], grid_size[k]).tolist()
        dim_ranges[k] = x_d
        extents.append([[x_d[i], x_d[i + 1]] for i in range(len(x_d) - 1)])
        boundaries[k]["up"] = {x_d[i]: [] for i in range(len(x_d))}
        boundaries[k]["low"] = {x_d[i]: [] for i in range(len(x_d))}

    state_extents = (itertools.product(*extents))
    discrete_sets = []
    for i, state in enumerate(state_extents):
        for j, k in enumerate(list(space)):
            if j == 0:
                discrete_sets.append({k: state[j]})
            else:
                discrete_sets[i][k] = state[j]
            for val in dim_ranges[k]:
                if val == state[j][1]:
                    boundaries[k]["up"][val].append(i)
                if val == state[j][0]:
                    boundaries[k]["low"][val].append(i)

    discrete_sets.append(space)
    return discrete_sets, boundaries


def separate_data_into_regions_new(regions, all_data):
    x_test = np.transpose(all_data[0].copy())
    y_test = np.transpose(all_data[1].copy())

    num_regions = len(regions)
    data_dict = [[None, None] for index in range(num_regions)]

    for idx in range(num_regions):
        region = regions[idx]

        # organize test data points
        in_region = []
        for point_idx, x in enumerate(x_test):
            if inside(x, region):
                in_region.append(point_idx)
        data_dict[idx][0] = np.transpose(x_test[in_region])
        data_dict[idx][1] = np.transpose(y_test[in_region])

        for index in sorted(in_region, reverse=True):
            # remove used data to reduce computation later?
            x_test = np.delete(x_test, index, axis=0)
            y_test = np.delete(y_test, index, axis=0)

    return data_dict


def inside(x, region):
    for idx, val in enumerate(x):
        if val > region[idx][1] or val < region[idx][0]:
            return False
    return True


def discretize_space_list(space, grid_size):
    extents = []
    space_list = []
    dim_ranges = {}
    for k in list(space):
        x = space[k]
        space_list.append(x)
        x_d = np.arange(x[0], x[1] + grid_size[k], grid_size[k]).tolist()
        dim_ranges[k] = x_d
        extents.append([[x_d[i], x_d[i + 1]] for i in range(len(x_d) - 1)])

    state_extents = (itertools.product(*extents))
    discrete_sets = []
    for i, state in enumerate(state_extents):
        for j, k in enumerate(list(space)):
            if j == 0:
                discrete_sets.append([state[j]])
            else:
                discrete_sets[i].append(state[j])

    discrete_sets.append(space_list)
    return discrete_sets
