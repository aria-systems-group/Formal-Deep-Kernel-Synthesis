from import_script import *
from ltlf2dfa.parser.ltlf import LTLfParser
import re


def specification_to_dfa(specification):
    # returns a tuple dfa as (states, labels, transitions, accepting_state, sink_state, init_state)

    parser = LTLfParser()
    formula = parser(specification)
    dfa_mona = formula.to_dfa(mona_dfa_out=True)

    ####################################################################################################################
    # Identify atomic propositions using mona_dfa

    idx_prop_s = dfa_mona.find(':')
    idx_prop_e = dfa_mona.find('\n')

    props = str.replace(str.lower(dfa_mona[idx_prop_s + 2:idx_prop_e - 1]), " ", "")

    dfa_props_1 = []
    for idx in range(len(props)):
        dfa_props_1.extend(props[idx])

    ####################################################################################################################
    # Identify states using mona_dfa
    m = re.search(r'Accepting states: ', dfa_mona)
    idx_state_e = dfa_mona.find('\n', m.span()[1])
    accepting_state_str = dfa_mona[m.span()[1]:idx_state_e - 1]

    m = re.search(r'Rejecting states: ', dfa_mona)
    idx_state_e = dfa_mona.find('\n', m.span()[1])
    rejecting_state_str = dfa_mona[m.span()[1] + 2:idx_state_e - 1]  # this removes the 0 state that is unneeded

    mona_states = accepting_state_str + " " + rejecting_state_str

    dfa_states_1 = get_state_from_str(mona_states)
    accepting_states = get_state_from_str(accepting_state_str)
    rejecting_states = get_state_from_str(rejecting_state_str)

    first_accept = accepting_states[0]
    if len(accepting_states) > 1:
        for accepting_idx in range(1, len(accepting_states)):
            index = dfa_states_1.index(accepting_states[accepting_idx])
            dfa_states_1.pop(index)

    ####################################################################################################################
    # Identify transitions, putting all accepting states under one label and removing transitions out of accepting state

    dfa_1 = formula.to_dfa()

    m = re.search(r'init -> ', dfa_1)
    idx_e = dfa_1.find(';', m.span()[1])

    dfa_init_state_1 = int(dfa_1[m.span()[1]:idx_e])

    dfa_transitions_str = str.replace(str.replace(str.replace(dfa_1[idx_e + 3:len(dfa_1) - 1], " & ", "∧"), "~", "!"),
                                      " | ", "v")
    dfa_transitions_1 = []

    idx_new = 0
    idx_trans_e = dfa_transitions_str.find('\n')
    while idx_new < (len(dfa_transitions_str) - 1):
        idx_start = dfa_transitions_str.find('-', idx_new)
        idx_end = dfa_transitions_str.find('[', idx_new)

        start_state = int(dfa_transitions_str[idx_new:idx_start - 1])
        end_state = int(dfa_transitions_str[idx_start + 3:idx_end - 1])

        start_changed = False
        if (start_state in accepting_states) and (start_state is not first_accept):
            start_state = first_accept
            start_changed = True

        if end_state in accepting_states:
            end_state = first_accept

        idx_trans_s = dfa_transitions_str.find("\"", idx_new)
        idx_trans_e = dfa_transitions_str.find("\"", idx_trans_s + 1)

        trans_str = dfa_transitions_str[idx_trans_s + 1:idx_trans_e]

        idx_new = dfa_transitions_str.find('\n', idx_new) + 1
        if start_changed is True:
            if trans_str != "true":
                # only keep "true" self transition for accepting states,
                # don't need to care if it leaves the accepting state
                continue

        input_ = (start_state, trans_str, end_state)
        dfa_transitions_1.append(input_)

    ####################################################################################################################
    # find the absorbing states as accept and sink

    dfa_temp = {"states": dfa_states_1, "props": dfa_props_1, "trans": dfa_transitions_1,
                "accept": None, "sink": None, "init": dfa_init_state_1}

    dfa_accepting_state_1 = first_accept
    dfa_sink_state_1 = None
    for test_state in rejecting_states:
        validate = get_next_dfa_labels(dfa_temp, test_state)
        if not validate:
            # validate is empty
            dfa_sink_state_1 = test_state
            break

    dfa = {"states": dfa_states_1, "props": dfa_props_1, "trans": dfa_transitions_1,
           "accept": dfa_accepting_state_1, "sink": dfa_sink_state_1, "init": dfa_init_state_1}

    return dfa


def get_state_from_str(input_str):
    new_str = str.replace(input_str, " ", ";") + ';'

    output = []
    prev_e = 0
    idx_state_e = new_str.find(';')
    while idx_state_e != (len(new_str) - 1):
        output.extend([int(new_str[prev_e:idx_state_e])])
        prev_e = idx_state_e + 1
        idx_state_e = new_str.find(';', prev_e)
    output.extend([int(new_str[prev_e:idx_state_e])])

    return output


def get_next_dfa_labels(dfa, state):
    labels = []
    for relation in dfa["trans"]:
        if (relation[0] is state) and (relation[0] is not relation[2]) and (relation[2] is not dfa["accept"]):
            labels.extend([relation[1]])

    return labels


def get_prior_dfa_labels(dfa, state):
    labels = {}

    # avoid checking self transitions, generally not what path we want
    for relation in dfa["trans"]:
        if (relation[2] is state) and (relation[0] is not relation[2]):
            labels[relation[0]] = [relation[1]]

    if len(labels) == 0:
        # now allow self transitions, might be required
        for relation in dfa["trans"]:
            if relation[2] is state:
                labels[relation[0]] = [relation[1]]

    return labels


def find_path_to_accept(dfa, paths, accept, prior_solutions):
    # returns a list of lists of potential transitions that reach accept from init in reverse
    # e.g. if you can go from init -> a -> 2 -> b -> accept and init -> c -> 3 -> a -> accept
    # it will return [[b,a], [a,c]]

    init = dfa["init"]

    labels = get_prior_dfa_labels(dfa, accept)
    prior_states = list(labels)
    for state in prior_states:
        sub_path = [labels[state]]
        if state is init:
            paths.extend(sub_path)
        else:
            prior_solutions = paths
            test_out = find_path_to_accept(dfa, sub_path, state, prior_solutions)
            paths.append(test_out)

    return paths


def adjust_paths(paths):
    new_paths = []
    for path in paths:
        if len(path) > 1:
            if all([len(i) == 1 for i in path]):
                new_paths.append(path)
            else:
                # found an index that has multiple ways back to init, parse paths
                first_index = 0
                for index, i in enumerate(path):
                    if len(i) > 1:
                        first_index = index
                        break

                temp_paths = [path[0:first_index]]
                counter = 0
                while first_index + counter < len(path):
                    temp_list = []
                    temp_list.extend(path[0:(first_index-1)])
                    temp_list.extend(path[first_index + counter])
                    temp_paths.append(temp_list)
                    counter += 1

                # ensure there aren't nested paths like this
                adjusted = adjust_paths(temp_paths)
                new_paths.extend(adjusted)
        else:
            new_paths.append(path)

    return new_paths


def find_transitions(q, dfa):
    labels = []
    for relation in dfa["trans"]:
        if relation[0] == q:
            labels.append((relation[1], relation[2]))

    return labels


def regions_by_label(x_prime, region_idx_by_label):
    labels = list(region_idx_by_label)
    x_by_label = {}
    for label in labels:
        x_by_label[label] = []
        x_by_label["!" + label] = []
        for x in x_prime:
            if x in region_idx_by_label[label]:
                x_by_label[label].append(x)
            else:
                x_by_label["!" + label].append(x)

    return x_by_label


def label_of_region(x, region_idx_by_label):
    labels = list(region_idx_by_label)

    possible_labels = []
    for label in labels:
        if x in region_idx_by_label[label]:
            possible_labels.append(label)
        else:
            possible_labels.append("!" + label)

    return possible_labels


def construct_product_automata(dfa, transitions, region_labels, modes, extents_poly, refinement, max_refinements, goal_regions=[]):
    states = dfa["states"]
    sink_state = dfa["sink"]
    accept_state = dfa["accept"]

    extent_len = len(extents_poly)

    states_sans_accept = states.copy()
    states_sans_accept.remove(accept_state)
    states_sans_accept.remove(sink_state)

    region_idx_by_label = {}
    for label in region_labels:
        region_idx_by_label[label] = []

        for label_idx in range(len(region_labels[label])):
            # this is if e.g. label b has two distinct non-overlapping regions in it
            label_dims = region_labels[label][label_idx]
            if label_dims is None:
                continue
            b_lims = [label_dims[k] for k in list(label_dims)]
            label_polytope = pc.box2poly(b_lims)
            b_label = label_polytope.b
            for extent_idx in range(len(extents_poly) - 1):
                b_region = extents_poly[extent_idx]
                # b_region = region_area.b
                if all(b_region <= b_label):
                    # the region extent_idx is contained in the label
                    region_idx_by_label[label].append(extent_idx)

    # new state is (region, dfa_state)
    product_transitions = {(region_idx, state): {mode: {"to": [], "from": []} for mode in modes} for region_idx in
                           range(extent_len) for state in states_sans_accept}

    product_transitions[(0, accept_state)] = {mode: {"to": [], "from": []} for mode in modes}
    product_transitions[(0, sink_state)] = {mode: {"to": [], "from": []} for mode in modes}

    copy_of_keys = product_transitions.copy()

    # transition from (x, q) -(a)-> (x', q') if x -(a)-> x' and q' = del(q, L(x))
    for init_state in copy_of_keys:
        if init_state[1] == accept_state:
            # we don't care about transitions from accept
            continue

        if init_state in goal_regions:
            # already have a solution for this state, remove it from product it will be labeled as accept
            product_transitions.pop(init_state)
            continue

        x = init_state[0]
        L_of_x = label_of_region(x, region_idx_by_label)
        q = init_state[1]
        dfa_trans = find_transitions(q, dfa)

        # (x, q)
        for mode in modes:
            # under action a

            if x == (extent_len - 1):
                # this is outside the compact set X
                product_transitions[init_state][mode]["to"].append((0, sink_state))
                product_transitions[(0, sink_state)][mode]["from"].append(init_state)
                continue

            x_prime = transitions[x][mode]["to"]

            for trans_pair in dfa_trans:
                # goes to (x', q')
                q_prime = trans_pair[1]
                label = trans_pair[0]

                # TODO, handle or
                trans_to_q_prime = True
                sub_parts = label.split('∧')
                for part in sub_parts:
                    if part not in L_of_x:
                        # no transition to q_prime
                        trans_to_q_prime = False
                        break

                if trans_to_q_prime:
                    for x_p in x_prime:
                        if (q_prime == accept_state) or (q_prime == sink_state):
                            product_transitions[init_state][mode]["to"].append((0, q_prime))
                            product_transitions[(0, q_prime)][mode]["from"].append(init_state)
                        else:
                            product_transitions[init_state][mode]["to"].append((x_p, q_prime))
                            product_transitions[(x_p, q_prime)][mode]["from"].append(init_state)

    # clean up product, make sure there aren't duplicate transitions
    for init_state in product_transitions:
        for mode in modes:
            goes_to = product_transitions[init_state][mode]["to"]
            product_transitions[init_state][mode]["to"] = set(goes_to)

            comes_in = product_transitions[init_state][mode]["from"]
            product_transitions[init_state][mode]["from"] = set(comes_in)

    # do I even need the sink state?
    product_transitions.pop((0, sink_state))      
    # if refinement == max_refinements:
    #     # don't need these states
    #     for k in reversed(list(product_transitions)):
    #         for mode in modes:
    #             if (0, sink_state) in product_transitions[k][mode]["to"]:
    #                 product_transitions[k][mode]["to"] = []

    return product_transitions


def construct_product_automata_w_policy(dfa, transitions, region_labels, modes, extents_poly, policy):
    states = dfa["states"]
    sink_state = dfa["sink"]
    accept_state = dfa["accept"]

    extent_len = len(extents_poly)

    states_sans_accept = states.copy()
    states_sans_accept.remove(accept_state)
    states_sans_accept.remove(sink_state)

    region_idx_by_label = {}
    for label in region_labels:
        region_idx_by_label[label] = []

        for label_idx in range(len(region_labels[label])):
            # this is if e.g. label b has two distinct non-overlapping regions in it
            label_dims = region_labels[label][label_idx]
            if label_dims is None:
                continue
            b_lims = [label_dims[k] for k in list(label_dims)]
            label_polytope = pc.box2poly(b_lims)
            b_label = label_polytope.b
            for extent_idx in range(len(extents_poly) - 1):
                b_region = extents_poly[extent_idx]
                # b_region = region_area.b
                if all(b_region <= b_label):
                    # the region extent_idx is contained in the label
                    region_idx_by_label[label].append(extent_idx)

    # new state is (region, dfa_state)
    product_transitions = {(0, accept_state) : {mode: {"to": []} for mode in modes}}

    product_transitions[(0, sink_state)] = {mode: {"to": []} for mode in modes}

    copy_of_keys = list(policy)

    # transition from (x, q) -(a)-> (x', q') if x -(a)-> x' and q' = del(q, L(x))
    for init_state in copy_of_keys:
        if init_state not in product_transitions:
            product_transitions[init_state] = {policy[init_state]: {"to": []}}
        if init_state[1] == accept_state:
            # we don't care about transitions from accept
            continue

        x = init_state[0]
        L_of_x = label_of_region(x, region_idx_by_label)
        q = init_state[1]
        dfa_trans = find_transitions(q, dfa)

        # (x, q)
        mode = policy[init_state]
        # under action a

        if x == (extent_len - 1):
            # this is outside the compact set X
            product_transitions[init_state][mode]["to"].append((0, sink_state))
            continue

        x_prime = transitions[x][mode]["to"]

        for trans_pair in dfa_trans:
            # goes to (x', q')
            q_prime = trans_pair[1]
            label = trans_pair[0]

            # TODO, handle or
            trans_to_q_prime = True
            sub_parts = label.split('∧')
            for part in sub_parts:
                if part not in L_of_x:
                    # no transition to q_prime
                    trans_to_q_prime = False
                    break

            if trans_to_q_prime:
                for x_p in x_prime:
                    if (q_prime == accept_state) or (q_prime == sink_state):
                        product_transitions[init_state][mode]["to"].append((0, q_prime))
                    else:
                        product_transitions[init_state][mode]["to"].append((x_p, q_prime))

    # clean up product, make sure there aren't duplicate transitions
    for init_state in product_transitions:
        modes_ = list(product_transitions[init_state])
        for mode in modes_:
            goes_to = product_transitions[init_state][mode]["to"]
            product_transitions[init_state][mode]["to"] = set(goes_to)

    return product_transitions
