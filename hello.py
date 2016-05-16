import sys
import itertools
import numpy as np
import pandas as pd
import scipy.stats as sp
from copy import deepcopy


def get_full_joint_from_data(given_data_df, values_dict):
    """ Obtain the full joint distribution from the data set
        P(R,G,H,J,A,C,V)
    """
    indices = (
        values_dict['Region'],
        values_dict['Gender'],
        values_dict['Handedness'],
        values_dict['Jacket'],
        values_dict['Age'],
        values_dict['Colour'],
        values_dict['Vote']
    )
    c_names = ['Region', 'Gender', 'Handedness',
               'Jacket', 'Age', 'Colour', 'Vote']
    full_joint_df = pd.DataFrame(
        np.nan, index=pd.MultiIndex.from_product(indices, names=c_names),
        columns=['P'])
    total_sample_cnt = len(given_data_df)
    for comb in list(itertools.product(*indices)):
        p = len(given_data_df[
                (given_data_df['Region'] == comb[0]) &
                (given_data_df['Gender'] == comb[1]) &
                (given_data_df['Handedness'] == comb[2]) &
                (given_data_df['Jacket'] == comb[3]) &
                (given_data_df['Age'] == comb[4]) &
                (given_data_df['Colour'] == comb[5]) &
                (given_data_df['Vote'] == comb[6])
                ].index) / total_sample_cnt
        full_joint_df['P'][comb] = p
    return full_joint_df


def get_model_joint(BN, values_dict, data):
    """ Obtain the Joint distribution of the current model under consideration
        This obtained by multiplying the model parameters.
    """
    inv_BN = {}
    for v in BN:
        for e in BN[v]:
            if e not in inv_BN:
                inv_BN[e] = []
            inv_BN[e].append(v)
    for v in BN:
        if v not in inv_BN:
            inv_BN[v] = []

    model_params = {}
    for key, value in inv_BN.items():
        if value == []:
            model_params[key] = get_marginal_from_data(
                key, values_dict, data)
        else:
            model_params[key] = get_conditional_from_data(
                key, value, values_dict, data)

    indices = (
        values_dict['Region'],
        values_dict['Gender'],
        values_dict['Handedness'],
        values_dict['Jacket'],
        values_dict['Age'],
        values_dict['Colour'],
        values_dict['Vote']
    )
    c_names = ['Region', 'Gender', 'Handedness',
               'Jacket', 'Age', 'Colour', 'Vote']
    full_joint_df = pd.DataFrame(
        np.nan, index=pd.MultiIndex.from_product(indices, names=c_names),
        columns=['P'])
    idx_map = {
        'Region': 0,
        'Gender': 1,
        'Handedness': 2,
        'Jacket': 3,
        'Age': 4,
        'Colour': 5,
        'Vote': 6
    }
    for comb in list(itertools.product(*indices)):
        p = 1
        for key, param in model_params.items():
            idx_param = [key] + inv_BN[key]
            sub_list = []
            for var in idx_param:
                sub_list.append(comb[idx_map[var]])
            p *= param['P'].loc[tuple(sub_list)]
        full_joint_df['P'][comb] = p

    return full_joint_df


def get_marginal_from_data(var, values_dict, data):
    """ Obtain the marginal distribution of a given variable from the data
    """
    indices = [values_dict[var]]
    c_names = [var]

    marg_df = pd.DataFrame(
        np.nan, index=pd.MultiIndex.from_product(indices, names=c_names),
        columns=['P'])

    for comb in list(itertools.product(*indices)):
        p = len(data[(data[var]) == comb[0]].index) / len(data)
        marg_df['P'][comb] = p

    return marg_df


def get_conditional_from_data(var, given, values_dict, data):
    """ Obtain the conditional distribution from the data.
    """
    indices = [values_dict[var]]
    for g in given:
        indices.append(values_dict[g])
    c_names = [var] + given

    cond_df = pd.DataFrame(
        np.nan, index=pd.MultiIndex.from_product(indices, names=c_names),
        columns=['P'])

    # reduce samples
    for comb in list(itertools.product(*indices)):
        given_sample_space = data[:]
        for idx, val in enumerate(comb[1:]):
            given_sample_space = given_sample_space[
                (given_sample_space[given[idx]] == val)]
        if len(given_sample_space) == 0:
            p = 10**-4
        else:
            p = len(given_sample_space[(given_sample_space[var]) == comb[
                    0]].index) / len(given_sample_space)
        cond_df['P'][comb] = p
    return cond_df


def cyclic(g):
    """Return True if the directed graph g has a cycle.
    g must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    Reference: http://codereview.stackexchange.com/a/86067
    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False

    return any(visit(v) for v in g)


def main():
    V_index_list = ['Bernie', 'Donald', 'Hillary', 'Ted']
    J_index_list = ['full', 'part', 'never']
    G_index_list = ['male', 'female']
    R_index_list = ['north', 'south', 'east', 'west']
    H_index_list = ['right', 'left']
    C_index_list = ['black', 'white']
    A_index_list = ['new', 'worn', 'old']

    values = {
        'Region': R_index_list,
        'Gender': G_index_list,
        'Handedness': H_index_list,
        'Jacket': J_index_list,
        'Age': A_index_list,
        'Colour': C_index_list,
        'Vote': V_index_list
    }

    # Reading Data
    better_samples = pd.read_csv('poll-data.csv')

    # Taking full joint on data
    print("\nTaking the full joint on data, this might take 20-30 seconds...")
    sys.stdout.flush()
    full_joint_df = list(get_full_joint_from_data(better_samples, values)['P'])

    # Topologically  Ordered Baeysian Net

    BN_root_model = {
        'Region': ['Vote', 'Jacket'],
        'Gender': ['Jacket', 'Colour'],
        'Handedness': ['Colour'],
        'Jacket': ['Age'],
        'Age': ['Vote'],
        'Colour': ['Vote'],
        'Vote': []
    }

    def bn_2_tup(BN):
        """Save model as a hashable object for comparison in search space."""
        tup_list = []
        for k in sorted(list(BN.keys())):
            for v in BN[k]:
                tup_list.append((k, v))
        return tuple(tup_list)

    closed_list = []

    # Use KL Divergence to compare the Joint distribution of the current model
    # with the Joint distribution of the data.
    model_joint = get_model_joint(BN_root_model, values, better_samples)
    score = sp.entropy(full_joint_df, qk=list(model_joint['P']))
    closed_list.append(bn_2_tup(BN_root_model))

    # Run multiple epochs to get a good Model consistently.
    max_score = (BN_root_model, score)
    for i in range(7):
        print("Running epoch:", i)
        sys.stdout.flush()
        BN_new = deepcopy(max_score[0])
        for edge in list(itertools.product(*[list(values.keys()), list(values.keys())])):
            rem_add = np.random.choice([True, False], p=[0.5, 0.5])
            if rem_add:
                if edge[1] in BN_new[edge[0]]:
                    # Fixing VcRAC as per question.
                    if edge[1] == 'Vote' and (edge[0] == 'Region' or edge[0] == 'Age' or edge[0] == 'Colour'):
                        continue
                    BN_new[edge[0]].remove(edge[1])
            else:
                if edge[1] not in BN_new[edge[0]]:
                    BN_new[edge[0]].append(edge[1])
                    if cyclic(BN_new):
                        BN_new[edge[0]].remove(edge[1])
                        continue
            if bn_2_tup(BN_new) in closed_list:
                continue
            model_joint = get_model_joint(BN_new, values, better_samples)
            score = sp.entropy(full_joint_df, qk=list(model_joint['P']))
            closed_list.append(bn_2_tup(BN_new))
            print(max_score)
            if score > max_score[1]:
                max_score = (BN_new, score)
    print("Optimal Structure:", max_score)
    return max_score[0]


if __name__ == '__main__':
    good_beysian_model = main()
