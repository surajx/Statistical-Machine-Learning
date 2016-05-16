import numpy as np
import pandas as pd


def cyclic(self, g):
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


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    Reference: http://stackoverflow.com/a/1235363/3779631
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def get_full_joint_from_data(given_data_df, values_dict):
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
    for comb in cartesian(indices):
        print(comb)
        p = len(given_data_df[
                (given_data_df['Region'] == comb[0]) &
                (given_data_df['Gender'] == comb[1]) &
                (given_data_df['Handedness'] == comb[2]) &
                (given_data_df['Jacket'] == comb[3]) &
                (given_data_df['Age'] == comb[4]) &
                (given_data_df['Colour'] == comb[5]) &
                (given_data_df['Vote'] == comb[6])
                ].index) / total_sample_cnt
        print(p)
        full_joint_df[tuple(indices)] = p
    return full_joint_df


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

    better_samples = pd.read_csv('poll-data.csv')
    joint_df = get_full_joint_from_data(better_samples, values)
    print(joint_df)

if __name__ == '__main__':
    main()
