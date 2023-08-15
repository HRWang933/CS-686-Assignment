import numpy as np
from typing import List, Dict


class Factor:
    '''
    A class that represents a factor in the Variable Elimination Algorithm. A factor is a function mapping the combined
    domains of a set of categorical random variables to a real number. The factor's values are represented in 2 ways; feel
    free to use either in your implementation of the Variable Elimination Algorithm methods.

    :attr var_list: List of variable names identifying the random variables constituting this factor
    :attr values: A numpy array containing the factor's values of shape (D_1, D_2, D_3, ... , D_n), where D_i is the size
                  of the domain of the i'th random variable in var_list.
    '''

    def __init__(self, var_list: List[str], values: np.ndarray):
        '''
        Initialize the attributes for this Factor object
        :param var_list: List of integers identifying the random variables constituting this factor\
        :param values: A numpy array containing the factor's values of shape (D_1, D_2, D_3, ... , D_n) as floats,
                       where D_i is the size of the domain of the i'th random variable in var_list.

        For example, suppose you created a factor as follows:
        f1 = Factor(["A", "B"],
                    [[0.4, 0.2, 0.5],
                     [0.1, 0.9, 0.2]])
        Then the values of the factor are represented as the following truth table:

        A         B        Value
        0         0        0.4
        0         1        0.2
        0         2        0.5
        1         0        0.1
        1         1        0.9
        1         2        0.2
        '''

        # Initialize variable list and factor values
        self.var_list = list(var_list)
        self.n_vars = len(self.var_list)
        self.values = np.array(values)

    def has_variable(self, variable: str):
        '''
        Returns True if variable is one of this factor's random variables and False otherwise
        :param variable: The name of the variable to check for
        :return: Boolean indicating if variable is one of this factor's random variables
        '''
        return variable in self.var_list

    def __repr__(self) -> str:
        '''
        Print this factor's random variables and the corresponding values for each random variable assignment
        :return: String representation of a truth table for this factor's values
        '''
        table_str = "\t".join(list(str(var) for var in self.var_list) + ["Value"])
        if np.ndim(self.values) == 0:
            table_str += "\n{:.4f}".format(self.values)
        else:
            truth_table = np.transpose(np.meshgrid(*[np.arange(d) for d in self.values.shape], indexing='ij'),
                                       np.roll(np.arange(self.n_vars + 1), -1)).reshape(-1, self.n_vars)
            flattened_values = self.values.flatten()
            for i in range(len(flattened_values)):
                table_str += "\n" + "\t".join(["{}".format(truth_table[i, n]) for n in range(len(self.var_list))]
                                                + ["{:.4f}".format(flattened_values[i])])
        return table_str

    def __eq__(self, other) -> bool:
        '''
        Returns true if this factor is equal to other
        :param other: object to compare this instance to
        :return: Whether this factor and other have the same variable list and values
        '''
        if isinstance(other, Factor):
            return self.var_list == other.var_list and np.array_equal(self.values, other.values)
        return False





