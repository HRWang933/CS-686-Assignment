from typing import Dict, List
from factor import Factor
import numpy as np
'''
########## READ CAREFULLY #############################################################################################
You should implement all the functions in this file. Do not change the function signatures.
#######################################################################################################################
'''

def restrict(factor: Factor, variable: str, value: int) -> Factor:
    '''
    Restrict a factor by assigning value to variable
    :param factor: a Factor object
    :param variable: the name of the variable to restrict
    :param value: the value to restrict variable to
    :return: a new Factor object resulting from restricting variable. This factor no longer includes variable in its
             var_list.
    '''
    variable_index=factor.var_list.index(variable)
    new_var_list = factor.var_list[:]
    new_var_list.remove(variable)
    new_values=np.take(factor.values,value,variable_index)
    new_factor = Factor(new_var_list,new_values)
    return new_factor

def sort_factor(factor:Factor):
    unsorted_ver_list = factor.var_list[:]
    # print("before sort",end=' ')
    # print(unsorted_ver_list)
    factor.var_list.sort()
    # print("after sort",end=' ')
    # print(factor.var_list)
    var_list_change =[]#new axis list to old axis index
    for str in factor.var_list:
        var_list_change.append(unsorted_ver_list.index(str))
    # print("before sort",end=' ')
    # print(factor.values.shape)
    # print("change list",end=' ')
    # print(var_list_change)
    factor.values=np.transpose(factor.values,var_list_change)
    # print("after sort",end=' ')
    # print(factor.values.shape)
    # print("new factor",end=' ')
    # print(factor.var_list)
    # print(factor.values.shape)

def multiply(factor_a: Factor, factor_b: Factor) -> Factor:
    '''
    Multiply two tests (factor_a and factor_b) together.
    :param factor_a: a Factor object representing the first factor in the multiplication operation
    :param factor_b: a Factor object representing the second factor in the multiplication operation
    :return: a new Factor object resulting from the multiplication of factor_a and factor_b. Note that the new factor's
             var_list is the union of the var_lists of factor_a and factor_b IN ALPHABETICAL ORDER.
    '''
    # print(factor_a.var_list)
    # print(factor_b.var_list)
    # print(factor_a.values.shape)
    # print(factor_b.values.shape)
    new_var_list=list(set(factor_a.var_list+factor_b.var_list))
    new_var_list.sort()
    sort_factor(factor_a)
    sort_factor(factor_b)
    # print(factor_a.var_list)
    # print(factor_b.var_list)
    # print(new_var_list)
    # print(factor_a.values.shape)
    # print(factor_b.values.shape)
    old_a_shape=factor_a.values.shape
    new_a_shape =[]
    for str in new_var_list:
        if str in factor_a.var_list:
            new_a_shape.append(old_a_shape[factor_a.var_list.index(str)])
        else:
            new_a_shape.append(1)
    new_a_values=np.reshape(factor_a.values,new_a_shape)
    old_b_shape=factor_b.values.shape
    new_b_shape =[]
    for str in new_var_list:
        if str in factor_b.var_list:
            new_b_shape.append(old_b_shape[factor_b.var_list.index(str)])
        else:
            new_b_shape.append(1)
    new_b_values=np.reshape(factor_b.values,new_b_shape)
    # print(factor_a.var_list)
    # print(factor_b.var_list)
    # print(new_a_shape)
    # print(new_b_shape)
    new_values= new_a_values*new_b_values
    new_factor = Factor(new_var_list,new_values)
    return new_factor




def sum_out(factor: Factor, variable: str) -> Factor:
    '''
    Sum out a variable from factor.
    :param factor: a Factor object
    :param variable: the name of the variable in factor that we wish to sum out
    :return: a Factor object resulting from performing the sum out operation on factor. Note that this new factor no
             longer includes variable in its var_list.
    '''
    variable_index=factor.var_list.index(variable)
    new_var_list = factor.var_list[:]
    new_var_list.remove(variable)
    new_values=np.sum(factor.values,axis=variable_index)
    new_factor = Factor(new_var_list,new_values)

    return new_factor


def normalize(factor: Factor) -> Factor:
    '''
    Normalize factor such that its values sum to 1.
    :param factor: a Factor object representing the factor to normalize
    :return: a Factor object resulting from performing the normalization operation on factor
    '''

    sum = np.sum(factor.values)
    f = lambda x: x/sum
    new_values= f(factor.values)
    new_factor = Factor(factor.var_list,new_values)
    
    return new_factor


def ve(factor_list: List[Factor], query_variables: List[str], evidence: Dict[str, int], ordered_hidden_variables: List[str], verbose: bool=False) -> Factor:
    '''
    Applies the Variable Elimination Algorithm for input tests factor_list, restricting tests according to the
    evidence in evidence_list, and eliminating hidden variables in the order that they appear in
    ordered_list_hidden_variables. The result is the distribution for the query variables. The query variables are, by
    process of elimination, those for which we do not have evidence for and do not appear in the list of hidden
    variables).
    :param factor_list: a list of Factor objects representing every conditional probability distribution in the
                        Bayesian network
    :param query_variables: a list of variable names corresponding to the query variables
    :param evidence_list: a dict mapping evidence variable names to corresponding values
    :param ordered_list_hidden_variables: a list of names of the hidden variables. Variables are to be eliminated in the
                                          order that they appear in this list.
    :param verbose: Whether to print results of intermediate VEA operations (use for debugging, if you like)
    :return: a Factor object representing the result of executing the Variable Elimination Algorithm for the given
             evidence and ordered list of hidden variables.
    '''
    #restrict
    for observation,value in evidence.items():
        for factor in factor_list:
            if observation in factor.var_list:
                new_factor = restrict(factor,observation,value)
                factor_list[factor_list.index(factor)]=new_factor
    #mul and sum out
    for hidden_variable in ordered_hidden_variables:
        #find factor contain hidden_variable
        mul_index = []
        for i,factor in enumerate(factor_list):
            if hidden_variable in factor.var_list:
                mul_index.append(i)
        #multiply 
        if len(mul_index) > 1:
            for x in range(1,len(mul_index)):
                factor_list[mul_index[0]]=multiply(factor_list[mul_index[0]],factor_list[mul_index[x]])
            #remove multiplied
            factor_list = [i for j, i in enumerate(factor_list) if j not in mul_index[1:]]
        #sum out
        if len(mul_index) > 0:    
            factor_list[mul_index[0]]=sum_out(factor_list[mul_index[0]],hidden_variable)
    #mul to 1
    while (len(factor_list)>1):
        factor_list[0] = multiply(factor_list[0],factor_list[-1])
        factor_list.pop(-1)
    #normalize
    final_factor = normalize(factor_list[0])
    return final_factor