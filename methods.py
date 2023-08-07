import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from numba import jit
import numba
import itertools
import scipy
import time
import copy
import random
import gc
from sklearn.ensemble import BaggingRegressor

from scipy.sparse import csc_matrix
import julia # set julia path below
julia.install(julia = r"C:\Users\bliu1\AppData\Local\Programs\Julia-1.7.1\bin\julia.exe")
julia.Julia(runtime=r"C:\Users\bliu1\AppData\Local\Programs\Julia-1.7.1\bin\julia.exe")
from julia import Lasso
from julia import Main


#### Methods for Setup

fused_lasso_def = """
using Lasso
function fused_lasso(theta_bar,alpha)
    fuse1d = fit(FusedLasso,theta_bar,alpha)
    return coef(fuse1d)
end
"""
Main.eval(fused_lasso_def)

def get_tree_matrix_sparse(X,tree1):
    leaf_all = np.where(tree1.tree_.feature < 0)[0]
    leaves_index = tree1.apply(X.values)
    leaves = np.unique(leaves_index)
    values = np.ndarray.flatten(tree1.tree_.value)
    leaves_values = [values[i] for i in leaves_index]
    df = pd.DataFrame(np.column_stack((range(0,len(leaves_index)),leaves_index,leaves_values))
             ,columns = ['instance','node','value'])
    setdiff = list(set(leaf_all) - set(np.unique(leaves_index)))
    toadd = pd.DataFrame(np.column_stack((np.zeros(len(setdiff)),setdiff,np.zeros(len(setdiff)))),
                        columns = ['instance','node','value'])
    df = df.append(toadd)
    matrix_temp = pd.pivot_table(df, index = 'instance',columns = 'node',values = 'value').fillna(0)
    return csc_matrix(matrix_temp.values), matrix_temp.columns.values

def get_node_harvest_matrix_sparse(X,tree_list):
    matrix_full, nodes = get_tree_matrix_sparse(X,tree_list[0])
    node_list = [nodes]
    for tree1 in tree_list[1:]:
        matrix_temp, nodes = get_tree_matrix_sparse(X,tree1)
        node_list.append(nodes)
        matrix_full = scipy.sparse.hstack([matrix_full,matrix_temp])
    return matrix_full.tocsc(), node_list


### Optimization Helper Methods

@jit(nopython = True)
def compute_residual(y,y_full,M_temp,w_temp):
    return y - (y_full - M_temp@w_temp)

@jit(nopython = True)
def evaluate_gradient(y,M,w):
    diff = y - M@w
    return -np.transpose(M)@diff

@jit(nopython = True)
def soft_threshold(coef,alpha_lasso):
    temp = np.abs(coef) - alpha_lasso
    temp[temp<0] = 0
    return np.multiply(np.sign(coef),temp)

@jit(nopython = True)
def MCplus_threshold(coef,alpha_lasso,gamma, L):
    gamma1 = gamma/L
    c1 = np.abs(coef) <= alpha_lasso*gamma1
    c2 = np.abs(coef) > alpha_lasso*gamma1
    temp = np.zeros(len(coef))
    temp[c1] = soft_threshold(coef[c1],alpha_lasso/L)*(gamma1*L)/(gamma1*L-1)
    temp[c2] = coef[c2]
    return temp

@jit(nopython = True)
def l2_norm(r):
    return r@r

@jit(nopython = True)
def l1_loss(w):
    return np.sum(np.abs(w))

@jit(nopython = True)
def MCplus_loss(w,alpha_lasso,gamma):
    c1 = np.abs(w) <= alpha_lasso*gamma
    c2 = np.abs(w) > alpha_lasso*gamma
    t1 = np.sum(alpha_lasso*(np.abs(w[c1] - w[c1]**2/(2*alpha_lasso*gamma))))
    t2 = np.sum(c2)*0.5*gamma*alpha_lasso**2
    return t1  + t2


def l1_fuse_loss(w,fused_ind):
    diff = [np.abs(w[j] - w[j-1]) for j in fused_ind]
    return sum(diff)

### Selection Rule Helper Methods

def evaluate_full_gradient(y,M,w):
    diff = y - M@w
    return -np.transpose(M)@diff

@jit(nopython = True)
def BGS_selection(gradient,blocks):
    gradient_l2 = np.array([l2_norm(gradient[blocks[i]:blocks[i+1]]) for i in range(len(blocks)-1)])
    ind = np.argmax(gradient_l2)
    return ind

@jit(nopython = True)
def shrinkage_operator(x,lambda1):
    cond = (np.abs(x)>=lambda1)
    shrunk_vector = np.zeros(len(x))
    shrunk_vector[cond] = x[cond] - np.sign(x[cond])*lambda1
    return(shrunk_vector)

def l1_steepest_direction(y,M,w,alpha_lasso):
    gradient = evaluate_full_gradient(y,M,w)
    steepest_vector = np.zeros(len(w))
    steepest_vector[w!=0] = gradient[w!=0] + np.sign(w[w!=0])*alpha_lasso
    steepest_vector[w==0] = shrinkage_operator(gradient[w==0],alpha_lasso)
    return steepest_vector

def MCP_steepest_direction(y,M,w,alpha_lasso,gamma, L_list):
    gradient = evaluate_full_gradient(y,M,w)
    steepest_vector = np.zeros(len(w))

    cond1 = ((w!=0) & (np.abs(w)<=alpha_lasso*gamma/L_list))
    cond2 = (np.abs(w) > alpha_lasso*gamma/L_list)

    steepest_vector[cond2] = gradient[cond2]
    steepest_vector[cond1] = gradient[cond1] + np.sign(w[cond1])*alpha_lasso - w[cond1]/(gamma/L_list[cond1])
    steepest_vector[w==0] = shrinkage_operator(gradient[w==0],alpha_lasso)
    return steepest_vector

def MCP_fuse_steepest_direction(y,M,w,alpha_lasso,gamma,alpha_fuse,breakpoints, L_list):
    gradient = evaluate_full_gradient(y,M,w)
    indicies = np.array(list(range(len(w))))
    dv = np.zeros(len(w))

    #breakpoints 1 is negative
    breakpoints2 = breakpoints[:-1].copy()
    breakpoints1 = breakpoints2 + 1
    breakpoints1 = np.append(breakpoints1,indicies[0])
    nonbreakpoints = np.array(list(set(indicies) - set(np.append(breakpoints1,breakpoints2))))

    Dw = np.append(0,np.diff(w))
    Dw[breakpoints[:-1]] = 0
    Dw1 = np.append(Dw[1:],0)


    #### Update to nonbreakpoints

    cond1 = nonbreakpoints[( (w[nonbreakpoints]!=0) & \
                            (np.abs(w[nonbreakpoints])<=alpha_lasso*gamma/L_list[nonbreakpoints]) ) \
                           &(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] != 0)]

    cond2 = nonbreakpoints[((w[nonbreakpoints]!=0) & \
                            (np.abs(w[nonbreakpoints])<=alpha_lasso*gamma/L_list[nonbreakpoints]))\
                           &(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] == 0)]

    cond3 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] == 0)]
    cond4 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] != 0)]
    cond5 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] != 0)]

    cond6 = nonbreakpoints[((w[nonbreakpoints]!=0) &\
                            (np.abs(w[nonbreakpoints])<=alpha_lasso*gamma/L_list[nonbreakpoints])) \
                           &(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] == 0)]

    cond7 = nonbreakpoints[((w[nonbreakpoints]!=0) & \
                            (np.abs(w[nonbreakpoints])<=alpha_lasso*gamma/L_list[nonbreakpoints])) \
                           &(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] != 0)]

    cond8 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] == 0)]



    cond9 = nonbreakpoints[(np.abs(w[nonbreakpoints])> alpha_lasso*gamma/L_list[nonbreakpoints])  \
                           &(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] != 0)]
    cond10 = nonbreakpoints[(np.abs(w[nonbreakpoints])> alpha_lasso*gamma/L_list[nonbreakpoints])  \
                           &(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] == 0)]
    cond11 = nonbreakpoints[(np.abs(w[nonbreakpoints])> alpha_lasso*gamma/L_list[nonbreakpoints])  \
                           &(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] == 0)]
    cond12 = nonbreakpoints[(np.abs(w[nonbreakpoints])> alpha_lasso*gamma/L_list[nonbreakpoints])  \
                           &(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] != 0)]


    dv[cond1] = gradient[cond1] \
                    + alpha_fuse*np.sign(Dw[cond1]) \
                    - alpha_fuse*np.sign(Dw1[cond1]) \
                    + alpha_lasso* np.sign(w[cond1]) - w[cond1]/(gamma/L_list[cond1])

    dv[cond2] = shrinkage_operator(gradient[cond2] \
                                       + alpha_lasso*np.sign(w[cond2]) - w[cond2]/(gamma/L_list[cond2]) \
                                        , 2*alpha_fuse)

    dv[cond3] = shrinkage_operator(gradient[cond3] \
                                       + alpha_fuse*np.sign(Dw[cond3])
                                        ,alpha_lasso + alpha_fuse)

    dv[cond4] = shrinkage_operator(gradient[cond4] \
                                       -alpha_fuse*np.sign(Dw1[cond4])
                                        ,alpha_lasso + alpha_fuse)

    dv[cond5] = shrinkage_operator(gradient[cond5] \
                                       + alpha_fuse*np.sign(Dw[cond5]) \
                                       - alpha_fuse*np.sign(Dw1[cond5])
                                        , alpha_lasso)

    dv[cond6] = shrinkage_operator(gradient[cond6] \
                                       + alpha_fuse*np.sign(Dw[cond6]) \
                                       + alpha_lasso*np.sign(w[cond6]) - w[cond6]/(gamma/L_list[cond6])
                                        ,alpha_fuse)

    dv[cond7] = shrinkage_operator(gradient[cond7] \
                                       - alpha_fuse*np.sign(Dw1[cond7]) \
                                       + alpha_lasso*np.sign(w[cond7]) - w[cond7]/(gamma/L_list[cond7])
                                      ,alpha_fuse)

    dv[cond8] = shrinkage_operator(gradient[cond8]
                                      , 2*alpha_fuse + alpha_lasso)

    dv[cond9] = gradient[cond9] \
                    + alpha_fuse*np.sign(Dw[cond9]) \
                    - alpha_fuse*np.sign(Dw1[cond9])

    dv[cond10] = shrinkage_operator(gradient[cond10]
                                        , 2*alpha_fuse)

    dv[cond11] = shrinkage_operator(gradient[cond11] \
                                       + alpha_fuse*np.sign(Dw[cond11]) \
                                        ,alpha_fuse)

    dv[cond12] = shrinkage_operator(gradient[cond12] \
                                       - alpha_fuse*np.sign(Dw1[cond12]) \
                                      ,alpha_fuse)

    #### breakpoints 1
    b1cond1 = breakpoints1[(w[breakpoints1] == 0) & (Dw1[breakpoints1] == 0)]

    b1cond2 = breakpoints1[((w[breakpoints1]!=0) &\
                            (np.abs(w[breakpoints1])<=alpha_lasso*gamma/L_list[breakpoints1]))\
                           & (Dw1[breakpoints1] == 0)]

    b1cond3 = breakpoints1[(w[breakpoints1] == 0) & (Dw1[breakpoints1] != 0)]

    b1cond4 = breakpoints1[((w[breakpoints1]!=0) &\
                            (np.abs(w[breakpoints1])<=alpha_lasso*gamma/L_list[breakpoints1]))\
                           & (Dw1[breakpoints1] != 0)]

    b1cond5 = breakpoints1[(np.abs(w[breakpoints1])> alpha_lasso*gamma/L_list[breakpoints1])\
                           & (Dw1[breakpoints1] == 0)]

    b1cond6 = breakpoints1[(np.abs(w[breakpoints1])> alpha_lasso*gamma/L_list[breakpoints1])\
                           & (Dw1[breakpoints1] != 0)]


    dv[b1cond1] = shrinkage_operator(gradient[b1cond1],
                                         alpha_fuse + alpha_lasso)

    dv[b1cond2] = shrinkage_operator(gradient[b1cond2] \
                                        + alpha_lasso*np.sign(w[b1cond2]) - w[b1cond2]/(gamma/L_list[b1cond2])
                                        ,alpha_fuse)

    dv[b1cond3] = shrinkage_operator(gradient[b1cond3] \
                                        - alpha_fuse*np.sign(Dw1[b1cond3]),
                                        alpha_lasso)

    dv[b1cond4] = gradient[b1cond4] \
                        + alpha_lasso*np.sign(w[b1cond4]) - w[b1cond4]/(gamma/L_list[b1cond4]) \
                        - alpha_fuse*np.sign(Dw1[b1cond4])

    dv[b1cond5] = shrinkage_operator(gradient[b1cond5]
                                        ,alpha_fuse)

    dv[b1cond6] = gradient[b1cond6] \
                        - alpha_fuse*np.sign(Dw1[b1cond6])

    ######### breakpoints2
    b2cond1 = breakpoints2[(w[breakpoints2] == 0) & (Dw[breakpoints2] == 0)]

    b2cond2 = breakpoints2[((w[breakpoints2]!=0) &\
                            (np.abs(w[breakpoints2])<=alpha_lasso*gamma/L_list[breakpoints2]))\
                           & (Dw[breakpoints2] == 0)]

    b2cond3 = breakpoints2[(w[breakpoints2] == 0) & (Dw[breakpoints2] != 0)]

    b2cond4 = breakpoints2[((w[breakpoints2]!=0) &\
                            (np.abs(w[breakpoints2])<=alpha_lasso*gamma/L_list[breakpoints2]))\
                           & (Dw[breakpoints2] != 0)]

    b2cond5 = breakpoints2[(np.abs(w[breakpoints2])> alpha_lasso*gamma/L_list[breakpoints2])\
                           & (Dw[breakpoints2] == 0)]

    b2cond6 = breakpoints2[(np.abs(w[breakpoints2]) > alpha_lasso*gamma/L_list[breakpoints2]) \
                           & (Dw[breakpoints2] != 0)]

    dv[b2cond1] = shrinkage_operator(gradient[b2cond1],
                                            alpha_fuse + alpha_lasso)

    dv[b2cond2] = shrinkage_operator(gradient[b2cond2] \
                                             + alpha_lasso*np.sign(w[b2cond2]) - w[b2cond2]/(gamma/L_list[b2cond2]),
                                            alpha_fuse)

    dv[b2cond3] = shrinkage_operator(gradient[b2cond3] \
                                            + alpha_fuse*np.sign(Dw[b2cond3]),
                                            alpha_lasso)

    dv[b2cond4] = gradient[b2cond4] \
                            + alpha_lasso*np.sign(w[b2cond4]) - w[b2cond4]/(gamma/L_list[b2cond4]) \
                            + alpha_fuse*np.sign(Dw[b2cond4])


    dv[b2cond5] = shrinkage_operator(gradient[b2cond5]
                                            ,alpha_fuse)

    dv[b2cond6] = gradient[b2cond6] \
                            + alpha_fuse*np.sign(Dw[b2cond6])

    return dv


def l1_fuse_steepest_direction(y,M,w,alpha_lasso,alpha_fuse,breakpoints):
    gradient = evaluate_full_gradient(y,M,w)
    dv = np.zeros(len(w))
    indicies = np.array(list(range(len(w))))

    #breakpoints 1 is negative
    breakpoints2 = breakpoints[:-1].copy()
    breakpoints1 = breakpoints2 + 1
    breakpoints1 = np.append(breakpoints1,indicies[0])
    nonbreakpoints = np.array(list(set(indicies) - set(np.append(breakpoints1,breakpoints2))))

    Dw = np.append(0,np.diff(w))
    Dw[breakpoints[:-1]] = 0
    Dw1 = np.append(Dw[1:],0)

    ### nonbreakpoint updates

    cond1 = nonbreakpoints[(w[nonbreakpoints] != 0)&(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] != 0)]
    cond2 = nonbreakpoints[(w[nonbreakpoints] != 0)&(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] == 0)]
    cond3 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] == 0)]
    cond4 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] != 0)]
    cond5 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] != 0)]
    cond6 = nonbreakpoints[(w[nonbreakpoints] != 0)&(Dw[nonbreakpoints] != 0)&(Dw1[nonbreakpoints] == 0)]
    cond7 = nonbreakpoints[(w[nonbreakpoints] != 0)&(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] != 0)]
    cond8 = nonbreakpoints[(w[nonbreakpoints] == 0)&(Dw[nonbreakpoints] == 0)&(Dw1[nonbreakpoints] == 0)]


    dv[cond1] = gradient[cond1] \
                + alpha_fuse*np.sign(Dw[cond1]) \
                - alpha_fuse*np.sign(Dw1[cond1]) \
                + alpha_lasso* np.sign(w[cond1])

    dv[cond2] = shrinkage_operator(gradient[cond2] \
                                   + alpha_lasso*np.sign(w[cond2]) \
                                    , 2*alpha_fuse)

    dv[cond3] = shrinkage_operator(gradient[cond3] \
                                   + alpha_fuse*np.sign(Dw[cond3])
                                    ,alpha_lasso + alpha_fuse)

    dv[cond4] = shrinkage_operator(gradient[cond4] \
                                   -alpha_fuse*np.sign(Dw1[cond4])
                                    ,alpha_lasso + alpha_fuse)

    dv[cond5] = shrinkage_operator(gradient[cond5] \
                                   + alpha_fuse*np.sign(Dw[cond5]) \
                                   - alpha_fuse*np.sign(Dw1[cond5])
                                    , alpha_lasso)

    dv[cond6] = shrinkage_operator(gradient[cond6] \
                                   + alpha_fuse*np.sign(Dw[cond6]) \
                                   + alpha_lasso*np.sign(w[cond6])
                                    ,alpha_fuse)

    dv[cond7] = shrinkage_operator(gradient[cond7] \
                                   - alpha_fuse*np.sign(Dw1[cond7]) \
                                   + alpha_lasso*np.sign(w[cond7])
                                  ,alpha_fuse)

    dv[cond8] = shrinkage_operator(gradient[cond8]
                                  , 2*alpha_fuse + alpha_lasso)


    #### breakpoints 1
    b1cond1 = breakpoints1[(w[breakpoints1] == 0) & (Dw1[breakpoints1] == 0)]
    b1cond2 = breakpoints1[(w[breakpoints1] != 0) & (Dw1[breakpoints1] == 0)]
    b1cond3 = breakpoints1[(w[breakpoints1] == 0) & (Dw1[breakpoints1] != 0)]
    b1cond4 = breakpoints1[(w[breakpoints1] != 0) & (Dw1[breakpoints1] != 0)]

    dv[b1cond1] = shrinkage_operator(gradient[b1cond1],
                                     alpha_fuse + alpha_lasso)

    dv[b1cond2] = shrinkage_operator(gradient[b1cond2] \
                                    + alpha_lasso*np.sign(w[b1cond2]),
                                    alpha_fuse)

    dv[b1cond3] = shrinkage_operator(gradient[b1cond3] \
                                    - alpha_fuse*np.sign(Dw1[b1cond3]),
                                    alpha_lasso)

    dv[b1cond4] = gradient[b1cond4] \
                    + alpha_lasso*np.sign(w[b1cond4]) \
                    - alpha_fuse*np.sign(Dw1[b1cond4])

    #### breakpoints 2
    b2cond1 = breakpoints2[(w[breakpoints2] == 0) & (Dw[breakpoints2] == 0)]
    b2cond2 = breakpoints2[(w[breakpoints2] != 0) & (Dw[breakpoints2] == 0)]
    b2cond3 = breakpoints2[(w[breakpoints2] == 0) & (Dw[breakpoints2] != 0)]
    b2cond4 = breakpoints2[(w[breakpoints2] != 0) & (Dw[breakpoints2] != 0)]

    dv[b2cond1] = shrinkage_operator(gradient[b2cond1],
                                    alpha_fuse + alpha_lasso)

    dv[b2cond2] = shrinkage_operator(gradient[b2cond2] \
                                     + alpha_lasso*np.sign(w[b2cond2]),
                                    alpha_fuse)

    dv[b2cond3] = shrinkage_operator(gradient[b2cond3] \
                                    + alpha_fuse*np.sign(Dw[b2cond3]),
                                    alpha_lasso)

    dv[b2cond4] = gradient[b2cond4] \
                    + alpha_lasso*np.sign(w[b2cond4]) \
                    + alpha_fuse*np.sign(Dw[b2cond4])
    return dv


### Optimization Functions

def l1_GBCD(M,y,blocks,alpha_lasso,num_prox_iters,threshold,warm_start = []):

    if len(warm_start) == 0:
        w = np.zeros(M.shape[1])
        y_update = np.zeros(len(y))
    else:
        w = warm_start.copy()
        y_update = M@w

    nblocks = len(blocks)-1
    iter1 = 0
    converged = False
    loss_sequence = []
    L_dict = {}

    while converged == False:
        gradient = l1_steepest_direction(y,M,w,alpha_lasso)
        ind = BGS_selection(gradient,blocks)
        ind_start = blocks[ind]
        ind_end = blocks[ind+1]

        w_temp = w[ind_start:ind_end]
        M_temp = np.array(M[:,ind_start:ind_end].todense())

        y_temp = compute_residual(y,y_update,M_temp,w_temp)

        if ind in L_dict:
            L = L_dict[ind]
        else:
            L = scipy.sparse.linalg.eigsh(np.transpose(M_temp)@M_temp,
                k = 1, which = 'LM', return_eigenvectors  = False)[0]
            L_dict[ind] = L

        for _ in range(num_prox_iters):
            theta_bar = w_temp - (1/L)*evaluate_gradient(y_temp,M_temp,w_temp)
            w_temp = soft_threshold(theta_bar,alpha_lasso/L)

        w[ind_start:ind_end] = w_temp
        r = y_temp - M_temp@w_temp #trick to avoid computing M@w, y_temp differences out original
        y_update = y - r # another trick to avoid computing the full M@w on the residual step

        loss = (l2_norm(r)/2 + alpha_lasso*l1_loss(w))/len(r)
        loss_sequence.append(loss)

        if len(loss_sequence) >= 2:
            converged = np.abs(loss_sequence[-1]-loss_sequence[-2])<= threshold

        iter1 = iter1 + 1

    return w, iter1, loss_sequence

def MCP_GBCD(M,y,blocks,alpha_lasso,gamma,num_prox_iters,threshold,warm_start = []):

    if len(warm_start) == 0:
        w = np.zeros(M.shape[1])
        y_update = np.zeros(len(y))
    else:
        w = warm_start.copy()
        y_update = M@w

    nblocks = len(blocks)-1

    L_dict = {}
    L_list = np.zeros(M.shape[1])
    for ind1 in range(0,nblocks):
        ind_start = blocks[ind1]
        ind_end = blocks[ind1+1]

        M_temp = np.array(M[:,ind_start:ind_end].todense())
        L = scipy.sparse.linalg.eigsh(np.transpose(M_temp)@M_temp,
                    k = 1, which = 'LM', return_eigenvectors  = False)[0]
        L_dict[ind1] = L
        L_list[ind_start:ind_end] = L

    iter1 = 0
    converged = False
    loss_sequence = []
    ind = 0
    mcp_losses = np.zeros(nblocks)

    while (converged == False):
        gradient = MCP_steepest_direction(y,M,w, alpha_lasso,gamma,L_list)
        ind = BGS_selection(gradient,blocks)

        ind_start = blocks[ind]
        ind_end = blocks[ind+1]

        w_temp = w[ind_start:ind_end]
        M_temp = np.array(M[:,ind_start:ind_end].todense())

        y_temp = compute_residual(y,y_update,M_temp,w_temp)
        L = L_dict[ind]

        for _ in range(num_prox_iters):
                theta_bar = w_temp - (1/L)*evaluate_gradient(y_temp,M_temp,w_temp)
                w_temp = MCplus_threshold(theta_bar,alpha_lasso,gamma,L)

        w[ind_start:ind_end] = w_temp
        r = y_temp - M_temp@w_temp #trick to avoid computing M@w, y_temp differences out original
        y_update = y - r # another trick to avoid computing the full M@w on the residual step

        mcp_losses[ind] = MCplus_loss(w_temp,alpha_lasso,gamma/L) # store regularizer loss
        loss = (l2_norm(r)/2  + np.sum(mcp_losses))/len(r)
        loss_sequence.append(loss)

        if len(loss_sequence) >= 2:
            converged = np.abs(loss_sequence[-1]-loss_sequence[-2])<= threshold
        iter1 = iter1 + 1

    return w, iter1, loss_sequence

def l1_fuse_GBCD(M,y,blocks,alpha_lasso,alpha_fuse,breakpoints, num_prox_iters,threshold, warm_start = []):
    fused_ind = np.array(list(set(range(1,M.shape[1])) - set(breakpoints+1)))
    if len(warm_start) == 0:
        w = np.zeros(M.shape[1])
        y_update = np.zeros(len(y))
    else:
        w = warm_start.copy()
        y_update = M@w

    nblocks = len(blocks)-1
    iter1 = 0
    converged = False
    loss_sequence = []
    L_dict = {}

    while converged == False:

        gradient = l1_fuse_steepest_direction(y,M,w,alpha_lasso,alpha_fuse,breakpoints)
        ind = BGS_selection(gradient,blocks)

        ind_start = blocks[ind]
        ind_end = blocks[ind+1]

        w_temp = w[ind_start:ind_end]
        M_temp = np.array(M[:,ind_start:ind_end].todense())
        y_temp = compute_residual(y,y_update,M_temp,w_temp)

        if ind in L_dict:
            L = L_dict[ind]
        else:
            L = scipy.sparse.linalg.eigsh(np.transpose(M_temp)@M_temp,
                k = 1, which = 'LM', return_eigenvectors  = False)[0]
            L_dict[ind] = L

        for _ in range(num_prox_iters):
            theta_bar = w_temp - (1/L)*evaluate_gradient(y_temp,M_temp,w_temp)
            coef = Main.fused_lasso(theta_bar,alpha_fuse/L)
            if alpha_lasso == 0:
                w_temp = coef
            else:
                w_temp = soft_threshold(coef,alpha_lasso/L)

        w[ind_start:ind_end] = w_temp
        r = y_temp - M_temp@w_temp #trick to avoid computing M@w, y_temp differences out original
        y_update = y - r # another trick to avoid computing the full M@w on the residual step
        loss = (l2_norm(r)/2 + alpha_lasso*l1_loss(w) + alpha_fuse*l1_fuse_loss(w,fused_ind))/len(r)
        loss_sequence.append(loss)

        if len(loss_sequence) >= 2:
            converged = np.abs(loss_sequence[-1]-loss_sequence[-2])<= threshold

        iter1 = iter1 + 1

    return w, iter1, loss_sequence


def MCP_fuse_GBCD(M,y,blocks,alpha_lasso,gamma,alpha_fuse,breakpoints, num_prox_iters,threshold, warm_start = []):

    fused_ind = np.array(list(set(range(1,M.shape[1])) - set(breakpoints+1)))
    if len(warm_start) == 0:
        w = np.zeros(M.shape[1])
        y_update = np.zeros(len(y))
    else:
        w = warm_start.copy()
        y_update = M@w

    nblocks = len(blocks)-1

    L_dict = {}
    L_list = np.zeros(M.shape[1])

    for ind1 in range(0,nblocks):
        ind_start = blocks[ind1]
        ind_end = blocks[ind1+1]

        M_temp = np.array(M[:,ind_start:ind_end].todense())
        L = scipy.sparse.linalg.eigsh(np.transpose(M_temp)@M_temp,
                    k = 1, which = 'LM', return_eigenvectors  = False)[0]
        L_dict[ind1] = L
        L_list[ind_start:ind_end] = L

    iter1 = 0
    converged = False
    loss_sequence = []
    mcp_losses = np.zeros(nblocks)

    while converged == False:

        dv = MCP_fuse_steepest_direction(y,M,w,alpha_lasso,gamma,alpha_fuse,breakpoints, L_list)
        ind = BGS_selection(dv,blocks)
        ind_start = blocks[ind]
        ind_end = blocks[ind+1]

        w_temp = w[ind_start:ind_end]
        M_temp = np.array(M[:,ind_start:ind_end].todense())
        y_temp = compute_residual(y,y_update,M_temp,w_temp)

        L = L_dict[ind]
        for _ in range(num_prox_iters):
            theta_bar = w_temp - (1/L)*evaluate_gradient(y_temp,M_temp,w_temp)
            coef = Main.fused_lasso(theta_bar,alpha_fuse/L)
            if alpha_lasso == 0:
                w_temp = coef
            else:
                w_temp = MCplus_threshold(coef,alpha_lasso,gamma, L)

        w[ind_start:ind_end] = w_temp
        r = y_temp - M_temp@w_temp #trick to avoid computing M@w, y_temp differences out original
        y_update = y - r # another trick to avoid computing the full M@w on the residual step

        mcp_losses[ind] = MCplus_loss(w_temp,alpha_lasso,gamma/L) # store regularizer loss
        loss = (l2_norm(r)/2  + np.sum(mcp_losses) + alpha_fuse*l1_fuse_loss(w,fused_ind))/len(r)
        loss_sequence.append(loss)

        if len(loss_sequence) >= 2:
            converged = np.abs(loss_sequence[-1]-loss_sequence[-2])<= threshold
        iter1 = iter1 + 1

    return w, iter1, loss_sequence


### Experiment Helper Functions

def analyze_nodes(nodes,w):
    nodes1 = copy.deepcopy(nodes)
    counter = 0
    for i in range(len(nodes1)):
        for j in range(len(nodes1[i])):
            if w[counter] == 0:
                nodes1[i][j] = int(0)
            else:
                nodes1[i][j] = int(1)
            counter = counter+1
    return nodes1

def prune_tree(tree1, inds):
    inds = np.array(inds)
    children_left = tree1.tree_.children_left.copy()
    children_right = tree1.tree_.children_right.copy()

    node_count = tree1.tree_.node_count
    nodes = np.ones(node_count) # main array of trees to prune: 1 on, 0 off

    nodes_off = set(np.where([children_left<0])[1][~inds.astype(bool)])
    nodes[list(nodes_off)] = 0
    while True:
        left_off = [i for i, e in enumerate(children_left) if e in nodes_off]
        right_off = [i for i, e in enumerate(children_right) if e in nodes_off]
        children_left[left_off] = -99
        children_right[right_off] = -99
        index_off = np.where((children_left == -99) & (children_right == -99))[0]
        nodes[index_off] = 0
        t1 = len(nodes_off)
        nodes_off.update(list(index_off))
        t2 = len(nodes_off)
        if t1 == t2:
            break
    return nodes

def get_pruned_nnodes(tree_list,nodes,w):
    nodes_eliminated = analyze_nodes(nodes,w)
    finalsizes = []
    originalsizes = []
    for i in range(len(tree_list)):
        tree_nodes = prune_tree(tree_list[i],nodes_eliminated[i])
        finalsizes.append(sum(tree_nodes))
        originalsizes.append(tree_list[i].tree_.node_count)
    return np.sum(finalsizes), np.sum(originalsizes)
