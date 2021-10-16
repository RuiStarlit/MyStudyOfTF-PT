# -*- coding: UTF-8 -*-
"""
written by Rui
file:oasis_giveup.py
create time:2021/03/28
"""
import numpy as np
import math
import random


def make_psd(W):
    """ Make matrix positive semi-definite.
        通过使用W[:]可以使得数组里面的元素被修改 """
    w, V = np.linalg.eig(0.5 * (W + W.T))
    D = np.diagflat(np.maximum(w, 0))
    W[:] = np.dot(np.dot(V, D), V.T)

def symmetrize(W):
    """ Symmetrize matrix."""
    W[:] = 0.5 * (W + W.T)


def take_from_struct(options, fieldname, default):
    """
     [VAL, OUT_OPTIONS] = TAKE_FROM_STRUCT(OPTIONS, FIELDNAME, DEFAULT)

    Take values from the options structure, use default if fiekd does
    not exist. Provide meaningful error messages. The function
    also updates the structure when the default is used.
     If default is not given, program aborts if field does not exist.

    Examples:

    1. get the values of n_restarts, use default of 20 if field isnt set
    [n_restarts, options] = take_from_struct(options,'n_restarts',20);

    2. get the values of n_restarts, abort if doesnt exist
    [n_restarts, options] = take_from_struct(options,'n_restarts');

    (C) GAl Chechik, 2004 Software available for academic use. Other
    uses require explicit permission.
    """
    out_options = options
    #?
    #return val,out_options

def oasis_m():
    pass


def oasis(data, class_labels, parms):
    """
    model = oasis(data, class_labels, parms)
    Input:
    -- data         - Nxd sparse matrix (each instance being a ROW)
    -- class_labels - label of each data point  (Nx1 integer vector)
    -- parms (do sym, do_psd, aggress etc.)

    Output:
    -- model.W - dxd matrix
    -- model.loss_steps - a binary vector: was there an update at
     each iterations
    -- modeo.parms, the actual parameters used in the run (inc. defaults)
    Parameters:
    -- aggress: The cutoff point on the size of the correction
          (default 0.1)
    -- rseed: The random seed for data point selection
          (default 1)
    -- do_sym: Whether to symmetrize the matrix every k steps
          (default 0)
    -- do_psd: Whether to PSD the matrix every k steps, including
          symmetrizing them (defalut 0)
    -- do_save: Whether to save the intermediate matrices. Note that
          saving is before symmetrizing and/or PSD in case they exist
          (default 0)
    -- save_path: In case do_save==1 a filename is needed, the
          format is save_path/part_k.mat
    -- num_steps - Number of total steps the algorithm will
          run (default 1M steps)
    -- save_every: Number of steps between each save point
          (default num_steps/10)
    -- sym_every: An integer multiple of "save_every",
          indicates the frequency of symmetrizing in case do_sym=1. The
          end step will also be symmetrized. (default 1)
    -- psd_every: An integer multiple of "save_every",
          indicates the frequency of projecting on PSD cone in case
          do_psd=1. The end step will also be PSD. (default 1)
    -- use_matlab: Use oasis_m.m instead of oasis_c.c
       This is provided in the case of compilation problems.
    """
    data = np.array(data)
    N, dim = data.shape
    W = np.eye(dim)
    #data.sort()
    classes = sorted(list(set(class_labels)))
    num_classes = len(classes)
    # Translate class labels to serial numbers 1,2,....
    class_labels = np.array(class_labels)
    new_class_lables = np.zeros(class_labels.shape)
    for i in range(len(classes)):
        new_class_lables[class_labels == classes[i]] = i
    class_labels = new_class_lables.copy()
    class_sizes = np.zeros(num_classes, 1)
    class_start = np.zeros(num_classes, 1)
    for k in range(num_classes):
        class_sizes[k] = sum(class_labels == k)
        class_start[k] = ( (class_labels==k) != 0).argmax(axis=0)
    # Initialize
    aggress = take_from_struct(parms, 'aggress', 0.1)
    rseed = take_from_struct(parms, 'rseed', 1)
    # num_steps = take_from_struct(parms, 'num_steps', 10 ^ 6); revised by eva
    num_steps = take_from_struct(parms, 'num_steps', 10 ^ 5)
    do_sym = take_from_struct(parms, 'do_sym', 0)
    sym_every = take_from_struct(parms, 'sym_every', 1)
    do_psd = take_from_struct(parms, 'do_psd', 0)
    psd_every = take_from_struct(parms, 'psd_every', 1)
    do_save = take_from_struct(parms, 'do_save', 1)
    save_every = take_from_struct(parms, 'save_every', math.ceil( num_steps / 10))
    # save_path = take_from_struct(parms, 'save_path', fullfile(pwd, 'oasis_saves'))
    use_matlab = take_from_struct(parms, 'use_matlab', 0)
    # 新建文件夹 (save_path)
    random.random() #？

    # Optimize
    loss_steps = np.zeros(1,num_steps)
    num_batches = math.ceil(num_steps / save_every)
    steps_vec = np.ones(1,num_batches-1)*save_every
    steps_vec[steps_vec.shape[0]+1] = num_steps - (num_batches-1)*save_every;
    # 写入文件 fprintf('num_batches = %d, total num_steps = %d\n', num_batches, num_steps);

    #把稀疏矩阵data转化为正常的矩阵data numpy好像没有稀疏矩阵，考虑调用OASIS之前就处理好data
    for i_batch in range(num_batches):
        # fprintf('num_batches = %d, total num_steps = %d\n', num_batches, num_steps);
        W, loss_steps_batch, los = oasis_m(W, data, class_labels, class_start, class_sizes, \
                                          steps_vec[i_batch], aggress)
        #fprintf('\n')
        loss_steps_batch[loss_steps_batch.shape[0]:save_every] = 0
        loss_steps[(i_batch-1)*save_every+1:i_batch*save_every] = loss_steps_batch
        if do_save:
            pass
            # 保存
        if do_sym:
            if i_batch%sym_every == 0 or i_batch == num_batches:
                symmetrize(W)
        if do_psd:
            if i_batch % psd_every == 0 or i_batch == num_batches:
                make_psd(W)
        #model类
    """
    model.W = W; 
    model.loss_steps = loss_steps;
    model.num_loss_steps = num_batches*save_every;
    model.parms = parms;
    model.los = los;
    """
    # return model



