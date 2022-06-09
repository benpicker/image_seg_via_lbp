import sys
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from PIL import Image
import skimage
import imageio
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
from matplotlib import colors


def get_super_image(img_cieluv,supix_labels,adj_supix_mat):
    """
    An array of colors for each super pixel, where each color is the average 
    color in each dimension of CIELUV space for all of the pixels associated 
    with that particular super pixel. 

    Arguments:
        img_cieluv (ndarray) : original image in CIELUV color space
        supix_labels (ndarray) : super pixel ids for each pixel 
        adj_supix_mat (ndarray) : adjacency matrix between super pixels  
    Return: 
        img_super (ndarray): average color for each super pixels
                                rows : number of super pixels 
                                cols : number of colors 
    """
    img_super = np.zeros((adj_supix_mat.shape[0],3), dtype=np.float32)
    for i in range(len(img_super)):
        idx = np.where(supix_labels==i)
        pix = img_cieluv[idx[0], idx[1],:]
        img_super[i,:] = np.mean(pix, axis=0)
    return img_super


def bg_fg_predict(supix_labels,bg_fg_pred):
    """
    For each super pixel, it applies super pixel bg/fg label to every pixel 
    assigned to a given super pixel and then returns the image, where 1 is 
    foreground and 0 is background 

    Arguments: 
        supix_labels (ndarray) : super pixel labels for each pixel of image 
        bg_fg_pred (ndarray) : super pixbel bg/fg labels for each super pixel
    Returns: 
        img_output (ndarray) : the bg/fg class for each pixel
    """
    H,W = supix_labels.shape
    img_output = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            img_output[i,j] = bg_fg_pred[supix_labels[i,j]]
    return img_output


def get_supix_adj_matrix(supix_labels):
    """
    Creates a super pixel adjacency matrix for super pixel labels of an image. 

    Argument: 
        supix_labels (ndarray) : super pixel labels for each pixel of image 
    Returns: 
        adj_supix_mat (ndarray) : superpixel adjacency matrix
    """
    H,W = supix_labels.shape
    pad_val = -1 
    n_supix = np.max(supix_labels) + 1
    
    # padded version of the super pixel labels to handle border cases 
    padded_supix_labels = np.full((H+2,W+2),pad_val)
    padded_supix_labels[1:H+1, 1:W+1] = supix_labels

    # loop over non-pad pixels to fill adjacency matrix
    adj_supix_mat = np.zeros((n_supix,n_supix))
    for i in range(1,H+1): 
        for j in range(1,W+1):  
            # idxs of adjacent nodes to current node i,j 
            adj_nodes = [
                        (i+1,j),   # top middle
                        (i,j-1),   # middle left 
                        (i, j+1),  # middle right 
                        (i-1,j),   # bottom middle 
            ]  

            # super pixel ids for neighboring nodes, 
            # excluding node (i,j) and padding  
            adj_supixs = padded_supix_labels[[neighbor[0] for neighbor in adj_nodes], 
                                             [neighbor[1] for neighbor in adj_nodes]]
            adj_supixs = [supix for supix in adj_supixs if supix not in [pad_val,padded_supix_labels[i,j]]]

            # add edge to adjacency matrix for adjacent nodes 
            adj_supix_mat[padded_supix_labels[i,j],adj_supixs] = 1
            
    return adj_supix_mat


def get_message_update(log_m_fg, log_m_bg, b_fg, b_bg, rho): 
    """
    Updates the messages according to the sum-product message passing 
    algorithm from Koller, Algorithm 11.1. The messages are then used to update
    the probabilities of the hidden states. 

    Arguments: 
        b_fg (ndarray) : initial array of log node potentials for foreground
        b_bg (ndarray) : initial array of log node potentials for background
        log_m_fg (ndarray) : array of messages for foreground
        log_m_bg (ndarray) : array of messages for background
        rho (float) : parameter that determines strength of edge potentials 
    Returns: 
        new_log_m_fg (ndarray) : update array of messages for foreground 
        new_log_m_bg (ndarray) : update array of messages for background 
    """

    # sum of log messages for each super pixel for foreground and background
    # and initial potential  
    # (step 1 of SP-Message from Koller, Algorithm 11.1)
    b_fg += log_m_fg.sum()
    b_bg += log_m_bg.sum()
    
    # unnormalized joint distribution over super pixel observables (Y) and 
    # hidden states (X). Intuitively, it is the exponentiated sum of log node 
    # potentials and edge potentials for foreground and background. 
    # (step 2 of SP-Message from Koller, Algorithm 11.1)
    m_fg_pass = np.exp(b_fg) + np.exp(b_bg - rho)
    m_bg_pass = np.exp(b_bg) + np.exp(b_fg - rho)

    # to create a probability distribution, we normalize over the potentials. 
    # rescaling to makes it a Gibbs distribution.
    scale = m_fg_pass + m_bg_pass
    new_log_m_fg = np.log(m_fg_pass)-np.log(scale)
    new_log_m_bg = np.log(m_bg_pass)-np.log(scale)

    return new_log_m_fg, new_log_m_bg


def get_GMM_models(img_rgb,mask,K_GMM,rand_seed):
    """
    Creates sklearn GMM models for foreground and background.  

    Arguments: 
        img_rgb (ndarray) : the original image in RGB color space 
        mask (ndarray) : mask with scribble to denote foreground and backgorund 
        K_GMM (int) : the number of GMM classes for initializing the beliefs 
                      (i.e. the initial probabilities) 
        rand_seed (int) : sets seed for GaussianMixture class  
    Returns: 
        gmm_fg (GaussianMixture) : a GMM of the foreground 
        gmm_bg (GaussianMixture) : a GMM of the background 
    """
    # converting RBG to CIELUV
    img_cieluv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)

    # data points corresponding to the mask scribbles 
    data_fg = img_cieluv[np.where(mask==1)[0],np.where(mask==1)[1],:]
    data_bg = img_cieluv[np.where(mask==2)[0],np.where(mask==2)[1],:]
    
    # make GMM models 
    model = GaussianMixture(n_components=K_GMM, covariance_type='diag',random_state=rand_seed)
    gmm_fg = model.fit(data_fg)
    model2 = GaussianMixture(n_components=K_GMM, covariance_type='diag',random_state=rand_seed)
    gmm_bg = model2.fit(data_bg)
    return gmm_fg,gmm_bg


def initialize_b_and_m(img_super,gmm_fg,gmm_bg):
    '''
    Provides initial values for the messages and beliefs. Equivalent to Koller's
    initialize_cgraph. 
    
    Argument: 
        img_super (ndarray) : average color for each super pixels
    Returns: 
        b_fg (ndarray) : initial array of log node potentials for foreground
        b_bg (ndarray) : initial array of log node potentials for background
        log_m_fg (ndarray) : initial array of messages for foreground
        log_m_bg (ndarray) : initial array of messages for background
    '''
    n_supix = len(img_super)
    
    # initialize beliefs as log probabilities from GMMs 
    b_fg = gmm_fg.score_samples(img_super)
    b_bg = gmm_bg.score_samples(img_super)

    # initializing messages 
    log_m_fg = np.zeros((n_supix,n_supix))
    log_m_bg = np.zeros((n_supix,n_supix))
    return b_fg, b_bg, log_m_fg, log_m_bg 


def get_bg_fg_predicitons(img_rgb, mask, supix_labels, rho, K_GMM, max_iter, rand_seed=0 , tol = 1e-5):

    """
    Finds the bg/fg labels for the image segmentation task using 
    Koller's cgraph_sp_calibrate algorithm. 

    Arguments:
        img_rgb (ndarray): original image in RGB color space 
        mask (ndarray) : mask with scribble to denote foreground and backgorund 
        supix_labels (ndarray) : super pixel ids for each pixel 
        rho (float) : parameter that determines strength of edge potentials 
        K_GMM (int) : the number of GMM classes for initializing the beliefs 
                      (i.e. the initial probabilities) 
        max_iter (int) : maximum iterations for updates
        rand_seed (int) : seed for GaussianMixture class  
        tol (float) : criterion for convergence in log messages updates
    Return: 
        bg_fg_pred (ndarray) : prediction for foreground/background, where
                                0 is background and 1 is foreground
    """
    # converting RBG to CIELUV
    img_cieluv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)

    # get adjacency matrix 
    adj_supix_mat = get_supix_adj_matrix(supix_labels)

    # get super pixel image (i.e. Y_i's)
    img_super = get_super_image(img_cieluv,supix_labels,adj_supix_mat)

    # initialize beliefs and messages 
    gmm_fg,gmm_bg = get_GMM_models(img_rgb,mask,K_GMM,rand_seed)
    b_fg, b_bg, log_m_fg, log_m_bg = initialize_b_and_m(img_super,gmm_fg,gmm_bg)
    edges = np.asarray(np.where(adj_supix_mat == 1)).T
    num_edges = edges.shape[0]

    # belief propogation to get updated messages 
    t = 0 
    max_delta = tol + 1
    while (t < max_iter) and (max_delta >= tol): 
        max_delta = 0
        for i, edge in enumerate(edges):
            
            # index for sending super pixel subgraph 
            node_i = edge[0] 
            # index for receiving super pixel subgraph 
            node_j = edge[1] 
            # super pixel indices which are neighbors of node i, 
            # excluding the super pixel index for node j
            neighbors_i_not_j = edges[:,0][(edges[:,1] == node_i) & (edges[:,0] != node_j)]


            log_m_fg_not_j = np.array([log_m_fg[node][node_i] for node in neighbors_i_not_j])
            log_m_bg_not_j = np.array([log_m_bg[node][node_i] for node in neighbors_i_not_j])

 
            # get old log_m_i->j and unaltered beta_i 
            b_fg_i = b_fg[node_i]
            b_bg_i =  b_bg[node_i]
            log_m_fg_old = log_m_fg[node_i][node_j]
            log_m_bg_old = log_m_bg[node_i][node_j]

        
            #  get new log_m_i->j 
            log_m_fg_new, log_m_bg_new = get_message_update(log_m_fg_not_j, 
                                                    log_m_bg_not_j, 
                                                    b_fg_i, 
                                                    b_bg_i, 
                                                    rho)
            

            # update messages stored 
            log_m_fg[node_i][node_j] = log_m_fg_new
            log_m_bg[node_i][node_j] = log_m_bg_new


            # check if difference small enough we have convergence 
            dif_fg = abs(log_m_fg_old-log_m_fg_new)
            dif_bg = abs(log_m_bg_old-log_m_bg_new)
            delta = max(dif_fg,dif_bg)
            if delta > max_delta:
                max_delta = delta

        t += 1

    # number of super pixels 
    n_supix = len(img_super)
    # calibrate the beliefs and return bg/fg labels 
    # (i.e. update the distribtution now that the messages have converged)  
    bg_fg_pred = np.zeros(n_supix)
    for node_i in range(n_supix):
        neighbors_i = edges[:,0][edges[:,1] == node_i]
        b_fg_cal_i = b_fg[node_i]
        b_bg_cal_i = b_bg[node_i]
        # add all the messages to the original beliefs to update the beliefs  
        for node in neighbors_i:
            b_fg_cal_i += log_m_fg[node][node_i]
            b_bg_cal_i += log_m_bg[node][node_i]
        # determine bg/fg label 
        bg_fg_pred[node_i] = 1 if b_fg_cal_i > b_bg_cal_i else 0
    return bg_fg_pred