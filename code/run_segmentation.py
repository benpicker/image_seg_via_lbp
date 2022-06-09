import sys
import numpy as np
import pandas as pd
import os
import time
from scipy.io import loadmat
from PIL import Image
import skimage
import imageio
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
import json
from helper_functions import bg_fg_predict, get_supix_adj_matrix, get_bg_fg_predicitons, get_GMM_models

dir_path = os.getcwd().removesuffix("/code")


# checks for relevant files. if they exists, it makes file paths
data_path = dir_path + "/data/"
mask_file_name = "scribble-mask.mat"
super_pixel_file_name = "super_pixel_map.mat"
original_image_file_name = "original_image.png"
files = os.listdir(data_path)
assert mask_file_name in files, "The scribble mask isn't in the right location. Needs to be in same folder as the script."
assert super_pixel_file_name in files, "The super pixel map isn't in the right location. Needs to be in same folder as the script."
assert original_image_file_name in files, "The original image isn't in the right location. Needs to be in same folder as the script."
mask_file_path = f"{data_path}/{mask_file_name}"
super_pixel_file_path = f"{data_path}/{super_pixel_file_name}"
original_image_file_path = f"{data_path}/{original_image_file_name}"


# checks for folder to save images
save_dir_name = "algorithm_outputs"
save_dir_path = f"{dir_path}/{save_dir_name}"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

# parameters customizable for different experiments
rho_list = [0.1,1,3,5,7,7.75,9]
K_GMM_list = [5]
rand_seed = 8
print_gmm = True

# original image in RBG color space 
img_rgb = imageio.imread(original_image_file_path)
# mask with scribble to denote foreground and backgorund 
mask = loadmat(mask_file_path)['scribble_mask']
# super pixel labels 
supix_labels = loadmat(super_pixel_file_path)['labels']
# hyper parameter for the edge potentials 

# collects gmm means and covariances for each experiments
gmm_data = {}

# get plots for combos of rhos and K_GMMs 
for rho in rho_list:
    for K_GMM in K_GMM_list: 
        start_time = time.time()
        # get background foreground predictions and convert to segmentation
        bg_fg_pred = get_bg_fg_predicitons(img_rgb, mask, supix_labels,rho, 
                                        K_GMM, max_iter = 100,rand_seed=8)
        img_segmented = bg_fg_predict(supix_labels,bg_fg_pred)

        # plot the segments
        plt.imshow(img_segmented, cmap='Greys')
        plt.title(f"rho = {rho}; K_GMM = {K_GMM}")
        plt.savefig(f"{save_dir_path}/segmented_img_rho_{rho}_K_GMM_{K_GMM}".replace(".","_")+ ".png")

        # adds GMM data to gmm_data for json output 
        gmm_fg, gmm_bg = get_GMM_models(img_rgb,mask,K_GMM,rand_seed)
        fg_dic = {"means" : gmm_fg.means_.tolist(), "covariances" : gmm_fg.covariances_.tolist()}
        bg_dic = {"means" : gmm_bg.means_.tolist(), "covariances" : gmm_bg.covariances_.tolist()}
        experiment_key = f"rho={rho}, K_GMM={K_GMM}"
        experiment_dic = {"fg" : fg_dic, "bg" : bg_dic}
        gmm_data[experiment_key] = experiment_dic
            
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Finished model rho={rho}, K_GMM = {K_GMM}, run_time = {total_time}")

# writes json object to file 
with open(f'{save_dir_path}/gmm_means_and_covariances.json', 'w', encoding='utf-8') as f:
    json.dump(gmm_data, f, ensure_ascii=False, indent=4)


# plots the adjacency matrix for super pixels
adj_supix_mat = get_supix_adj_matrix(supix_labels)

plt.imshow(adj_supix_mat,cmap="Greys")
plt.xlabel("Super Pixel Index")
plt.ylabel("Super Pixel Index")
plt.title("Adjacency Matrix for Super Pixels")
plt.savefig(f"{save_dir_path}/adjacency_matrix_for_super_pixels.png")
print("Finished adjacency matrix plot.")

# plots histogram of number of adjacent neighbors for super pixels
plt.hist(np.sum(adj_supix_mat,axis=1))
plt.title("Histogram of Number of Adjacent Super Pixels")
plt.xlabel("Number of Adjacent Super Pixels")
plt.ylabel("Count")
plt.savefig(f"{save_dir_path}/histogram_of_count_of_adjacent_neighbors.png")
print("Finished plot of histogram of count of adjacent neighbors.")






