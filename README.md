# Image Segmentation via Loopy Belief Propogation 

Image segmentation is a classical problem in vision research. Probabilistic graphical models offer a useful way to tackle such problems. In this project, I use loopy belief propagation as a means of segmenting partially labeled images for semi-supervised foreground/background segmentation. 

A few notes to find more information: 

- The project_summary pdf contains an explanation of what's going on in the project. Highly recommended the user read this inconjunction with the code to familiarize themselves. 
- Code can be found in the code file. Run `run_segmentation` to obtain results. User will need to set two parameters `rho` and `K_GMM`. `rho` is a parameter for controlling influence of edge potentials. `K_GMM` sets the number of GMM classes.  
- Data contains the three files needed to run the algorithm 
- The code outputs to a folder algorithm_outputs, where the user can find (1) the GMM model parameters (i.e. means and covariances) in a json format, (2) plots of the segmentations, (3) plot of the adjacency matrix for each super pixel, (4) histogram of number of adjacent nodes for each super pixel

***Original Image***

![Original Image](https://github.com/benpicker/image_seg_via_lbp/blob/main/data/original_image.png)

***Segmentation, with rho=7.75, K_GMM=5***


![rho = 7.75, K_GMM = 5](https://github.com/benpicker/image_seg_via_lbp/blob/main/algorithm_outputs/segmented_img_rho_7_75_K_GMM_5.png)



