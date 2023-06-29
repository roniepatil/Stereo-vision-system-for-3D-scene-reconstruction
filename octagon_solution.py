import utilityfunctions as uf
import numpy as np
import cv2
import matplotlib.pyplot as plt


##########################################################################
########################### STEP : 1 : CALIBRATION #######################
##########################################################################
K1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
K2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
baseline = 221.76
f = K1[0,0]
# Read images from curule dataset
images = uf.read_images('./octagon_dataset/')
uf.progressbar("Reading images from octagon_dataset", 0)
# Employ SIFT to extract features and feature descriptors
sift = cv2.SIFT_create()
img0 = images[0].copy()
img1 = images[1].copy()
# Greyscale conversion
grey_img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(grey_img0, None)
kp2, des2 = sift.detectAndCompute(grey_img1, None)
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x :x.distance)
selected_matches = matches[0:100]
uf.progressbar("Extracting key feature points using SIFT", 0.2)
# Match points between two images
sift_matched_pairs = uf.sift_matching_pairs(selected_matches, kp1, kp2)
concat = uf.display_matches_on_images(img0, img1, sift_matched_pairs, [0,0,255], "SIFT_on_octagon.png")
fundamental_matrix, sift_matched_pairs_inliers = uf.obtain_inliers(sift_matched_pairs)
uf.progressbar("Estimating Fundamental Matrix using RANSAC",0.1)
print("Fundamental Matrix: ", fundamental_matrix)
essential_matrix = uf.compute_essential_matrix(K1, K2, fundamental_matrix)
uf.progressbar("Estimating Essential Matrix",0.1)
print("Essential Matrix is: ", essential_matrix)
# Extracting rotation[R] and translation[T] matrix from essential[E] matrix
rotation, translation = uf.compute_camera_pose(essential_matrix)
# Compute 3D points from 4 solution
points3D4 = uf.compute_3D_points(K1, K2, sift_matched_pairs_inliers, rotation, translation)
count_1 = []
count_2 = []
rot = np.identity(3)
tran = np.zeros((3,1))
for i in range(len(points3D4)):
    points3D = points3D4[i]
    points3D = points3D/points3D[3, :]
    # Get only positive z values
    count_2.append(uf.obtain_positive_values(points3D, rotation[i], translation[i]))
    count_1.append(uf.obtain_positive_values(points3D, rot, tran))
count_1 = np.array(count_1)
count_2 = np.array(count_2)
count_threshold = int(points3D4[0].shape[1] / 2)
idx = np.intersect1d(np.where(count_1 > count_threshold), np.where(count_2 > count_threshold))
best_estimate_rot_matrix = rotation[idx[0]]
best_estimate_trans_matrix = translation[idx[0]]
uf.progressbar("Estimating rotation[R] and translation[T] Matrix",0.2)
print("Rotation[R] Matrix is: ", best_estimate_rot_matrix)
print("Translation[T] Matrix is: ", best_estimate_trans_matrix)


##########################################################################
######################### STEP : 2 : RECTIFICATION #######################
##########################################################################
set1, set2 = sift_matched_pairs_inliers[:,0:2], sift_matched_pairs_inliers[:,2:4]
# Display feature points and epipolar lines on unrectified image
lines1, lines2, unrectified_output = uf.compute_epipolar_lines(set1, set2, fundamental_matrix, img0, img1, False)
uf.progressbar("Compute epipolar lines on unrectified image",0.1)
cv2.imwrite("epipolar_lines_on_unrectified_images_octagon_dataset.png", unrectified_output)
# Compute homography matrices
h1, w1 = img0.shape[:2]
h2, w2 = img1.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), fundamental_matrix, imgSize=(w1, h1))
uf.progressbar("Estimating H1 and H2 matrices",0.1)
print("Estimated H1 matrix:", H1)
print("Estimated H2 matrix:", H2)
# Recitify images with homography
rectified_img1 = cv2.warpPerspective(img0, H1, (w1, h1))
rectified_img2 = cv2.warpPerspective(img1, H2, (w2, h2))
rectified_set1 = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
rectified_set2 = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)
# Compute fundamental matrix of rectified images
H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
rectified_F = np.dot(H2_T_inv, np.dot(fundamental_matrix, H1_inv))
uf.progressbar("Compute fundamental matrix of rectified images",0.1)
# Display feature points and epipolar lines on rectified image
rectified_lines1, recrified_lines2, rectified_output = uf.compute_epipolar_lines(rectified_set1, rectified_set2, rectified_F, rectified_img1, rectified_img2, True)
uf.progressbar("Compute epipolar lines on rectified image",0.1)
cv2.imwrite("epipolar_lines_on_rectified_images_octagon_dataset.png", rectified_output)


##########################################################################
######################### STEP : 3 : CORRESPONDANCE ######################
##########################################################################
# Greyscale conversion
grey_1 = cv2.cvtColor(rectified_img1,cv2.COLOR_BGR2GRAY)
grey_2 = cv2.cvtColor(rectified_img2,cv2.COLOR_BGR2GRAY)
# Window size on second image
disparities = 50
# Pixel size of block in first image
block = 10
height, width = grey_1.shape
disparity_img = np.zeros(shape = (height,width))
print("Computing disparity map using SSD")
# Compute disparity map using SSD by comparing matches along epipolar lines
for i in range(block, grey_1.shape[0] - block - 1):
    for j in range(block + disparities, grey_1.shape[1] - block - 1):
        ssd = np.empty([disparities, 1])
        l = grey_1[(i - block):(i + block), (j - block):(j + block)]
        height, width = l.shape
        for d in range(0, disparities):
            r = grey_2[(i - block):(i + block), (j - d - block):(j - d + block)]
            ssd[d] = np.sum((l[:,:]-r[:,:])**2)
        disparity_img[i, j] = np.argmin(ssd)
# Rescale SSD to 0-255
dispairity_computed_img = ((disparity_img/disparity_img.max())*255).astype(np.uint8)
# print("Disparity based on SSD: ", ssd)
# Generate color heatmap
color_map = plt.get_cmap('inferno')
heatmap_img = (color_map(dispairity_computed_img) * 2**16).astype(np.uint16)[:,:,:3]
heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
uf.progressbar("Save disparity in greyscale and color heatmap",0.1)
cv2.imwrite("Disparity_in_greyscale_on_octagon_dataset.png", dispairity_computed_img)
cv2.imwrite("Disparity_in_heatmap_on_octagon_dataset.png", heatmap_img)


##########################################################################
##################### STEP : 4 : COMPUTE DEPTH IMAGE #####################
##########################################################################
# Compute depth of image
depth = np.zeros(shape=grey_1.shape).astype(float)
depth[dispairity_computed_img > 0] = (f * baseline) / (dispairity_computed_img[dispairity_computed_img > 0])
depth_img = ((depth/depth.max())*255).astype(np.uint8)
uf.progressbar("Computing depth of image based on disparity",0.1)
# Generate color heatmap
color_map = plt.get_cmap('inferno')
depth_img_heatmap = (color_map(depth_img) * 2**16).astype(np.uint16)[:,:,:3]
depth_img_heatmap  = cv2.cvtColor(depth_img_heatmap, cv2.COLOR_RGB2BGR)
uf.progressbar("Save depth in greyscale and color heatmap",0.1)
cv2.imwrite("Depth_in_greyscale_on_octagon_dataset.png", depth_img)
cv2.imwrite("Depth_in_heatmap_on_octagon_dataset.png", depth_img_heatmap)


key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()