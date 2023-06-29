import numpy as np
import cv2
from time import sleep
from tqdm import tqdm

def progressbar(text, t):
    for i in tqdm(range(10),text):
        sleep(t)

def read_images(dataset_folder):
    list_of_images = []
    for i in range(0,2):
        path = dataset_folder + "/" + "im" + str(i) + ".png"
        image = cv2.imread(path)
        list_of_images.append(image)
    return list_of_images

def sift_matching_pairs(matches, kp1, kp2):
    list_of_matched_pairs = []
    for i, m in enumerate(matches):
        point1 = kp1[m.queryIdx].pt
        point2 = kp2[m.trainIdx].pt
        list_of_matched_pairs.append([point1[0], point1[1], point2[0], point2[1]])
    list_of_matched_pairs = np.array(list_of_matched_pairs).reshape(-1,4)
    return list_of_matched_pairs

def normalize(ab):
    ab_dash = np.mean(ab, axis=0)
    a_dash, b_dash = ab_dash[0], ab_dash[1]
    a_cap = ab[:,0] - a_dash
    b_cap = ab[:,1] - b_dash
    s = (2/np.mean(a_cap**2 + b_cap**2))**(0.5)
    T_scale = np.diag([s, s, 1])
    T_trans = np.array([[1, 0, -a_dash],[0, 1, -b_dash],[0, 0, 1]])
    T = T_scale.dot(T_trans)
    xim = np.column_stack((ab, np.ones(len(ab))))
    x_n = (T.dot(xim.T)).T
    return x_n, T

def compute_fundamental_matrix(matches):
    x1 = matches[:,0:2]
    x2 = matches[:,2:4]
    if(x1.shape[0] > 7):
        x1_norm, T1 = normalize(x1)
        x2_norm, T2 = normalize(x2)
        A = np.zeros((len(x1_norm), 9))
        for i in range(0, len(x1_norm)):
            x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2, y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])
        U, S, V = np.linalg.svd(A, full_matrices=True)
        F = V.T[:, -1]
        F = F.reshape(3, 3)
        u, s, v = np.linalg.svd(F)
        s = np.diag(s)
        s[2, 2] = 0.0
        F = np.dot(u, np.dot(s, v))
        F = np.dot(T2.T, np.dot(F, T1))
        return F
    else:
        return None

def error_of_F(feature, F):
    x1, x2 = feature[0:2], feature[2:4]
    x1_temp = np.array([x1[0], x1[1], 1]).T
    x2_temp = np.array([x2[0], x2[1], 1])
    error = np.dot(x1_temp, np.dot(F, x2_temp))
    return np.abs(error)

def obtain_inliers(features):
    n_iterations = 1000
    error_threshold = 0.02
    inliers_threshold = 0
    selected_indices = []
    F = 0
    for i in range(0, n_iterations):
        indices = []
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        eight_features = features[random_indices, :]
        eight_f = compute_fundamental_matrix(eight_features)
        for j in range(n_rows):
            feature = features[j]
            error = error_of_F(feature, eight_f)
            if (error < error_threshold):
                indices.append(j)
        if len(indices) > inliers_threshold:
            inliers_threshold = len(indices)
            selected_indices = indices
            F = eight_f
    filtered_features = features[selected_indices, :]
    return F, filtered_features

def compute_essential_matrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U, S, V = np.linalg.svd(E)
    S = [1, 1, 0]
    resulted_E = np.dot(U, np.dot(np.diag(S), V))
    return resulted_E

def compute_and_match_image_sizes(image_files):
    images = image_files.copy()
    sizes = []
    for image in images:
        x, y, channel = image.shape
        sizes.append([x, y, channel])
    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis=0)
    resized_images = []
    for i, image in enumerate(images):
        resized_image = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        resized_image[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        resized_images.append(resized_image)
    return resized_images

def display_matches_on_images(image_file1, image_file2, matched_pairs, color, filename):
    image1 = image_file1.copy()
    image2 = image_file2.copy()
    image1, image2 = compute_and_match_image_sizes([image1, image2])
    concat = np.concatenate((image1, image2), axis=1)
    if matched_pairs is not None:
        corners_1_x = matched_pairs[:,0].copy().astype(int)
        corners_1_y = matched_pairs[:,1].copy().astype(int)
        corners_2_x = matched_pairs[:,2].copy().astype(int)
        corners_2_y = matched_pairs[:,3].copy().astype(int)
        corners_2_x += image1.shape[1]
        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i], corners_2_y[i]), color, 2)
    cv2.imwrite(filename, concat)
    return concat

def compute_camera_pose(E):
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotational = []
    translation = []
    rotational.append(np.dot(U, np.dot(W, V)))
    rotational.append(np.dot(U, np.dot(W, V)))
    rotational.append(np.dot(U, np.dot(W.T, V)))
    rotational.append(np.dot(U, np.dot(W.T, V)))
    translation.append(U[:, 2])
    translation.append(-U[:, 2])
    translation.append(U[:, 2])
    translation.append(-U[:, 2])
    for i in range(4):
        if(np.linalg.det(rotational[i]) < 0):
            rotational[i] = -rotational[i]
            translation[i] = -translation[i]
    return rotational, translation

def obtain_positive_values(pts, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3, 1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    n_positive = 0
    for i in range(pts.shape[1]):
        X = pts[:,i]
        X = X.reshape(4,1)
        X_c = np.dot(P, X)
        X_c = X_c/X_c[3]
        z = X_c[2]
        if(z > 0):
            n_positive += 1
    return n_positive

def compute_3D_points(K1, K2, inliers, rotational, translational):
    points_3D_4 = []
    rotational_1 = np.identity(3)
    translational_1 = np.zeros((3, 1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(rotational_1, np.hstack((I, -translational_1.reshape(3, 1)))))
    for i in range(len(translational)):
        x1 = inliers[:,0:2].T
        x2 = inliers[:,2:4].T
        P2 = np.dot(K2, np.dot(rotational_1, np.hstack((I, -translational[i].reshape(3, 1)))))
        X = cv2.triangulatePoints(P1, P2, x1, x2)
        points_3D_4.append(X)
    return points_3D_4

def compute_x(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

def compute_epipolar_lines(set1, set2, F, image0, image1, rectified=False):
    lines_1, lines_2 = [], []
    img_1 = image0.copy()
    img_2 = image1.copy()
    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)
        line_2 = np.dot(F, x1)
        lines_2.append(line_2)
        line_1 = np.dot(F.T, x2)
        lines_1.append(line_1)
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = compute_x(line_2, y2_min)
            x2_max = compute_x(line_2, y2_max)
            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = compute_x(line_1, y1_min)
            x1_max = compute_x(line_1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line_2[2]/line_2[1]
            y2_max = -line_2[2]/line_2[1]
            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line_1[2]/line_1[1]
            y1_max = -line_1[2]/line_1[1]
        cv2.circle(img_2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_2 = cv2.line(img_2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (int(i*2.55), 255, 0), 2)
        cv2.circle(img_1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_1 = cv2.line(img_1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (int(i*2.55), 255, 0), 2)
    image_1, image_2 = compute_and_match_image_sizes([img_1, img_2])
    concat = np.concatenate((image_1, image_2), axis = 1)
    concat = cv2.resize(concat, (1920, 660))
    return lines_1, lines_2, concat