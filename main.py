import numpy as np
import random
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from time import time
class OopsieException(Exception):
    pass

#enable ecc
debug = False
numerrs = 2
def make_check(val):
    checks = []

    '''
    consult this paper for the (12,8) hamming code used: http://faculty.washington.edu/lcrum/Archives/TCSS372AF08/Hamming.pdf

    '''

    for i in range(0, val.size, 8):
        check = []
        check.append(val[i] ^ val[i + 1] ^ val[i + 3] ^ val[i + 4] ^ val[i + 6])
        check.append(val[i] ^ val[i + 2] ^ val[i + 3] ^ val[i ^ 5] ^ val[i + 6])
        check.append(val[i + 1] ^ val[i + 2] ^ val[i + 3] ^ val[i + 7])
        check.append(val[i + 4] ^ val[i + 5] ^ val[i + 6] ^ val[i + 7])

        checks.append(check)

    return checks
            
def error_inj(descriptors):
    i = random.randrange(len(descriptors))
    y = descriptors[i]
    k = random.randrange(len(y))
    x = y[k]
    j = random.randrange(len(x))
    x[j] = bool(not x[j])
    return i, k, j

def correct(val, checks):

    '''

    H = 1 1 0 1 1 0 1 0 | 1 0 0 0
        1 0 1 1 0 1 1 0 | 0 1 0 0
        0 1 1 1 0 0 0 1 | 0 0 1 0
        0 0 0 0 1 1 1 1 | 0 0 0 1

       v0------------>v7 c0---->c3
    '''
    count = 0
    for i in range(0, val.size, 8):
        p = val[i:i+8].tolist() + checks[count]
        z = []
        z.append(p[0] ^ p[1] ^ p[3] ^ p[4] ^ p[6] ^ p[8])
        z.append(p[0] ^ p[2] ^ p[3] ^ p[5] ^ p[6] ^ p[9])
        z.append(p[1] ^ p[2] ^ p[3] ^ p[7] ^ p[10])
        z.append(p[4] ^ p[5] ^ p[6] ^ p[7] ^ p[11])

        if z == [0, 0, 0, 0]:
            count += 1
            continue
        elif z == [1, 1, 0, 0]:
            val[i] = not val[i]
        elif z == [1, 0, 1, 0]:
            val[i + 1] = not val[i + 1]
        elif z == [0, 1, 1, 0]:
            val[i + 2] = not val[i + 2]
        elif z == [1, 1, 1, 0]:
            val[i + 3] = not val[i + 3]
        elif z == [1, 0, 0, 1]:
            val[i + 4] = not val[i + 4]
        elif z == [0, 1, 0, 1]:
            val[i + 5] = not val[i + 5]
        elif z == [1, 1, 0, 1]:
            val[i + 6] = not val[i + 6]
        elif z == [0, 0, 1, 1]:
            val[i + 7] = not val[i + 7]
        elif z == [1, 0, 0, 0]:
            checks[count][0] = not checks[count][0]
        elif z == [0, 1, 0, 0]:
            checks[count][1] = not checks[count][1]
        elif z == [0, 0, 1, 0]:
            checks[count][2] = not checks[count][2]
        elif z == [0, 0, 0, 1]:
            checks[count][3] = not checks[count][3]
        else:
            raise OopsieException
        count += 1
    return val, checks

def FAST(img, N=9, threshold=0.15, nms_window=2):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16  # 3x3 Gaussian Window

    img = convolve2d(img, kernel, mode='same')

    cross_idx = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
    circle_idx = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                           [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])

    corner_img = np.zeros(img.shape)
    keypoints = []
    for y in range(3, img.shape[0] - 3):
        for x in range(3, img.shape[1] - 3):
            Ip = img[y, x]
            t = threshold * Ip if threshold < 1 else threshold
            # fast checking cross idx only
            if np.count_nonzero(Ip + t < img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3 or np.count_nonzero(
                    Ip - t > img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3:
                # detailed check -> full circle
                if np.count_nonzero(img[y + circle_idx[0, :], x + circle_idx[1, :]] >= Ip + t) >= N or np.count_nonzero(
                        img[y + circle_idx[0, :], x + circle_idx[1, :]] <= Ip - t) >= N:
                    # Keypoint [corner]
                    keypoints.append([x, y])  # Note: keypoint = [col, row]
                    corner_img[y, x] = np.sum(np.abs(Ip - img[y + circle_idx[0, :], x + circle_idx[1, :]]))

    # NMS - Non Maximal Suppression
    if nms_window != 0:
        fewer_kps = []
        for [x, y] in keypoints:
            window = corner_img[y - nms_window:y + nms_window + 1, x - nms_window:x + nms_window + 1]
            # v_max = window.max()
            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps)


def corner_orientations(img, corners):
    # mask shape must be odd to have one centre point which is the corner
    OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
    OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
    for i in range(-15, 16):
        for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
            OFAST_MASK[15 + j, 15 + i] = 1
    mrows, mcols = OFAST_MASK.shape
    mrows2 = int((mrows - 1) / 2)
    mcols2 = int((mcols - 1) / 2)

    # Padding to avoid errors @ corners near image edges. 
    # Padding value=0 to not affect the orientation calculations
    img = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

    # Calculating orientation by the intensity centroid method
    orientations = []
    for i in range(corners.shape[0]):
        c0, r0 = corners[i, :]
        m01, m10 = 0, 0
        for r in range(mrows):
            m01_temp = 0
            for c in range(mcols):
                if OFAST_MASK[r, c]:
                    I = img[r0 + r, c0 + c]
                    m10 = m10 + I * (c - mcols2)
                    m01_temp = m01_temp + I
            m01 = m01 + m01_temp * (r - mrows2)
        orientations.append(np.arctan2(m01, m10))

    return np.array(orientations)


def BRIEF(img, keypoints, orientations=None, n=256, patch_size=9, sigma=1, mode='uniform', sample_seed=42):
    '''
    BRIEF [Binary Robust Independent Elementary Features] keypoint/corner descriptor
    '''
    random = np.random.RandomState(seed=sample_seed)

    # kernel = np.array([[1,2,1],
    #                    [2,4,2],
    #                    [1,2,1]])/16      # 3x3 Gaussian Window

    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273  # 5x5 Gaussian Window

    img = convolve2d(img, kernel, mode='same')

    if mode == 'normal':
        samples = (patch_size / 5.0) * random.randn(n * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[(samples < (patch_size // 2)) & (samples > - (patch_size - 2) // 2)]
        pos1 = samples[:n * 2].reshape(n, 2)
        pos2 = samples[n * 2:n * 4].reshape(n, 2)
    elif mode == 'uniform':
        samples = random.randint(-(patch_size - 2) // 2 + 1, (patch_size // 2), (n * 2, 2))
        samples = np.array(samples, dtype=np.int32)
        pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    if orientations is None:
        mask = (((patch_size // 2 - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - patch_size // 2 + 1))
                & ((patch_size // 2 - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - patch_size // 2 + 1)))

        keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for p in range(pos1.shape[0]):
            pr0 = pos1[p, 0]
            pc0 = pos1[p, 1]
            pr1 = pos2[p, 0]
            pc1 = pos2[p, 1]
            for k in range(keypoints.shape[0]):
                kr = keypoints[k, 1]
                kc = keypoints[k, 0]
                if img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]:
                    descriptors[k, p] = True
    else:
        # Using orientations

        # masking the keypoints with a safe distance from borders
        # instead of the patch_size//2 distance used in case of no rotations.
        distance = int((patch_size // 2) * 1.5)
        mask = (((distance - 1) < keypoints[:, 0])
                & (keypoints[:, 0] < (cols - distance + 1))
                & ((distance - 1) < keypoints[:, 1])
                & (keypoints[:, 1] < (rows - distance + 1)))

        keypoints = np.array(keypoints[mask], dtype=np.intp, copy=False)
        orientations = np.array(orientations[mask], copy=False)
        descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

        for i in range(descriptors.shape[0]):
            angle = orientations[i]
            sin_theta = np.sin(angle)
            cos_theta = np.cos(angle)

            kr = keypoints[i, 1]
            kc = keypoints[i, 0]
            for p in range(pos1.shape[0]):
                pr0 = pos1[p, 0]
                pc0 = pos1[p, 1]
                pr1 = pos2[p, 0]
                pc1 = pos2[p, 1]

                # Rotation is based on the idea that:
                # x` = x*cos(th) - y*sin(th)
                # y` = x*sin(th) + y*cos(th)
                # c -> x & r -> y
                spr0 = round(sin_theta * pr0 + cos_theta * pc0)
                spc0 = round(cos_theta * pr0 - sin_theta * pc0)
                spr1 = round(sin_theta * pr1 + cos_theta * pc1)
                spc1 = round(cos_theta * pr1 - sin_theta * pc1)

                if img[kr + spr0, kc + spc0] < img[kr + spr1, kc + spc1]:
                    descriptors[i, p] = True

    checks = []
    if debug:
        for i in range(len(descriptors)):
            checks.append(make_check(descriptors[i]))

    return descriptors, checks


def match(descriptors1, descriptors2, c1, c2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    if debug:
        for i in range(len(descriptors1)):
            correct(descriptors1[i], c1[i])
        for j in range(len(descriptors2)):
            correct(descriptors2[j], c2[j])

    distances = cdist(descriptors1, descriptors2, metric='hamming')  # distances.shape: [len(d1), len(d2)]

    indices1 = np.arange(descriptors1.shape[0])  # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances,
                         axis=1)  # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"
    # Each d1 point has a d2 point that is the most close to it.
    if cross_check:
        '''
        Cross check idea:
        what d1 matches with in d2 [indices2], should be equal to 
        what that point in d2 matches with in d1 [matches1]
        '''
        matches1 = np.argmin(distances, axis=0)  # [15, 37, 283, ..., len(d2)] "list of d1 points closest to d2 points"
        # Each d2 point has a d1 point that is closest to it.
        # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
        mask = indices1 == matches1[indices2]  # len(mask) = len(d1)
        # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if distance_ratio is not None:
        '''
        the idea of distance_ratio is to use this ratio to remove ambigous matches.
        ambigous matches: matches where the closest match distance is similar to the second closest match distance
                          basically, the algorithm is confused about 2 points, and is not sure enough with the closest match.
        solution: if the ratio between the distance of the closest match and
                  that of the second closest match is more than the defined "distance_ratio",
                  we remove this match entirly. if not, we leave it as is.
        '''
        modified_dist = distances
        fc = np.min(modified_dist[indices1, :], axis=1)
        modified_dist[indices1, indices2] = np.inf
        fs = np.min(modified_dist[indices1, :], axis=1)
        mask = fc / fs <= 0.5
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches

def match_hamming(error_locations,matches,descriptors1, descriptors2, a):

    differences = []

    ret = []
    for b in range(len(a)):
        error_location = error_locations[b]
        d1 = descriptors1
        d2 = descriptors2
        match_kp0 = -1
        for i in range(len(matches)):

            if(a[b] == 0 and matches[i][0]==error_location):
                match_kp0 = matches[i][1]
                d1 = descriptors1
                d2 = descriptors2
            if(a[b] == 1 and matches[i][1]==error_location):
                match_kp0 = matches[i][0]
                d2 = descriptors1
                d1 = descriptors2

        if match_kp0 < 0:
            # print('Match N/A ', 0)
            ret.append(0)
            continue
        match_0 = [error_location,match_kp0]
        differences = np.bitwise_xor(d1[error_location],d2[match_0[1]])
        hamming_dist = sum(differences)
        # print('Hamming distance between kp0 and its match = ',hamming_dist)
        ret.append(hamming_dist)
    return ret


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.feature import plot_matches
    from skimage.transform import pyramid_gaussian

    # Trying multi-scale
    for i in range(20):
        with open("nodebug_doubleerror_64.txt", "a") as f:
            N_LAYERS = 4
            DOWNSCALE = 2

            img1 = cv2.imread('test_imgs/guy2.PNG')
            original_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

            img2 = cv2.imread('test_imgs/guy.PNG')
            original_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            grays2 = list(pyramid_gaussian(gray2, downscale=2, max_layer=4, multichannel=False))

            scales = [(i * DOWNSCALE if i > 0 else 1) for i in range(N_LAYERS)]
            features_img1 = np.copy(img1)
            features_img2 = np.copy(img2)

            kps1 = []
            kps2 = []
            ds1 = []
            ds2 = []
            ms = []

            start = time()

            scale_kp1 = FAST(grays1[0], N=9, threshold=0.15, nms_window=3)
            kps1.append(scale_kp1 * scales[0])
            scale_kp2 = FAST(grays2[0], N=9, threshold=0.15, nms_window=3)
            kps2.append(scale_kp2 * scales[0])
            for keypoint in scale_kp1:
                features_img1 = cv2.circle(features_img1, tuple(keypoint * scales[0]), 3 * scales[0], (0, 255, 0), 1)
            for keypoint in scale_kp2:
                features_img2 = cv2.circle(features_img2, tuple(keypoint * scales[0]), 3 * scales[0], (0, 255, 0), 1)
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(grays1[0], cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(features_img1)
            #
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(grays2[0], cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(features_img2)

            d1, c1 = BRIEF(grays1[0], scale_kp1, mode='uniform', patch_size=8, n=64)
            ds1.append(d1)
            d2, c2 = BRIEF(grays2[0], scale_kp2, mode='uniform', patch_size=8, n=64)
            ds2.append(d2)

            end1 = time()
            a = []
            indeces = []
            for i in range(numerrs):
                if debug:
                    b = random.choice([0,1,2,3])
                    if b == 0:
                        vector_index1, vector_index2, val_index = error_inj([d1])
                        indeces.append(vector_index2)
                    elif b == 1:
                        vector_index1, vector_index2, val_index = error_inj([d2])
                        indeces.append(vector_index2)
                    elif b == 2:
                        vector_index1, vector_index2, val_index = error_inj(c1)
                        indeces.append(vector_index2)
                    else:
                        vector_index1, vector_index2, val_index = error_inj(c2)
                        indeces.append(vector_index2)

                else:
                    b = random.choice([0, 1])
                    if b == 0:
                        vector_index1, vector_index2, val_index = error_inj([d1])
                        indeces.append(vector_index2)
                    elif b == 1:
                        vector_index1, vector_index2, val_index = error_inj([d2])
                        indeces.append(vector_index2)
                a.append(b)
            end2 = time()
            matches = match(d1, d2, c1, c2, cross_check=True)
            end3 = time()
            ms.append(matches)

            # fig = plt.figure(figsize=(20, 10))
            # ax = fig.add_subplot(1, 1, 1)

            hamming_dist = match_hamming(indeces,matches,d1,d2,a)

            # plot_matches(ax, grays1[0], grays2[0], np.flip(scale_kp1, 1), np.flip(scale_kp2, 1), matches)
            f.write(f'no. of matches: {matches.shape[0]}\n')

            f.write(f'time: {(end3 - end2) + (end1 - start):5.5f} seconds\n')

            for h in hamming_dist:
                f.write(f'Hamming distance between kp0 and its match = {h}\n')
            f.write('\n*********************************\n')

    # plt.show()
    #
    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(features_img1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(features_img2)
    # plt.show()
