import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import sys

def compute_orb_keypoints(filename):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img = cv2.imread(filename)
    
    # create orb object
    orb = cv2.ORB_create()
    
    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)
    
    # detect keypoints
    kp = orb.detect(img,None)

    # for detected keypoints compute descriptors. 
    kp, des = orb.compute(img, kp)
    
    return img,kp, des

    
def brute_force_matcher(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches


def loop_verification(img_fname_1, img_fname_2, new_size=None, is_vis=False):
    
    # """
    # Takes in filenames of two input images 
    # Return Fundamental matrix computes 
    # using 8 point algorithm
    # """

    # compute ORB keypoints and descriptor for each image
    img1, kp1, des1 = compute_orb_keypoints(img_fname_1)
    img2, kp2, des2 = compute_orb_keypoints(img_fname_2)
    
    # compute keypoint matches using descriptor
    matches = brute_force_matcher(des1, des2)
    
    # extract points
    pts1 = []
    pts2 = []
    good_matches = []
    for i,(m) in enumerate(matches):
        if m.distance < 20:
            #print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(matches[i])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=0.99)

    # We select only inlier points
    if mask is None:
    #   print(0, end="\t")
      return 0
    else: 
      inlier_matches = [b for a, b in zip(mask, good_matches) if a]

    return len(inlier_matches)

    # 可视化匹配结果和验证结果
    if is_vis:
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,inlier_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def evaluate(gt_txt):
    match_points = []
    labels = []
    fp = open(gt_txt, "r")
    for line in tqdm(fp):
        line_str = line.split(", ")
        query, reference, gt = line_str[0], line_str[1], int(line_str[2])
        match_points.append(loop_verification(query, reference, new_size=None, is_vis=False))
        labels.append(gt)
    return np.array(match_points), np.array(labels)


if __name__ == '__main__':
    # visualization
    # loop_verification("Kudamm_mini_query/image0197.jpg", "Kudamm_mini_ref/image0174.jpg", new_size=None, is_vis=True)

    # evaluate
    datasets = ["Kudamm_easy_final.txt", "Kudamm_diff_final.txt", "robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt", "robotcar_qAutumn_dbSunCloud_easy_final.txt", "robotcar_qAutumn_dbSunCloud_diff_final.txt"]

    for dataset in datasets:
        print("-------- Processing {} ----------".format(dataset.strip(".txt")))
        match_points, labels = evaluate(dataset)
        scaled_scores = match_points / max(match_points)
        precision, recall, thresholds = precision_recall_curve(labels, scaled_scores)
        average_precision = average_precision_score(labels, scaled_scores)
        plt.plot(recall, precision, label="{} (AP={:.3f})".format(dataset.strip(".txt"), average_precision))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall Curves for ORB baseline")
        plt.savefig("pr_curve_{}.png".format(dataset.strip(".txt")))
        plt.close()
    
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend()
    # plt.title("Precision-Recall Curves for SIFT baseline")
    # plt.savefig("pr_curve_SIFT.png")








    
