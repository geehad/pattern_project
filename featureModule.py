import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import label
import statistics


#******************************************************************************************** Gehad's Function ********************#
# This function take one line as grey scale image and returns CC features of this line
def connected_Component_features(img_gray):

    features = []  # connected_Component features of one line   <7 features>   5 finished , 2 remain

    ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    labeled_img, num_connected_comp = label(thresh)
    objs = ndimage.find_objects(labeled_img)

    # list of tuples(x1,y1,x2,y2) each tuple for one connected component
    coorinates_bounding_rec = []

    for i in range(0, len(objs)):
        #width_bounding_rec = (int(objs[i][1].stop) - 1) - (int(objs[i][1].start))
        height_bounding_rec = (int(objs[i][0].stop) - 1) - int(objs[i][0].start)
        if height_bounding_rec > 5:  # to exclude noise and points over letter
            coorinates_bounding_rec.append((int(objs[i][1].start), int(objs[i][0].stop) - 1, int(objs[i][1].stop) - 1, int(objs[i][0].start)))
            cv2.rectangle(thresh, (coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][0],coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][1]), (coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][2],coorinates_bounding_rec[len(coorinates_bounding_rec) - 1][3]), (255, 255, 255), 1)

    #print(coorinates_bounding_rec)
    #print(len(coorinates_bounding_rec))

    # sort according to x1
    coorinates_bounding_rec.sort(key=lambda tup: tup[0])

    #print(coorinates_bounding_rec)
    #################################################### get feature1 #######################################################

    sum_diffDist_boundRec = 0.0
    for i in range(1, len(coorinates_bounding_rec)):
        if (coorinates_bounding_rec[i][0] - coorinates_bounding_rec[i - 1][2]) > 0:
            sum_diffDist_boundRec = sum_diffDist_boundRec + (coorinates_bounding_rec[i][0] - coorinates_bounding_rec[i - 1][2])

    if len(coorinates_bounding_rec) > 1:  # more than 1 CC (akid feh akter mn 1 bs bt check e7tyaty)
        sum_diffDist_boundRec = sum_diffDist_boundRec / ((len(coorinates_bounding_rec)) - 1)

    #print(sum_diffDist_boundRec)

    features.append(sum_diffDist_boundRec)  # get feature1

    ###################################################################################################################

    ################################## Clustering connected_comp into words  ########################################

    Words = []
    max_distance_words = 10  # max distance to consider CC in one word

    word = []
    word.append(0)
    for i in range(1, len(coorinates_bounding_rec)):
        if ((coorinates_bounding_rec[i][0] - coorinates_bounding_rec[i - 1][2]) < max_distance_words):
            word.append(i)
            # print(word)
        else:
            Words.append(word)
            word = []
            word.append(i)

    Words.append(word)
    #print(Words)
    ################################################################################################################

    #################################################### get feature2 #######################################################
    sum_diffDist_words = 0.0
    for i in range(1, len(Words)):
        index_bound_rec_SecondWord = Words[i][0]
        index_bound_rec_FirstWord = Words[i - 1][len(Words[i - 1]) - 1]
        sum_diffDist_words = sum_diffDist_words + ((coorinates_bounding_rec[index_bound_rec_SecondWord][0]) - (
        coorinates_bounding_rec[index_bound_rec_FirstWord][2]))

    if len(Words) > 1:
        sum_diffDist_words = sum_diffDist_words / ((len(Words)) - 1)

    features.append(sum_diffDist_words)  # get feature2
    #print(sum_diffDist_words)

    ################################################### get features 4,5,6 ##################################################
    ############## get width of each bounding_rec ########################

    width_bound_recS = []
    for i in range(0, len(coorinates_bounding_rec)):
        width_bound_rec = coorinates_bounding_rec[i][2] - coorinates_bounding_rec[i][0]
        width_bound_recS.append(width_bound_rec)
    #####################################################################

    med_width_bound_recS = statistics.median(width_bound_recS)
    features.append(med_width_bound_recS)  # get feature 4
    std_width_bound_recS = statistics.stdev(width_bound_recS)
    features.append(std_width_bound_recS)  # get feature 5
    mean_width_bound_recS = sum(width_bound_recS) / len(width_bound_recS)
    features.append(mean_width_bound_recS)  # get feature 6

    # print(width_bound_recS)
    # print(med_width_bound_recS)
    # print(std_width_bound_recS)
    # print(mean_width_bound_recS)

    #########################################################################################################################
    #cv2.imshow("img", img_gray)
    #cv2.imshow("thresh", thresh)
    #cv2.waitKey(0)

    return features

#******************************************************************************************** Gehad's Function ********************#

# This function take one line as grey scale image and returns CC features of this line
def Enclosed_Regions_features(img_gray):

    features = []   #Enclosed_Regions features of one line   < 1 feature >  1 finished

    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Determine which openCV version were using
    if cv2.__version__.startswith('2.'):
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect the blobs in the image
    keypoints = detector.detect(img_gray)

    # Draw detected keypoints as red circles
    imgKeyPoints = cv2.drawKeypoints(img_gray, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    sum_sizesBlobs_line = 0.0
    for kp in keypoints:
        #print("(%d, %d) size=%.1f resp=%.1f" % (kp.pt[0], kp.pt[1], kp.size, kp.response))
        sum_sizesBlobs_line = sum_sizesBlobs_line + kp.size

    Avg_sum_sizesBlobs_line = sum_sizesBlobs_line / (len(keypoints))
    features.append(Avg_sum_sizesBlobs_line)
    #print(Avg_sum_sizesBlobs_line)
    # Display found keypoints
    # cv2.imshow("threshold", threshold)

    #cv2.imshow("Keypoints", imgKeyPoints)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return features
#**************************************************************************************************************************************#

def main():
    # Read the image you want connected components of
    img_gray = cv2.imread('blob3.png', cv2.IMREAD_GRAYSCALE)
    CC_features=connected_Component_features(img_gray)
    print(CC_features)
    ER__features=Enclosed_Regions_features(img_gray)
    print(ER__features)

if __name__ == "__main__":
    main()
