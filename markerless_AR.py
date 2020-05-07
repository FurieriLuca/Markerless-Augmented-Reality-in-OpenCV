import sys
import numpy as np
import cv2 as cv
import imageio
import urllib.request
import math



print(sys.version_info)
print("OpenCV version: ", end='')
print(cv.__version__)

while True:
    selection = input("Please select '1' for PLAGIARISM example and '2' for UFO example:\n " )
    if selection in {'1','2'}:
        break
    else:
        print("Please select either '1' or '2'\n")

selection = int(selection)

while True:
    mode = input("Would you like to select F2R, F2F or the proposed hybrid approach? Type 'F2F', 'F2R' or 'H':\n " )
    if mode.lower() in {'f2r','f2f','h'}:
        break
    else:
        print("Please select either 'F2F', 'F2R' or 'H'\n")


######          Loading pictures and video   #######

if selection == 2:
#Import Gif
    url = "https://media.giphy.com/media/3bb5jcIADH9ewHnpl9/giphy.gif"
    fname = "tmp.gif"

    ## Read the gif from the web, save to the disk
    imdata = urllib.request.urlopen(url).read()
    imbytes = bytearray(imdata)
    open(fname,"wb+").write(imdata)

    ## Read the gif from disk to `RGB`s using `imageio.miread`
    gif = imageio.mimread(fname)
    nums = len(gif)


    # convert form RGB to BGR
    augmentation_gif = [cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in gif]
    #Create padded augmentation gif
    height_gif, width_gif, cc_gif= augmentation_gif[0].shape
    original = cv.VideoCapture('img/video_outside.avi')
else:
    augmentation_layer = cv.imread('img/AugmentedLayerCyanLab.png')
    original = cv.VideoCapture('img/original_short.avi')



fourcc = cv.VideoWriter_fourcc(*"DIVX")

ret, reference = original.read()

# Compute height, width of the images and video
height, width, channels = reference.shape

vidout = cv.VideoWriter("output.avi", fourcc, 30, (width*4, height*2))




# Construction of an object mask based on blue channel for the reference frame
b_ref,g,r = cv.split(reference)
threshold, b_ref = cv.threshold(b_ref, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Otsu thresholding



# Extract feature points in the Reference frame
sift = cv.xfeatures2d.SIFT_create()
kp_ref, descr_ref = sift.detectAndCompute(cv.cvtColor(reference, cv.COLOR_BGR2GRAY), None)
kp_prev = kp_ref
descr_prev = descr_ref

# Create BFMatcher object
matcher = cv.BFMatcher()

### Video Cycle: find matching keypoints in thisframe and reference
nloop = 0
while (original.isOpened()):
    ret, thisframe = original.read()
    if ret:
        #color corrections
        if selection == 2:
            #need to pad the gif with black
            augmentation_layer = cv.copyMakeBorder(cv.resize(augmentation_gif[nloop%nums],(math.floor(width_gif / 2.5), math.floor(height_gif / 2.5))),8,0,math.floor((width-width_gif)/2)+20,math.floor((width-width_gif)/2),cv.BORDER_CONSTANT,(0,0,0))
            augmentation_layer = cv.addWeighted(augmentation_layer, -0.6, augmentation_layer, 1, 0)
        elif nloop == 0:
            augmentation_layer = cv.imread('img/AugmentedLayerCyanLab.png')
            augmentation_layer = cv.addWeighted(thisframe, -1, augmentation_layer, 1, 0)


        #keypoints of the frame
        kp_this, descr_this = sift.detectAndCompute(cv.cvtColor(thisframe, cv.COLOR_BGR2GRAY), None)

        #Extract matches points F2R and F2F
        matchesF2R = matcher.knnMatch(descr_ref, descr_this, k=2) #F2R
        matchesF2F = matcher.knnMatch(descr_prev, descr_this, k=2) #F2F

        # Apply ratio test F2R
        goodF2R = []
        for m, n in matchesF2R:
            if m.distance < 0.75 * n.distance:
                goodF2R.append(m)
        # Apply ratio test F2F
        goodF2F = []
        for m, n in matchesF2F:
            if m.distance < 0.75 * n.distance:
                goodF2F.append(m)

        ###Retrieve and Reshape the good keypoints

        ref_ptsF2R = np.float32([kp_ref[m.queryIdx].pt for m in goodF2R]).reshape(-1, 1, 2)
        ref_ptsF2F = np.float32([kp_prev[m.queryIdx].pt for m in goodF2F]).reshape(-1, 1, 2)
        this_ptsF2R = np.float32([kp_this[m.trainIdx].pt for m in goodF2R]).reshape(-1, 1, 2)
        this_ptsF2F = np.float32([kp_this[m.trainIdx].pt for m in goodF2F]).reshape(-1, 1, 2)

        # Compute Homography through Ransac estimation
        MF2R, maskF2R = cv.findHomography(ref_ptsF2R, this_ptsF2R, cv.RANSAC, 5.0)
        MF2F, maskF2F = cv.findHomography(ref_ptsF2F, this_ptsF2F, cv.RANSAC, 5.0) #this is with respect to previous frame
        if nloop > 0:
            MF2F = np.matmul(M, MF2F) #successive multiplication to go back fo reference frame



        #### Select best homography for this frame between F2R and F2F ####

        # Extract mask from this frame (the book is blue, the sky is blue, so blue channel is relevant for both examples)
        b_this, g, r = cv.split(thisframe)
        threshold, b_this = cv.threshold(b_this, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Otsu thresholding

        # Try MF2R and MF2F to find which one gives most matching pixels between warped ref and thisframe
        warped_b_refF2R = cv.warpPerspective(b_ref, MF2R, (width, height))
        warped_b_refF2F = cv.warpPerspective(b_ref, MF2F, (width, height))

        wrong_F2R = np.sum(b_this != warped_b_refF2R)
        wrong_F2F = np.sum(b_this != warped_b_refF2F)

        #Apply best homography
        if mode.lower() == 'h':
            if wrong_F2R <= wrong_F2F:
                warped_augmentation_layer = cv.warpPerspective(augmentation_layer, MF2R, (width, height))
                M = MF2R
                #print('F2R')
            else:
                warped_augmentation_layer= cv.warpPerspective(augmentation_layer, MF2F, (width, height))
                #print('F2F')
                M = MF2F

        elif mode.lower() == 'f2r':
                warped_augmentation_layer = cv.warpPerspective(augmentation_layer, MF2R, (width, height))
                M = MF2R #placeholder
        else:
                warped_augmentation_layer = cv.warpPerspective(augmentation_layer, MF2F, (width, height))
                M = MF2F

        grey_warped_augmentation_layer = cv.cvtColor(warped_augmentation_layer, cv.COLOR_BGR2GRAY)


        #Create the augmented frame by adding augmentation layer and this frame together
        augmented_frame = thisframe.copy()
        tmpImage = cv.resize(warped_augmentation_layer, (width * 2, height * 2)) #better for visualization

        cv.imshow('asd',warped_augmentation_layer)

        augmented_frame = cv.addWeighted(warped_augmentation_layer, 1, augmented_frame, 1, 0)

        augmented_frame = cv.resize(augmented_frame, (width * 2, height * 2))
        cv.imshow('Augmented video', augmented_frame)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        nloop = nloop + 1
        kp_prev = kp_this
        descr_prev = descr_this


        vidout.write(np.concatenate((augmented_frame, cv.resize(thisframe, (width * 2, height * 2))), axis=1))

    else:
        break


vidout.release()

original.release()
cv.destroyAllWindows()
