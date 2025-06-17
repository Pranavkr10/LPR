import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import re
import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int64)

        tp = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

model3_path = r"D:\My LPR project\Model_training\char_recog3.keras"
model1_path = r"D:\My LPR project\Model_training\char_recog1.keras"
model1 = tf.keras.models.load_model(model1_path, custom_objects={'F1Score': F1Score})
model3 = tf.keras.models.load_model(model3_path, custom_objects={'F1Score': F1Score})

#define index_to_char mapping
index_to_char = {i: str(i) if i < 10 else chr(ord('A') + i - 10) for i in range(36)}

def ShowImage(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def resize(img, scaleInPercent):
    width = int(img.shape[1] * scaleInPercent / 100)
    height = int(img.shape[0] * scaleInPercent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

#dynamic parameter adjustment
def getAdaptiveParameters(img):
    #with the help of slicing we are getting the height and width of the img
    height, width = img.shape[:2]
    ###################################################################################################################################################################
    '''
    This dictionary stores the parameters which are used for the detection of the license plate
    scale_factor -> used to compare the resolution of the img , if the height of the img is  greater than 1k(high resolution img) then we use scale factor accordingly
    min_neighbors -> specicy how many neighbors each candidate rectangle should have to retain it if the  height of the img greater than 1k then min_neigbors 5 
    block_size -> specify the size of the pixel neighborhood to calculate the threshold value max(11, int(width*0.02))->returns the max value of 11
    why (width * 0.02)? to avoid small blocks 
    aspect_ratio_range -> range of the aspect ratio of the bounding box(it is used to filter the bounding boxes)
    area_threshold -> threshold value used to filter the bounding boxes
    '''
    ####################################################################################################################################################################
    return {
        'scale_factor': 1.05 if height > 1000 else 1.1,
        'min_neighbors': 5 if height > 1000 else 3,
        'block_size': max(11, int(width * 0.02)),
        'aspect_ratio_range': (1.8, 5.0),
        'area_threshold': max(300, (width * height) * 0.0005)
    }

###_____STEP-1 PREPROCESSING________###
#pre-processing function
def preProcessingImg(imgPath):
    #reads the image in grayscale
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     #applying the clahe(contrast limiting adaptive threshold) algo to the img
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    imgGray = clahe.apply(imgGray)
     #applying adative threshold to the img for smoothing i.e. to remove noise
    #255-> max intensity value, 13-> diameter of the pixel or block size, 4-> constant value

    imgAdaptiveThresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    #kernel->matrix of ones(structural element) of 3X3 size used for morphological opertions
    kernel = np.ones((3, 3), np.uint8)
    #MORPH_CLOSE->used to close small holes in the img
    imgMorph = cv.morphologyEx(imgAdaptiveThresh, cv.MORPH_CLOSE, kernel)
    '''----FOR EDGE DETECTION WE ARE USING CANNY-----'''
    edges = cv.Canny(imgAdaptiveThresh, 30, 100)
    #30->min threshold value , 100->max threhsold value
    return imgGray, imgMorph, edges

#colour classification for Indian plates
def isValidPlateColour(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    #this dictionarystores the colour profile of the plates
    colour_profiles = {
        'private_white': ([0, 0, 150], [180, 30, 255]),
        'commercial_yellow': ([20, 100, 100], [30, 255, 255]),
        'government_blue': ([100, 50, 50], [130, 255, 255])
    }
    #this loop iterartes theough each colour profile and checks if the roi falls in the range
    for _, (lower, upper) in colour_profiles.items():
        mask = cv.inRange(hsv, np.array(lower), np.array(upper))
        #if the sum of the mask is greater than 0.3 i.e. 30% of the mask size  then it returns true
        if np.sum(mask) > 0.3 * mask.size:
            return True
    return False

#check if a region is a potential license plate
def isPossibleLicensePlate(region, img, params):
    #coordinates of the bounding box
    minRow, minCol, maxRow, maxCol = region.bbox
    #region of interest dimesnions
    region_height = maxRow - minRow
    region_width = maxCol - minCol
    aspect_ratio = region_width / region_height
    img_height, img_width = img.shape[:2]
    return (params['aspect_ratio_range'][0] <= aspect_ratio <= params['aspect_ratio_range'][1] and region.area > params['area_threshold'])

###___STEP-2 PLATE LOCALIZATION_____###

#Haar cascade detection with dynamic parameters
def haarCascadeDetect(gray_img, params):
    #cascade -> cascade classifie obj
    #cv.data.haarcascades -> returns the path of the haarcascade xml file
    cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    #detectMultiscale -> detcets the obj in the img
    #scaleFactor -> specifies how much the image size is reduced at each image scale
    #minNeighbors -> specifies how many neighbors each candidate rectangle should have to retain it
    return cascade.detectMultiScale(gray_img, scaleFactor=params['scale_factor'], minNeighbors=params['min_neighbors'], minSize=(30, 30))

###___STEP-3 CHARACTER SEGMENTATION___###

#Character segmentation function 
def segmentCharacters(plateImg):
    plateGray = cv.cvtColor(plateImg, cv.COLOR_BGR2GRAY)
    plateGray = cv.bilateralFilter(plateGray, 11, 17, 17)
    #applying adaptive thersholding to smooth the img
    plateThresh = cv.adaptiveThreshold(plateGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 6)
    #####################################################################################################################################
    '''
    kernel is used as a structural element for morphological operations, here we are using a 3X3 sized matrix
    MORPH_ELLIPSE -> used to create elliptical shaped kernel. why? coz some characters in the plates can be elleptical in shape
    MORPH_RECT -> used to create a regular rectangular shaped kernel. why? coz some characters in the plates can be rectangular in shape
    why use a (3X3) sized matrix? to remove small black spots in the img
    cv.MorphologyEx is used here to apply morphological operation, these operations includes dilation, erosion, opening , closing
    MORPH_CLOSE -> used to close small holes in the img
    MORPH_OPEN -> used to remove small white spots in the img
    '''
    #####################################################################################################################################
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    plateThresh = cv.morphologyEx(plateThresh, cv.MORPH_CLOSE, kernel_close)
    plateThresh = cv.morphologyEx(plateThresh, cv.MORPH_OPEN, kernel_open)
    #trying to find the contours /outline of the obj in the thresholded img
    #RETR_LIST -> retrieves all the contours, but doesn't create any parent-child relationship
    #CHAIN_APPROX_SIMPLE -> compresses horizontal, vertical, diagonal segments and leaves only their end points
    contours, _ = cv.findContours(plateThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    plate_height, plate_width = plateGray.shape
    ####################################################################################################################################
    '''
    min_width -> minimum width of the bounding box
    max(10, int(plate_height * 0.2)) -> return the ax value of 5
    why int(plate_width * 0.03)? to avoid small bounding boxes

    max_width -> maximum width of the bounding box
    why int(plate_width * 0.45)? to avoid large bouding boxes
    '''
    ###################################################################################################################################
    min_width = max(5, int(plate_width * 0.03))
    max_width = int(plate_width * 0.45)
    min_height = max(10, int(plate_height * 0.2))
    max_height = int(plate_height * 0.9)
    aspect_ratio_min = 0.15
    aspect_ratio_max = 1.5
    #filtering the contour based on the bounding box dimensions and aspect ratio
    char_bboxes = []
    for contour in contours:
        ########################################################################################################################
        '''
        this conditional block checks if the bounding box is valid or not if the bounding box is not valid then it continues to
        the next iteration, if the bounding box is valid then it appends the bounding box to char_bboxes list
        '''
        #########################################################################################################################       
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)
        if (w < min_width or w > max_width or
                h < min_height or h > max_height or
                aspect_ratio < aspect_ratio_min or
                aspect_ratio > aspect_ratio_max):
            continue
        char_bboxes.append((x, y, w, h))
    char_bboxes = mergeOverlappingboxes(char_bboxes)
    refined_boxes = []
    for box in char_bboxes:
        x, y, w, h = box
        avg_width = np.mean([b[2] for b in char_bboxes]) if char_bboxes else w
        if w > avg_width * 1.65:
            split_boxes = splitTouchingCharacters(plateThresh[y:y + h, x:x + w], x, y)
            refined_boxes.extend(split_boxes)
        else:
            refined_boxes.append(box)
    refined_boxes = sorted(refined_boxes, key=lambda b: b[0])
    refined_boxes = filterVerticalOutliers(refined_boxes)
    return refined_boxes

def mergeOverlappingboxes(bboxes, overlap_threshold=0.7):
    #overlap_threshold=0.7 -> if the overlap area of the two boxes is greater than the threshold value then it is merged
    #this list will store the coordinates of the merged bounding boxes
    merged = []
    #sorting the bounidng boxes on the basis of x coordinates 
    for box in sorted(bboxes, key=lambda b: b[0]):
        x, y, w, h = box
        #initially found is set to false coz we have not found any overlapping bunding boxes
        found = False
        #this loop iterates through each boundig box in the merged list
        #enumerate(merged) -> returns the index and bounding box
        for i, mbox in enumerate(merged):
            #coordinates of the merged bounding box
            mx, my, mw, mh = mbox
            #dx -> width of the overlapping area, dy-> height of the overlapping area
            dx = min(x + w, mx + mw) - max(x, mx)
            dy = min(y + h, my + mh) - max(y, my)
            if dx > 0 and dy > 0:#checking of the overlapping area exists
                overlap_area = dx * dy
                min_area = min(w * h, mw * mh)
                if overlap_area / min_area > overlap_threshold:
                    nx = min(x, mx)
                    ny = min(y, my)
                    nw = max(x + w, mx + mw) - nx
                    nh = max(y + h, my + mh) - ny
                    #replaces the merged bounding box with the new merged bounding box
                    merged[i] = (nx, ny, nw, nh)
                    found = True
                    break
        if not found:
            merged.append(box)
    return merged

def splitTouchingCharacters(char_region, orig_x, orig_y):
    #char_rgeion -> regions of the character, orig_x,orig_y-> coordinate sof the bounding box
    #verticle_projection is the sum of the pixel values in the verticle direction
    vertical_projection = np.sum(char_region, axis=0)
    #this threhsold value is used to to split characters, if the pixel values is less than the threshold value then the characters are split
    threshold = 0.2 * np.max(vertical_projection)
    gaps = np.where(vertical_projection < threshold)[0]
    split_boxes = []
    if len(gaps) > 0:
         #prev-> this variable is used to store the previous gap value on the basis of its we split the chars
        prev = 0
        for gap in gaps:
             #if the gap-prev > 2 we split the characters . why 2? to avoid small bounding boxes
            if gap - prev > 2:
                split_boxes.append((orig_x + prev, orig_y, gap - prev, char_region.shape[0]))
                #this statement updates the prev value
                prev = gap
        split_boxes.append((orig_x + prev, orig_y, char_region.shape[1] - prev, char_region.shape[0]))
    else:
        split_boxes.append((orig_x, orig_y, char_region.shape[1], char_region.shape[0]))
    valid_boxes = []
    for box in split_boxes:
        x, y, w, h = box
        if w > 5 and h > 10:
            valid_boxes.append(box)
    return valid_boxes

########################################################################################################################
'''
this function is  used to filter the verticle (outliers-> data pts that differ significantly from other observations)
in simple terms bounding boxes which are not in the same line
median_y -> median of the y coordinates of the bounding boxes(here median is used to avoid outliers)
median is a measure of central tendency which is less affected by the outliers
median_h -> median of the height of the bounding boxes
box[1] -> y coordinate of the bounding box, box[3]-> height of the bounding box
'''
##########################################################################################################################
def filterVerticalOutliers(bboxes, threshold=0.7):
    if not bboxes:
        return []
    median_y = np.median([b[1] for b in bboxes])
    median_h = np.median([b[3] for b in bboxes])
    filtered = []
    #this loop iterates through each bounding box in bboxes list
    for box in bboxes:
        y, h = box[1], box[3]
        if (abs(y - median_y) < median_h * 0.5 and
                abs(h - median_h) < median_h * threshold):
            filtered.append(box)
    return filtered

###___STEP-4 CHARACTER RECOGNITION___###

#function to preprocess character images for inference
def preprocessChar(char_img):
    #convert to grayscale
    char_gray = cv.cvtColor(char_img, cv.COLOR_BGR2GRAY)
    #apply binary thresholding (inverted: white characters on black background)
    _, char_thresh = cv.threshold(char_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    #resize to 28x28
    char_resized = cv.resize(char_thresh, (28, 28), interpolation=cv.INTER_AREA)
    #convert to 3 channels TO replicate grayscale to RGB
    char_rgb = cv.cvtColor(char_resized, cv.COLOR_GRAY2RGB)
    #normalize pixel values
    char_normalized = char_rgb.astype('float32') / 255.0
    return char_normalized

#function to recognize characters in a license plate
def recognizeCharacters(roi):
    char_bboxes = segmentCharacters(roi)
    characters = []
    for idx, (cx, cy, cw, ch) in enumerate(char_bboxes):
        char_img = roi[cy:cy+ch, cx:cx+cw]
        if char_img.size == 0:
            continue
        #prepprocess character
        char_input = preprocessChar(char_img)
        char_input = np.expand_dims(char_input, axis = 0) #adding batch dimension why? coz the model expects a batch of images
        ##########################################################################################################################################
        ''' 
        Here we are loading two models for character recognition, primary model->char_recog3, specialized model(for 1 and 7)->char_recog1
        whenever we get all the bounding box of the characters we use the default one which is model3 but whenever it predicts  1 or 7 we cross
        verify it with the specialized model char_recog1
        during the the time of validation of '1' or'7' overrides char_recog3 prediction for '1' or '7' only if char_recog1 confidently predicts a
        different valid digit(1 or 7)
        '''
        ##########################################################################################################################################
        #prediction udsing model3
        predictions3 = model3.predict(char_input, verbose = 0)
        predicted_idx3 = np.argmax(predictions3, axis = 1)[0]
        predicted_char3 = index_to_char.get(predicted_idx3, '?')
        #default-> model3
        final_char = predicted_char3

        #cross verifying if char_recog3 is predicting '1' or '7'
        if predicted_char3 in ['1', '7']:
            predictions1 = model1.predict(char_input, verbose = 0)
            predicted_idx1 = np.argmax(predictions1, axis = 1)[0]
            predicted_char1 = index_to_char.get(predicted_idx1, '?')

            #override if char_recog1 predicts a valid '1' or '7'
            if predicted_char1 in ['1', '7']:
                final_char = predicted_char1
        
        characters.append(final_char)
        cv.putText(roi, final_char, (cx, cy-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    #process detected characters
    plate_str = ''.join(characters)
    #add full plate text to the image
    cv.putText(roi, plate_str, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"License Plate: {plate_str}")
    return plate_str

#license plate localization function with character segmentation
def plateLocalization(imgPath):
    img = cv.imread(imgPath)
    params = getAdaptiveParameters(img)
    gray, thresh, edges = preProcessingImg(imgPath)
    detected_plates = []
    plate_text = ""
    #Haar Cascade Detection
    plates_haar = haarCascadeDetect(gray, params)
    for (x, y, w, h) in plates_haar:
        roi = img[y:y + h, x:x + w]
        if isValidPlateColour(roi):
            detected_plates.append((x, y, w, h))
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            plate_text = recognizeCharacters(roi)
            
    return img, plate_text
    #display results
    #ShowImage(resize(img, 60))