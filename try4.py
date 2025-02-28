import cv2 as cv
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Function to display the image
def ShowImage(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

# Function to resize the image for better display
def resize(img, scaleInPercent):
    width = int(img.shape[1] * scaleInPercent / 100)
    height = int(img.shape[0] * scaleInPercent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Dynamic parameter adjustment
def getAdaptiveParameters(img):
    height, width = img.shape[:2]
    return {
        'scale_factor': 1.05 if height > 1000 else 1.1,
        'min_neighbors': 5 if height > 1000 else 3,
        'block_size': max(11, int(width * 0.02)),
        'aspect_ratio_range': (1.8, 5.0),
        'area_threshold': max(300, (width * height) * 0.0005)
    }

# Pre-processing function
def preProcessingImg(imgPath):
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    imgGray = clahe.apply(imgGray)
    imgAdaptiveThresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv.morphologyEx(imgAdaptiveThresh, cv.MORPH_CLOSE, kernel)
    edges = cv.Canny(imgAdaptiveThresh, 30, 100)
    return imgGray, imgMorph, edges

# Color classification for Indian plates
def isValidPlateColour(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    colour_profiles = {
        'private_white': ([0, 0, 150], [180, 30, 255]),
        'commercial_yellow': ([20, 100, 100], [30, 255, 255]),
        'government_blue': ([100, 50, 50], [130, 255, 255])
    }
    for _, (lower, upper) in colour_profiles.items():
        mask = cv.inRange(hsv, np.array(lower), np.array(upper))
        if np.sum(mask) > 0.3 * mask.size:
            return True
    return False

# Function to check if a region is a potential license plate
def isPossibleLicensePlate(region, img, params):
    minRow, minCol, maxRow, maxCol = region.bbox
    region_height = maxRow - minRow
    region_width = maxCol - minCol
    aspect_ratio = region_width / region_height
    img_height, img_width = img.shape[:2]
    return (params['aspect_ratio_range'][0] <= aspect_ratio <= params['aspect_ratio_range'][1] and
            region.area > params['area_threshold'])

# Haar cascade detection with dynamic parameters
def haarCascadeDetect(gray_img, params):
    cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    return cascade.detectMultiScale(gray_img, scaleFactor=params['scale_factor'],
                                    minNeighbors=params['min_neighbors'], minSize=(30, 30))

# Region properties detection
def regionPropsDetect(edges, img, params):
    labelImage = measure.label(edges)
    detectedRegions = []
    for region in regionprops(labelImage):
        if region.area < 50:
            continue
        if isPossibleLicensePlate(region, img, params):
            detectedRegions.append(region.bbox)
    return detectedRegions

# Character segmentation function (integrated)
def segmentCharacters(plateImg):
    plateGray = cv.cvtColor(plateImg, cv.COLOR_BGR2GRAY)
    plateGray = cv.bilateralFilter(plateGray, 11, 17, 17)
    plateThresh = cv.adaptiveThreshold(plateGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 21, 6)
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    plateThresh = cv.morphologyEx(plateThresh, cv.MORPH_CLOSE, kernel_close)
    plateThresh = cv.morphologyEx(plateThresh, cv.MORPH_OPEN, kernel_open)
    contours, _ = cv.findContours(plateThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    plate_height, plate_width = plateGray.shape
    min_width = max(5, int(plate_width * 0.03))
    max_width = int(plate_width * 0.45)
    min_height = max(10, int(plate_height * 0.2))
    max_height = int(plate_height * 0.9)
    aspect_ratio_min = 0.15
    aspect_ratio_max = 1.5
    char_bboxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)
        if (w < min_width or w > max_width or
                h < min_height or h > max_height or
                aspect_ratio < aspect_ratio_min or
                aspect_ratio > aspect_ratio_max):
            continue
        char_bboxes.append((x, y, w, h))
    char_bboxes = merge_overlapping_boxes(char_bboxes)
    refined_boxes = []
    for box in char_bboxes:
        x, y, w, h = box
        avg_width = np.mean([b[2] for b in char_bboxes]) if char_bboxes else w
        if w > avg_width * 1.65:
            split_boxes = split_touching_characters(plateThresh[y:y + h, x:x + w], x, y)
            refined_boxes.extend(split_boxes)
        else:
            refined_boxes.append(box)
    refined_boxes = sorted(refined_boxes, key=lambda b: b[0])
    refined_boxes = filter_vertical_outliers(refined_boxes)
    return refined_boxes

def merge_overlapping_boxes(bboxes, overlap_threshold=0.7):
    merged = []
    for box in sorted(bboxes, key=lambda b: b[0]):
        x, y, w, h = box
        found = False
        for i, mbox in enumerate(merged):
            mx, my, mw, mh = mbox
            dx = min(x + w, mx + mw) - max(x, mx)
            dy = min(y + h, my + mh) - max(y, my)
            if dx > 0 and dy > 0:
                overlap_area = dx * dy
                min_area = min(w * h, mw * mh)
                if overlap_area / min_area > overlap_threshold:
                    nx = min(x, mx)
                    ny = min(y, my)
                    nw = max(x + w, mx + mw) - nx
                    nh = max(y + h, my + mh) - ny
                    merged[i] = (nx, ny, nw, nh)
                    found = True
                    break
        if not found:
            merged.append(box)
    return merged

def split_touching_characters(char_region, orig_x, orig_y):
    vertical_projection = np.sum(char_region, axis=0)
    threshold = 0.2 * np.max(vertical_projection)
    gaps = np.where(vertical_projection < threshold)[0]
    split_boxes = []
    if len(gaps) > 0:
        prev = 0
        for gap in gaps:
            if gap - prev > 2:
                split_boxes.append((orig_x + prev, orig_y, gap - prev, char_region.shape[0]))
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

def filter_vertical_outliers(bboxes, threshold=0.7):
    if not bboxes:
        return []
    median_y = np.median([b[1] for b in bboxes])
    median_h = np.median([b[3] for b in bboxes])
    filtered = []
    for box in bboxes:
        y, h = box[1], box[3]
        if (abs(y - median_y) < median_h * 0.5 and
                abs(h - median_h) < median_h * threshold):
            filtered.append(box)
    return filtered

# License plate localization function with character segmentation
def plateLocalization(imgPath):
    img = cv.imread(imgPath)
    params = getAdaptiveParameters(img)
    gray, thresh, edges = preProcessingImg(imgPath)
    detected_plates = []

    # Haar Cascade Detection
    plates_haar = haarCascadeDetect(gray, params)
    for (x, y, w, h) in plates_haar:
        roi = img[y:y + h, x:x + w]
        if isValidPlateColour(roi):
            detected_plates.append((x, y, w, h))
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # Perform character segmentation on the detected plate
            char_bboxes = segmentCharacters(roi)
            for (cx, cy, cw, ch) in char_bboxes:
                cv.rectangle(roi, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

    # Regionprops Detection
    regions = regionPropsDetect(edges, img, params)
    for bbox in regions:
        minCol, minRow, maxCol, maxRow = bbox
        roi = img[minRow:maxRow, minCol:maxCol]
        if isValidPlateColour(roi):
            detected_plates.append(bbox)
            cv.rectangle(img, (minCol, minRow), (maxCol, maxRow), (0, 0, 255), 2)
            # Perform character segmentation on the detected plate
            char_bboxes = segmentCharacters(roi)
            for (cx, cy, cw, ch) in char_bboxes:
                cv.rectangle(roi, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

    # Display results
    ShowImage(resize(img, 60))
#imgPath = r"D:\sample img\CarPlate.jpg"
#imgPath =r"D:\sample img\CarPlate2(JPG).jpg" #X
#imgPath = r"D:\sample img\CarPlate3.jpg" 
#imgPath = r"D:\sample img\CarPlate4.jpg" 
#imgPath = r"D:\sample img\CarPlate5.jpg" 
#imgPath = r"D:\sample img\CarPlate6.jpg"
#imgPath = r"D:\sample img\CarPlate7.jpg"
#imgPath = r"D:\sample img\CarPlate8.jpg" 
#imgPath = r"D:\sample img\CarPlate9.jpg" 
#imgPath = r"D:\sample img\CarPlate10.jpg" 
#imgPath = r"D:\sample img\CarPlate11.jpg" 
#imgPath = r"D:\sample img\CarPlate12.jpg" #X
#imgPath = r"D:\sample img\CarPlate13.jpg" #X
imgPath = r"D:\sample img\CarPlate14.jpg" 
#imgPath = r"D:\sample img\CarPlate15.jpg" 
plateLocalization(imgPath)
