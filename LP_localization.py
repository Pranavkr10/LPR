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
    #here we are extracting only the height and width with the help of slicing here
    height, width = img.shape[:2]
    #for high-resolution img(h>1k pxl) a smaller scale factor(1.05)
    #this will reduce the size of the image in each iteration
    #for low resolution img we will use a larger scale factor 1.1 for faster processing
    #min neighbors: it will aqdjust min no. of neighboring rectangles needed to retain a detection
    #for higher resolution -5(reduces false +ve), lower resolution img-3(requires fewer neighbors)
    #block size:here the max fn ensures that the block size is at least 11 to maintain a reasonable window size for processing
    #aspect ratio:range for potential plates, wider range accounts for angled or skewed plates
    #area threshold:min area of a detected region to be considered a potenital plate
    return {
        'scale_factor': 1.05 if height > 1000 else 1.1,  # Adjust for high-res images
        'min_neighbors': 5 if height > 1000 else 3,
        'block_size': max(11, int(width * 0.02)),  # Adaptive block size for thresholding
        'aspect_ratio_range': (1.8, 5.0),  # Wider aspect ratio for angled plates
        'area_threshold': max(300, (width * height) * 0.0005)
    }

# Pre-processing function
def preProcessingImg(imgPath):
    #reading the img in grayscale 
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    imgGray = clahe.apply(imgGray)
    #adaptive threshold is used here to control uneven lightning
    #cvADAPTIVE_THRESH_GAUSSIAN_C->find teh threshold value by taking the weighted sum of the neighbourhood val
    #cvTHRESH_BINARY_INV is used to invert the binary img
    #11->block size, 2->constant subtracted from the weighted sum
    #255->max value assigned to a pixel
    imgAdaptiveThresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    #kernel->use to define the shape of theimg
    #here it is a 3X3 matrix will all the values as 1
    #np.uint8->defines the data type of the kernel which is unsigned int here of 8 bit
    kernel = np.ones((3, 3), np.uint8)
    #cv.MORPH_CLOSE->closes the small holes(black spots)in the img
    #morphologyEx->apply morphological operation on the img
    imgMorph = cv.morphologyEx(imgAdaptiveThresh, cv.MORPH_CLOSE, kernel)
    #Canny is used for noise reduction and edge detection
    #30->lower threshold/min value, 100->upper threshold/max value
    edges = cv.Canny(imgAdaptiveThresh, 30, 100)
    return imgGray, imgMorph, edges

# Color classification for Indian plates
def isValidPlateColour(roi):
#the region of interest(roi) is converted from BGR to HSV colour space
#we are using HSV coz it is often more helpful for colour segmentation coz it separates
#chromatic content(collour info)from intensity(brightness)
#here hsvis a numpy arr and colour_profiles a dictinoary
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    
    # Define color ranges (Hue, Saturation, Value)
    colour_profiles = {
        'private_white': ([0, 0, 150], [180, 30, 255]),     # White background
        'commercial_yellow': ([20, 100, 100], [30, 255, 255]), # Yellow background
        'government_blue': ([100, 50, 50], [130, 255, 255])    # Blue background
    }
#for each colour profile a mask is created to identify pixels within the specified HSV range
#the loop iterates over each colour profile defined above dictionary to check if the roi matches
# _ is used to ignore the key(colour profile name)and obly retrive the value from the tuple containing lower and 
# upper bound of HSV  
    for _, (lower, upper) in colour_profiles.items():
#checks if the mask covers more than 30% of roi if yes return true else false
#this will make sure roi matches one of the colour profiles
#for each colour profile cv.inRange fn is sued to create a binary mask. This mask highlights the pixels
#within the range 255 to 0
        mask = cv.inRange(hsv, np.array(lower), np.array(upper))
#np.sum(mask) calculates the total number of white pixels in the mask representing the pixels within the range
        if np.sum(mask) > 0.3 * mask.size:  # > 30% color coverage
            return True
    return False

# Function to check if a region is a potential license plate
def isPossibleLicensePlate(region, img, params):
    #these are the coordinates of the bounding  box and region.bbox is used here to get those coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    #calculating the height and width of the region
    region_height = maxRow - minRow
    region_width = maxCol - minCol
    #calculating the ratio of width and height of the region
    aspect_ratio = region_width / region_height
    #extracting only the height and width of the img using slicinig
    img_height, img_width = img.shape[:2]
#1.8<=params['aspect ratio range'][0]<=5 is the lower bound where [0]->first element  
#params['aspect ratio range'][1]->upper bound of the range
#region.area>params['area threshold']->checks if the area of region is > that specified threshold
    return (params['aspect_ratio_range'][0] <= aspect_ratio <= params['aspect_ratio_range'][1] and
            region.area > params['area_threshold'])

# Haar cascade detection with dynamic parameters
def haarCascadeDetect(gray_img, params):
    cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    return cascade.detectMultiScale(gray_img,scaleFactor=params['scale_factor'],
     minNeighbors=params['min_neighbors'],minSize=(30,30))  # Smaller minSize for distant plates

# Region properties detection
def regionPropsDetect(edges, img, params):
    #measure.label->label the connected regions in the img
    labelImage = measure.label(edges)
    #storing the detected regions in a list
    detectedRegions = []
    #this loop iterates through the regions in the img
    for region in regionprops(labelImage):
        #if the region is wayy too much small like here lets say 50 pxl then continue to next region
        if region.area < 50:
            continue
        #validating if tthe region is a possible license plate or not
        if isPossibleLicensePlate(region, img, params):
            detectedRegions.append(region.bbox)
    return detectedRegions

# License plate localization function
def plateLocalization(imgPath):
    #reads the img
    img = cv.imread(imgPath)
    params = getAdaptiveParameters(img)
    #pre processing
    gray, thresh, edges = preProcessingImg(imgPath)
    #storing detected plates in a list in the form of indices of the boundig boxes of the plate
    detected_plates = []

    # Haar Cascade Detection with dynamic parameters
    plates_haar = haarCascadeDetect(gray, params)
    #(x,y,w,h)->coordinates of the bounding box
    #this loop is used to iterate throught  the plates
    for (x,y,w,h) in plates_haar:
        roi = img[y:y+h, x:x+w]
        if isValidPlateColour(roi):
            detected_plates.append((x,y,w,h))
            cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

    # Regionprops Detection
    regions = regionPropsDetect(edges, img, params)
    #this loop is used here to iterate throught the plates
    for bbox in regions:
        #these are  the coordinate sof teh bounding box
        minCol, minRow, maxCol, maxRow = bbox
        roi = img[minRow:maxRow, minCol:maxCol]
        if isValidPlateColour(roi):
            detected_plates.append(bbox)
            cv.rectangle(img, (minCol,minRow), (maxCol,maxRow), (0,0,255), 2)

    # Display results
    ShowImage(resize(img, 60))
imgPath=r"sampleImg.jpg"
plateLocalization(imgPath)