import cv2
import numpy as np
from keras.models import load_model
import imutils
import easyocr
from typing import List
from PIL import Image

classes = np.arange(0, 16)

model = load_model('model-OCR.h5')

input_size = 48

reader = easyocr.Reader(['en'], gpu=False)


def get_perspective(img, location, height=1008, width=1008):
    """Takes an image and location os interested region.
        And return the only the selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def get_InvPerspective(img, masked_num, location, height=1008, width=1008):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result


def get_number_of_contours(board):
    bfilter = cv2.bilateralFilter(board, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    no_of_contours = len(contours)
    print(f"Number of contours : {len(contours)}")
    return no_of_contours


def find_board(image):
    """Takes an image as input and finds a sudoku board inside of the image"""
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(grayImg, 13, 20, 20)

    edged = cv2.Canny(bfilter, 30, 180)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)

    newimg = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None

    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(image, location)
    return result, location


# split the board into 81 individual images
def split_boxes(sudoku_board, size=9):
    """Takes a sudoku board and split it into 81 cells.
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(sudoku_board, size)
    boxes = []
    images = []
    for r in rows:
        cols = np.hsplit(r, size)
        for box in cols:
            images.append(box)
            box = cv2.resize(box, (input_size, input_size)) / 255.0

            boxes.append(box)
    # cv2.destroyAllWindows()
    return boxes, images


def predictRois(rois: List[np.ndarray]):
    results = []
    for roi in rois:
        result = reader.readtext(roi, detail=0, allowlist='0123456789')
        results.append(result)
    return results


def displayNumbers(img, numbers, color=(0, 255, 0), board_size=9):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1] / board_size)
    H = int(img.shape[0] / board_size)
    for i in range(board_size):
        for j in range(board_size):
            if numbers[(j * board_size) + i] != 0:
                cv2.putText(img, str(numbers[(j * board_size) + i]),
                            (i * W + int(W / 2) - int((W / 4)), int((j + 0.7) * H)), cv2.FONT_HERSHEY_COMPLEX, 2, color,
                            2, cv2.LINE_AA)
    return img


# Read image
img = cv2.imread('sudoku3.jpg')

# extract board from input image
board, location = find_board(img)
num_of_contours = get_number_of_contours(board)
if (num_of_contours > 800):
    board_size = 16
else:
    board_size = 9
# print(f"location : {location}")

gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Resized board", gray)
# print(gray.shape)
rois, images = split_boxes(gray, board_size)

# print(f"Type of images: {type(images)}")
results = predictRois(images)
print(results)
# board_results = np.array(results).astype('uint8').reshape(board_size, board_size)
# print("Results length :")
# print(len(results))

for id, result in enumerate(results):
    if (len(result) == 0):
        results[id] = 0
    else:
        results[id] = int(result[0])

results_mat = np.array(results)
results_mat = np.array(results_mat).astype('uint8').reshape(board_size, board_size)

rois = np.array(rois).reshape(-1, input_size, input_size, 1)

# get prediction
# easyocr_results = []
#
# for roi in images:
#     easy_ocr_prediction = reader.readtext(roi, detail=0, allowlist='0123456789')
#     easyocr_results.append(easy_ocr_prediction)

# print(easyocr_results)
predicted_numbers = []

# get classes from prediction
# for i in easyocr_results:
#     index = (np.argmax(i))  # returns the index of the maximum number of the array
#     predicted_number = classes[index]
#     predicted_numbers.append(predicted_number)

# print(predicted_numbers)

# reshape the list
board_num = np.array(predicted_numbers).astype('uint8').reshape(board_size, board_size)

np.savetxt('output.txt', board_num, fmt='%d', delimiter=' ')

print(img)
# cv2.imshow("Input image", img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
