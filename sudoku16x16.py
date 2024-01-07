import cv2
import imutils
import numpy as np
import easyocr

# Load EasyOCR model
reader = easyocr.Reader(['en'])

input_size = 48

def get_perspective(img, location, height=1600, width=1600):
    """Takes an image and location of interested region.
        And return only the selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)

    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None

    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location, height=1600, width=1600)
    return result, location

# Split the board into individual cells
def split_boxes(board):
    rows = np.vsplit(board, 16)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 16)
        for box in cols:
            boxes.append(box)
    return boxes


# Perform OCR on each cell
def perform_ocr(roi):
    roi = cv2.resize(roi, (input_size, input_size))
    roi_text = reader.readtext(roi)

    return roi_text[0][1] if roi_text else ''


# Read image
img = cv2.imread('sudoku2.jpg')

# Extract board from input image
board, location = find_board(img)

# Convert the board to grayscale
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

# Split the board into individual cells
rois = split_boxes(gray)

# Perform OCR on each cell and extract the text
extracted_text = [[perform_ocr(cell) for cell in row] for row in rois]

# Convert the extracted text to integers to form the Sudoku board
board_num = np.array([[int(cell) if cell.isdigit() else 0 for cell in row] for row in extracted_text])

# Save the Sudoku board to a text file
with open("sudoku_board.txt", "w") as file:
    for row in board_num:
        row_str = " ".join(map(str, row))
        file.write(row_str + "\n")

# Print the Sudoku board
print(board_num)

# Display the input image
cv2.imshow("Input image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
