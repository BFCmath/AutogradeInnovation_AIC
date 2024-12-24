import cv2
import json
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
def show_sheet(sheet):
    plt.figure(figsize=(20, 10))
    plt.imshow(sheet)
    plt.show()
    
# Attempt 1

# Updated find_contour function
def detect_black_squares(image_name):
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=0, threshold2=100)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []

    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximated contour has 4 points
        if len(approx) == 4:
            # Compute the bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Define criteria for squares (adjust thresholds as needed)
            if 0.9 < aspect_ratio < 1.1 and 20 < w < 200:
                squares.append(approx)

    return img, squares

def expand_squares(squares, image_shape, scale_factor=2.0):
    """
    Expand each square by the given scale factor.
    
    Args:
        squares: List of square contours.
        image_shape: Shape of the original image.
        scale_factor: Factor by which to scale the squares.
        
    Returns:
        List of expanded squares as rectangles (x, y, w, h).
    """
    expanded_rects = []
    img_height, img_width = image_shape[:2]
    
    for square in squares:
        # Compute bounding rectangle
        x, y, w, h = cv2.boundingRect(square)
        cX, cY = x + w / 2, y + h / 2
        
        # New dimensions
        new_w = w * scale_factor
        new_h = h * scale_factor
        
        # New top-left corner
        new_x = int(cX - new_w / 2)
        new_y = int(cY - new_h / 2)
        
        # Ensure the expanded rectangle is within image boundaries
        new_x = max(new_x, 0)
        new_y = max(new_y, 0)
        new_w = int(new_w)
        new_h = int(new_h)
        
        if new_x + new_w > img_width:
            new_w = img_width - new_x
        if new_y + new_h > img_height:
            new_h = img_height - new_y
        
        expanded_rects.append((new_x, new_y, new_w, new_h))
    
    return expanded_rects


# Validate contours against pivots
def validate_contours_against_pivots(contours, pivots):
    valid_contours = []
    for contour in contours:
        # Compute the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour lies within one of the pivots
        valid = False
        for pivot in pivots:
            px, py, pw, ph = pivot
            if px <= x <= px + pw and py <= y <= py + ph and \
            px <= x + w <= px + pw and py <= y + h <= py + ph:
                valid = True
                break

        if valid:
            valid_contours.append(contour)

    return len(valid_contours) == 31, valid_contours

# Process multiple files and validate contours
def process_files(image_folder, label_folder, pivots, save_log_path, error_log_path, save_contours_path):
    save_log = []
    error_log = []

    # Ensure the save_contours_path exists
    os.makedirs(save_contours_path, exist_ok=True)

    for image_name in tqdm(os.listdir(image_folder)):
        if not image_name.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(image_folder, image_name)
        # label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + ".txt")

        # Find contours in the image
        img, contours = detect_black_squares(image_path)

        # Validate contours
        fits, valid_contours = validate_contours_against_pivots(contours, pivots)

        if fits:
            save_log.append(image_name)

            # Save the contours to a JSON file in save_contours_path
            contour_data = [cv2.boundingRect(cnt) for cnt in valid_contours]
            contour_file_path = os.path.join(save_contours_path, os.path.splitext(image_name)[0] + "_contours.json")
            with open(contour_file_path, "w") as f:
                json.dump(contour_data, f)

        else:
            error_log.append(image_name)

    # Save logs
    with open(save_log_path, "w") as f:
        json.dump(save_log, f)

    with open(error_log_path, "w") as f:
        json.dump(error_log, f)


## Attempt 2
def validate_contours_against_pivots_2(contours, pivots, iou_threshold=0.1):
    """
    Validate contours against pivots based on IoU.

    Args:
        contours (list): List of contours (as found by OpenCV).
        pivots (list): List of pivot rectangles [(x, y, w, h)].
        iou_threshold (float): Minimum IoU required for a contour to be considered valid.

    Returns:
        bool: True if the number of valid contours matches the number of pivots, False otherwise.
        list: List of valid contours.
    """
    def calculate_iou(rect1, rect2):
        """Calculate IoU between two rectangles."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Convert to (x_min, y_min, x_max, y_max) format
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0  # No intersection

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        rect1_area = w1 * h1
        rect2_area = w2 * h2
        union_area = rect1_area + rect2_area - inter_area

        return inter_area / union_area

    valid_contours = []

    for contour in contours:
        # Compute the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        contour_rect = (x, y, w, h)

        # Check if IoU with any pivot exceeds the threshold
        valid = False
        for pivot in pivots:
            iou = calculate_iou(contour_rect, pivot)
            if iou >= iou_threshold:
                valid = True
                break

        if valid:
            valid_contours.append(contour)

    # Return True if the number of valid contours matches the number of pivots
    return len(valid_contours) == len(pivots), valid_contours



# Updated find_contour function
def detect_black_squares_2(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # Invert to make black squares white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    black_squares = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check for 4 vertices (quadrilateral)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Check aspect ratio and area
            if 0.8 <= aspect_ratio <= 1.2 and 500 < cv2.contourArea(contour) < 3000:
                black_squares.append(approx)

    # Draw detected squares
    output = image.copy()
    return output, black_squares



# Rerun validation for error images
def process_files_2(error_file, image_folder, pivots, save_log_path, error_log_path, save_contours_path):
    with open(error_file, 'r') as f:
        error_images = json.load(f)

    save_log = []
    error_log = []

    for image_name in tqdm(error_images):
        image_path = os.path.join(image_folder, image_name)

        # Detect contours using the updated algorithm
        _, contours = detect_black_squares(image_path)

        # Validate contours with IoU
        fits, valid_contours = validate_contours_against_pivots_2(contours, pivots)

        if fits:
            save_log.append(image_name)

            # Save the contours to a JSON file in save_contours_path
            contour_data = [cv2.boundingRect(cnt) for cnt in valid_contours]
            contour_file_path = os.path.join(save_contours_path, os.path.splitext(image_name)[0] + "_contours.json")
            with open(contour_file_path, "w") as f:
                json.dump(contour_data, f)

        else:
            error_log.append(image_name)


    # Save results
    with open(save_log_path, 'w') as f:
        json.dump(save_log, f)

    with open(error_log_path, 'w') as f:
        json.dump(error_log, f)


# Attempt 3
def calculate_iou(rect1, rect2):
    """Calculate Intersection over Union (IoU) between two rectangles."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Convert to (x_min, y_min, x_max, y_max) format
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0  # No intersection

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    union_area = rect1_area + rect2_area - inter_area

    return inter_area / union_area

def is_nearly_square(points, tolerance=400):  # Tolerance set to 10 for real-world cases
    # Flatten the points structure if needed
    flat_points = [point[0] for point in points]
    
    # Calculate distances between all pairs of points
    from itertools import combinations
    distances = []
    for (x1, y1), (x2, y2) in combinations(flat_points, 2):
        distances.append((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Sort distances
    distances.sort()

    # Check for nearly square:
    # 4 approximately equal sides and 2 approximately equal diagonals
    sides_close = (
        abs(distances[0] - distances[1]) <= tolerance and
        abs(distances[1] - distances[2]) <= tolerance and
        abs(distances[2] - distances[3]) <= tolerance
    )
    diagonals_close = abs(distances[4] - distances[5]) <= tolerance
    diagonal_to_side_ratio_close = abs(distances[4] - 2 * distances[0]) <= tolerance

    return sides_close and diagonals_close and diagonal_to_side_ratio_close

def detect_black_squares_3(image_path, iou_threshold=0.7, min_area_threshold=500):  # Added min_area_threshold
    """
    Detects black squares using a combined approach of find_contour and detect_black_squares.

    Args:
        image_path (str): Path to the image.
        iou_threshold (float): IoU threshold for agreement between the two methods.
        min_area_threshold (int): Minimum area for a square to be considered valid.

    Returns:
        tuple: (img, squares), where img is the original image and squares is a list of detected squares.
    """
    img, squares1 = detect_black_squares(image_path)
    _, squares2 = detect_black_squares_2(image_path)

    rects1 = [cv2.boundingRect(s) for s in squares1]
    rects2 = [cv2.boundingRect(s) for s in squares2]

    confirmed_squares = []

    all_squares = squares1 + squares2
    all_rects = rects1 + rects2

    eliminated = [False] * len(all_squares)

    # --- 1. Check for agreement between the two methods (Pairwise Matching) ---
    for i in range(len(all_rects)):
        for j in range(i + 1, len(all_rects)):
            iou = calculate_iou(all_rects[i], all_rects[j])
            if iou >= iou_threshold:
                confirmed_squares.append(all_squares[i])
                eliminated[i] = True
                eliminated[j] = True
                break

    # --- 2. Double-check unconfirmed squares ---
    for i, square in enumerate(all_squares):
        if not eliminated[i]:
            x, y, w, h = cv2.boundingRect(square)
            aspect_ratio = w / float(h)
            area = w * h

            # 2.1 Check if it's nearly a square
            is_square = 0.8 <= aspect_ratio <= 1.2

            # 2.2 Check if the area is above the threshold
            is_large_enough = area > min_area_threshold
            # Pass if all conditions are met
            
            if is_square and is_large_enough and is_nearly_square(square):
                confirmed_squares.append(square)

    return img, confirmed_squares

def validate_contours_3(contours):
    # eliminate nearly the same contours
    def is_same_contour(cnt1, cnt2):
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        return abs(x1 - x2) < 10 and abs(y1 - y2) < 10 and abs(w1 - w2) < 10 and abs(h1 - h2) < 10
    
    # Filter out nearly the same contours
    unique_contours = []
    for cnt in contours:
        is_unique = True
        for unique_cnt in unique_contours:
            if is_same_contour(cnt, unique_cnt):
                is_unique = False
                break
        if is_unique:
            unique_contours.append(cnt)
    
    return len(unique_contours) == 31    

# Rerun validation for error images
def process_files_3(error_file, image_folder, pivots, save_log_path, error_log_path, save_contours_path):
    with open(error_file, 'r') as f:
        error_images = json.load(f)

    save_log = []
    error_log = []

    for image_name in tqdm(error_images):
        image_path = os.path.join(image_folder, image_name)

        # Detect contours using the updated algorithm
        _, contours = detect_black_squares_3(image_path)
        fits = validate_contours_3(contours)
        if fits:
            save_log.append(image_name)

            # Save the contours to a JSON file in save_contours_path
            contour_data = [cv2.boundingRect(cnt) for cnt in contours]
            contour_file_path = os.path.join(save_contours_path, os.path.splitext(image_name)[0] + "_contours.json")
            with open(contour_file_path, "w") as f:
                json.dump(contour_data, f)

        else:
            error_log.append(image_name)


    # Save results
    with open(save_log_path, 'w') as f:
        json.dump(save_log, f)

    with open(error_log_path, 'w') as f:
        json.dump(error_log, f)
