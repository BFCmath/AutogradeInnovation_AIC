import cv2
import numpy as np
import os
import pandas as pd

# Cropping logic (using the matrix indices you provided)
crop_instructions = [
    ("SBD", [1, 0, 1, 1, 2, 0, 2, 1]),  # mat[1][0], mat[1][1], mat[2][0], mat[2][1]
    ("MDT", [1, 1, 0, 1, 2, 1, 3, 3]),  # mat[1][1], mat[2][1], mat[0][1], mat[3][3]
    ("1.00", [3, 0, 3, 1, 4, 0, 4, 1]), # first 10 questions of part 1
    ("1.10", [3, 1, 3, 2, 4, 1, 4, 2]), # next 10 questions of part 1
    ("1.20", [3, 2, 4, 2, 4, 3]), # Use only 3 contours
    ("1.30", [2, 0, 3, 3, 4, 3, 4, 4]),
    ("2.00", [4, 0, 4, 1, 5, 0, 5, 2]),
    ("2.02", [4, 1, 4, 2, 5, 2, 5, 4]),
    ("2.04", [4, 2, 4, 3, 5, 4, 5, 6]),
    ("2.06", [4, 3, 4, 4, 5, 6, 5, 8]),
    ("3.00", [5, 0, 5, 1, 6, 0, 6, 1]),
    ("3.01", [5, 1, 5, 3, 6, 1, 6, 2]),
    ("3.02", [5, 3, 5, 4, 6, 2, 6, 3]),
    ("3.03", [5, 4, 5, 5, 6, 3, 6, 4]),
    ("3.04", [5, 5, 5, 7, 6, 4, 6, 5]),
    ("3.05", [5, 7, 5, 8, 6, 5, 6, 6])
]

def read_sorted_contours(filepath):
    """
    Reads the sorted contours from a text file.

    Args:
        filepath: The path to the sorted contours text file.

    Returns:
        A list of lists, where each inner list represents a line of contours,
        and each contour is a list [x, y, w, h].
        Returns None if the file does not exist or if there is an error parsing the file.
    """
    lines = []
    try:
        with open(filepath, 'r') as f:
            current_line = []
            for line in f:
                line = line.strip()
                if line == "":  # Empty line indicates the end of a line of contours
                    if current_line:
                        lines.append(current_line)
                        current_line = []
                else:
                    try:
                        x, y, w, h = map(int, line.split())
                        current_line.append([x, y, w, h])
                    except ValueError:
                        print(f"Error: Could not parse line: {line} in file {filepath}")
                        return None

            # Add the last line if it exists
            if current_line:
                lines.append(current_line)

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    return lines

def convert_sorted_contours_to_matrix(sorted_contours):
    """
    Converts the list of sorted contour lines into a 2D matrix representation.

    Args:
        sorted_contours: A list of lists representing sorted contour lines.

    Returns:
        A 2D NumPy array (matrix) where each element is a contour [x, y, w, h].
    """
    # Find the maximum number of columns (contours in a line)
    max_cols = 0
    for line in sorted_contours:
        max_cols = max(max_cols, len(line))

    # Create an empty matrix with enough rows and columns
    matrix = np.full((len(sorted_contours), max_cols, 4), -1)  # -1 indicates an empty cell

    # Fill the matrix with contour data
    for i, line in enumerate(sorted_contours):
        for j, contour in enumerate(line):
            matrix[i, j] = contour

    return matrix

def cut_image_based_on_matrix(image_path, matrix, output_dir):
    """
    Cuts an image into multiple images based on the contour matrix.
    Handles the "1.20" crop using only three contours.

    Args:
        image_path: Path to the input image.
        matrix: The 2D matrix of sorted contours.
        output_dir: Directory to save the cropped images.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    cut_images_count = 0
    for crop_name, indices in crop_instructions:
        # Get contour coordinates from the matrix
        contours = []
        valid_crop = True
        for i in range(0, len(indices), 2):
            row, col = indices[i], indices[i + 1]
            contour = matrix[row, col]

            if np.all(contour == -1):
                print(f"Warning: Skipping crop '{crop_name}' due to missing contour at matrix[{row},{col}].")
                valid_crop = False
                break

            contours.append(contour)

        if valid_crop:
            # Calculate inner rectangle coordinates
            if crop_name == "1.20":
                # Use only 3 contours for "1.20"
                c1, c2, c3 = contours
                x1 = max(c1[0] + c1[2], c2[0] + c2[2])  # Max of right-edges of contours 1 and 2
                y1 = c1[1] + c1[3] # bottom edge of contour 1
                x2 = c3[0]  # Min of left-edges of contours 3
                y2 = min(c2[1], c3[1])  # Min of top-edges of contours 2 and 3
            else:
                # Use 4 contours for other crops
                c1, c2, c3, c4 = contours
                x1 = max(c1[0] + c1[2], c3[0] + c3[2])  # Max of right-edges of contours 1 and 3
                y1 = max(c1[1] + c1[3], c2[1] + c2[3])  # Max of bottom-edges of contours 1 and 2
                x2 = min(c2[0], c4[0])  # Min of left-edges of contours 2 and 4
                y2 = min(c3[1], c4[1])  # Min of top-edges of contours 3 and 4

            # Crop and save
            if x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                output_path = os.path.join(output_dir, f"{image_name}_{crop_name}.jpg")
                cv2.imwrite(output_path, cropped_img)
                cut_images_count += 1
            else:
                print(f"Warning: Invalid inner rectangle for crop '{crop_name}'. Skipping.")

    print(f"Cropped {cut_images_count} images from {image_name}")

    


def create_cropped_labels(image_path, matrix, label_path, output_label_dir,crop_instructions=crop_instructions):
    """
    Creates cropped label files based on the cropped images and the original YOLOv8 label file.

    Args:
        image_path: Path to the original image.
        matrix: The 2D matrix of sorted contours.
        label_path: Path to the original YOLOv8 label file.
        crop_instructions: List of tuples defining the cropping logic.
        output_label_dir: Directory to save the cropped label files.
    """

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_label_dir, exist_ok=True)

    # Load original labels
    try:
        with open(label_path, 'r') as f:
            original_labels = [line.strip().split() for line in f]
    except FileNotFoundError:
        print(f"Error: Label file not found: {label_path}")
        return

    # Image dimensions (for converting normalized coordinates)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    img_height, img_width = img.shape[:2]

    # Create cropped labels for each crop instruction
    for crop_name, indices in crop_instructions:
        # Get contour coordinates from the matrix
        contours = []
        valid_crop = True
        for i in range(0, len(indices), 2):
            row, col = indices[i], indices[i + 1]
            contour = matrix[row, col]

            if np.all(contour == -1):
                print(f"Warning: Skipping crop '{crop_name}' due to missing contour at matrix[{row},{col}].")
                valid_crop = False
                break

            contours.append(contour)

        if valid_crop:
            # Calculate inner rectangle coordinates
            if crop_name == "1.20":
                # Use only 3 contours for "1.20"
                c1, c2, c3 = contours
                x1 = max(c1[0] + c1[2], c2[0] + c2[2])  # Max of right-edges of contours 1 and 2
                y1 = c1[1] + c1[3] # bottom edge of contour 1
                x2 = c3[0]  # Min of left-edges of contours 3
                y2 = min(c2[1], c3[1])  # Min of top-edges of contours 2 and 3
            else:
                # Use 4 contours for other crops
                c1, c2, c3, c4 = contours
                x1 = max(c1[0] + c1[2], c3[0] + c3[2])  # Max of right-edges of contours 1 and 3
                y1 = max(c1[1] + c1[3], c2[1] + c2[3])  # Max of bottom-edges of contours 1 and 2
                x2 = min(c2[0], c4[0])  # Min of left-edges of contours 2 and 4
                y2 = min(c3[1], c4[1])  # Min of top-edges of contours 3 and 4

            # Ensure valid rectangle
            if x1 < x2 and y1 < y2:
                # Create a new label file for the cropped region
                cropped_label_path = os.path.join(output_label_dir, f"{image_name}_{crop_name}.txt")
                with open(cropped_label_path, 'w') as out_file:
                    for label in original_labels:
                        class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, label)

                        # Convert normalized coordinates to absolute coordinates
                        x_center = x_center_norm * img_width
                        y_center = y_center_norm * img_height
                        width = width_norm * img_width
                        height = height_norm * img_height

                        # Check if the bounding box is within the cropped region
                        if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                            # Adjust coordinates relative to the cropped region
                            new_x_center = x_center - x1
                            new_y_center = y_center - y1
                            new_width = width
                            new_height = height

                            # Normalize coordinates for the cropped region
                            new_x_center_norm = new_x_center / (x2 - x1)
                            new_y_center_norm = new_y_center / (y2 - y1)
                            new_width_norm = new_width / (x2 - x1)
                            new_height_norm = new_height / (y2 - y1)

                            # Write the adjusted label to the new label file
                            out_file.write(f"{int(class_id)} {new_x_center_norm:.6f} {new_y_center_norm:.6f} {new_width_norm:.6f} {new_height_norm:.6f}\n")
            else:
                print(f"Warning: Invalid inner rectangle for crop '{crop_name}'. Skipping.")

def reverse_cropped_labels(cropped_label_dir, original_image_path, matrix, output_label_path, crop_instructions = crop_instructions):
    """
    Reverses the label cropping process, taking cropped labels and mapping them back to the original image's coordinate system.

    Args:
        cropped_label_dir: Directory containing the cropped label files.
        original_image_path: Path to the original (uncropped) image.
        matrix: The 2D matrix of sorted contours.
        crop_instructions: List of tuples defining the cropping logic.
        output_label_path: Path to save the reversed (original image) label file.
    """

    # Read the original image to get its dimensions
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Error: Could not read image at {original_image_path}")
        return
    img_height, img_width = img.shape[:2]

    # Create an empty list to store the reversed labels
    reversed_labels = []

    # Iterate through each crop instruction
    for crop_name, indices in crop_instructions:
        cropped_label_file = os.path.join(cropped_label_dir, f"{os.path.splitext(os.path.basename(original_image_path))[0]}_{crop_name}.txt")

        # Check if the cropped label file exists
        if not os.path.exists(cropped_label_file):
            print(f"Warning: Cropped label file not found: {cropped_label_file}. Skipping.")
            continue

        # Get the inner rectangle coordinates (x1, y1, x2, y2) for the current crop
        contours = []
        valid_crop = True
        for i in range(0, len(indices), 2):
            row, col = indices[i], indices[i + 1]
            contour = matrix[row, col]

            if np.all(contour == -1):
                print(f"Warning: Skipping crop '{crop_name}' due to missing contour at matrix[{row},{col}].")
                valid_crop = False
                break

            contours.append(contour)

        if not valid_crop:
            continue

        if crop_name == "1.20":
            c1, c2, c3 = contours
            x1 = max(c1[0] + c1[2], c2[0] + c2[2])
            y1 = c1[1] + c1[3]
            x2 = c3[0]
            y2 = min(c2[1], c3[1])
        else:
            c1, c2, c3, c4 = contours
            x1 = max(c1[0] + c1[2], c3[0] + c3[2])
            y1 = max(c1[1] + c1[3], c2[1] + c2[3])
            x2 = min(c2[0], c4[0])
            y2 = min(c3[1], c4[1])

        # Read the cropped labels from the file
        with open(cropped_label_file, 'r') as f:
            cropped_labels = [line.strip().split() for line in f]

        # Reverse the label coordinates for each cropped label
        for label in cropped_labels:
            class_id, x_center_norm_crop, y_center_norm_crop, width_norm_crop, height_norm_crop = map(float, label)

            # Convert normalized coordinates to absolute coordinates within the cropped region
            x_center_crop = x_center_norm_crop * (x2 - x1)
            y_center_crop = y_center_norm_crop * (y2 - y1)
            width_crop = width_norm_crop * (x2 - x1)
            height_crop = height_norm_crop * (y2 - y1)

            # Map the coordinates back to the original image
            x_center_orig = x_center_crop + x1
            y_center_orig = y_center_crop + y1
            width_orig = width_crop
            height_orig = height_crop

            # Normalize the coordinates with respect to the original image dimensions
            x_center_norm_orig = x_center_orig / img_width
            y_center_norm_orig = y_center_orig / img_height
            width_norm_orig = width_orig / img_width
            height_norm_orig = height_orig / img_height

            # Append the reversed label to the list
            reversed_labels.append(
                f"{int(class_id)} {x_center_norm_orig:.6f} {y_center_norm_orig:.6f} {width_norm_orig:.6f} {height_norm_orig:.6f}"
            )

    # Save the reversed labels to a new file
    with open(output_label_path, 'w') as out_file:
        for label in reversed_labels:
            out_file.write(label + "\n")

    print(f"Reversed labels saved to: {output_label_path}")