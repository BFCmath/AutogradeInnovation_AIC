import json
import os
from tqdm import tqdm
import cv2
import json
import os
import numpy as np

def sort_contours_horizontal(contours):
    return sorted(contours, key=lambda c: c[0])

def sort_contours_vertical(contours):
    return sorted(contours, key=lambda c: c[1])
def check_contours_in_same_line(line, tolerance_percentage=0.1):
    """
    Checks if contours in a line are approximately in the same line vertically.

    Args:
        line: A list of contours, where each contour is a list [x, y, w, h].
        tolerance_percentage: The allowed percentage difference in y-coordinates to be considered in the same line.

    Returns:
        True if the contours are approximately in the same line, False otherwise.
    """
    if not line:
        return True  # Empty line is considered valid

    # Get the y-coordinates and heights of all contours in the line
    y_coords = np.array([c[1] for c in line])
    heights = np.array([c[3] for c in line])

    # Calculate the average y-coordinate and average height
    avg_y = np.mean(y_coords)
    avg_height = np.mean(heights)

    # Calculate the tolerance based on the average height and tolerance percentage
    tolerance = avg_height * tolerance_percentage

    # Check if the y-coordinate of each contour is within the tolerance of the average y-coordinate
    for y in y_coords:
        if abs(y - avg_y) > tolerance:
            return False

    return True

def sort_contours(contours, tolerance_percentage=0.8):
    """
    Sorts contours first vertically (top to bottom) and then horizontally (left to right) within each line,
    with a check to ensure contours in a line are approximately aligned.

    Args:
        contours: A list of contours, where each contour is a list [x, y, w, h].
        tolerance_percentage: The allowed percentage difference in y-coordinates to be considered in the same line.

    Returns:
        A list of lists, where each inner list represents a line of sorted contours.
        Returns None if any line fails the alignment check.
    """

    # 1. Sort contours by y-coordinate (top to bottom)
    sorted_contours = sort_contours_vertical(contours)

    # 2. Group contours into lines based on predefined line sizes
    lines = []
    line_sizes = [2, 2, 2, 4, 5, 9, 7]  # Number of contours in each line
    start_index = 0
    for line_size in line_sizes:
        end_index = start_index + line_size
        line = sorted_contours[start_index:end_index]

        # 3. Check if contours in the line are approximately aligned
        if not check_contours_in_same_line(line, tolerance_percentage):
            print(f"Warning: Contours in a line are not approximately aligned vertically (tolerance: {tolerance_percentage * 100}%).")
            # You might want to handle this differently, e.g., by splitting the line further
            # or returning None to indicate an issue with the input data
            # for now i will stop the program
            return None  

        lines.append(line)
        start_index = end_index

    # 4. Sort contours horizontally (left to right) within each line
    sorted_lines = []
    for line in lines:
        sorted_line = sort_contours_horizontal(line)
        sorted_lines.append(sorted_line)

    return sorted_lines

def process_contour_files(contours_dir, output_dir):
    """
    Processes contour JSON files, sorts the contours, and saves the sorted contours to a text file.

    Args:
        contours_dir: Directory containing the contour JSON files.
        output_dir: Directory to save the sorted contours text files.
    """

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(contours_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(contours_dir, filename)
            with open(filepath, 'r') as f:
                contours = json.load(f)

            # Ensure exactly 31 contours
            if len(contours) != 31:
                print(f"Warning: File {filename} does not contain exactly 31 contours. Skipping.")
                continue

            # Sort contours
            sorted_lines = sort_contours(contours)

            # Save sorted contours to a text file
            output_filename = os.path.splitext(filename)[0] + "_sorted.txt"
            output_filepath = os.path.join(output_dir, output_filename)
            with open(output_filepath, 'w') as outfile:
                for line in sorted_lines:
                    for contour in line:
                        outfile.write(f"{contour[0]} {contour[1]} {contour[2]} {contour[3]}\n")
                    outfile.write("\n")  # Add an extra newline to separate lines visually