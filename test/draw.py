import cv2
import os
import json

# Initialize global variables
boxes = []
start_point = None
end_point = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, boxes, resized_image, scale_factor, original_image_shape

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            temp_image = resized_image.copy()
            cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        
        # Draw the final rectangle on the resized image
        cv2.rectangle(resized_image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Image", resized_image)
        
        # Calculate coordinates relative to the original image
        x1_resized, y1_resized = start_point
        x2_resized, y2_resized = end_point

        # Scale coordinates back to original image size
        x1_original = int(x1_resized / scale_factor)
        y1_original = int(y1_resized / scale_factor)
        x2_original = int(x2_resized / scale_factor)
        y2_original = int(y2_resized / scale_factor)

        # Ensure coordinates are within image bounds
        x1_original = max(0, min(x1_original, original_image_shape[1] - 1))
        y1_original = max(0, min(y1_original, original_image_shape[0] - 1))
        x2_original = max(0, min(x2_original, original_image_shape[1] - 1))
        y2_original = max(0, min(y2_original, original_image_shape[0] - 1))

        # Convert (x1, y1, x2, y2) to (x, y, w, h)
        x = x1_original
        y = y1_original
        w = x2_original - x1_original
        h = y2_original - y1_original

        # Append the box as [x, y, w, h] in original image coordinates
        boxes.append([x, y, w, h])

def convert_to_json(boxes, json_path):
    """
    Saves bounding boxes to a JSON file as a list of lists.
    
    Parameters:
    - boxes (list of lists): List of bounding boxes in [x, y, w, h] format.
    - json_path (str): Path to the output JSON file.
    """
    with open(json_path, 'w') as f:
        json.dump(boxes, f, indent=4)

def main(image_path, output_json_path):
    global resized_image, scale_factor, original_image_shape

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    original_image_shape = image.shape  # (height, width, channels)
    original_height, original_width = original_image_shape[:2]

    # Resize the image to fit the screen
    screen_height = 800  # Set maximum screen height
    scale_factor = screen_height / original_height
    resized_width = int(original_width * scale_factor)
    resized_image = cv2.resize(image, (resized_width, screen_height))

    # Create a window and set the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    # Display the image
    cv2.imshow("Image", resized_image)

    cv2.imshow("Image", resized_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if not boxes:
                # Optionally, provide on-screen feedback using OpenCV
                pass
            else:
                # Save boxes to JSON
                convert_to_json(boxes, output_json_path)
            break

        elif key == ord('c'):
            # Clear all boxes and reset the image
            boxes.clear()
            resized_image = cv2.resize(image, (resized_width, screen_height))
            cv2.imshow("Image", resized_image)

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    # Define paths
    testset1_path = "data/testset1/images"
    image_filename = "IMG_3960_iter_0.jpg"  # Replace with your image file name
    image_path = os.path.join(testset1_path, image_filename)  # Path to the image
    output_json_path = os.path.join("test/manual_bbox", os.path.splitext(image_filename)[0][:8] + ".json")  # Output JSON file

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at {output_dir}")

    main(image_path, output_json_path)
