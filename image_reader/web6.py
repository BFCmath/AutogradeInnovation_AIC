from flask import Flask, render_template_string, Response, request, send_from_directory
import os
import cv2
import pandas as pd
import numpy as np
import sys

# Add parent directory to Python path (if needed for your project structure)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flask app setup
app = Flask(__name__)

# Paths (Updated)
image_dir = "data/cropped_images"  
label_dir = "data/std_num_bbox_labels" 
data_csv_path = "data/std_num_bbox_data.csv"

# Load image data from CSV
def load_data(data_csv_path):
    data_df = pd.read_csv(data_csv_path)
    return data_df

# Function to draw bounding boxes and labels
def draw_bounding_boxes_and_labels(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Draw labels if label file exists
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.readlines()
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            
            # Convert normalized coordinates to pixel values
            x_center = int(x_center * img.shape[1])
            y_center = int(y_center * img.shape[0])
            width = int(width * img.shape[1])
            height = int(height * img.shape[0])
            
            # Calculate top-left corner
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)
            
            # Draw bounding box (different color for each class)
            if class_id == 0:
                color = (255, 0, 0)  # Blue for class 0
            elif class_id == 1:
                color = (0, 255, 0)  # Green for class 1
            elif class_id == 2:
                color = (0, 0, 255)  # Red for class 2
            else:
                color = (0, 0, 0) # Black for other classes

            cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
            
            # Add label text
            cv2.putText(
                img,
                f"{int(class_id)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

# HTML template
template = """
<!doctype html>
<html lang="en">
<head>
    <title>Image Viewer</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; }
        .image-container { display: inline-block; margin: 10px; text-align: center; border: 1px solid #ddd; padding: 5px; }
        img { max-width: 400px; max-height: 400px; }
        form { margin-bottom: 20px; }
        .label-text { font-size: 14px; color: #333; }
    </style>
</head>
<body>
    <h1>Image Viewer</h1>
    <p>Filter images based on image prefix and type.</p>
    <form method="POST">
        <label for="image_prefix">Image Prefix:</label>
        <select id="image_prefix" name="image_prefix">
            <option value="">All</option>
            {% for prefix in image_prefixes %}
                <option value="{{ prefix }}" {% if prefix == selected_image_prefix %}selected{% endif %}>{{ prefix }}</option>
            {% endfor %}
        </select>
        <label for="type">Type:</label>
        <select id="type" name="type">
            <option value="">All</option>
            {% for type in types %}
                <option value="{{ type }}" {% if type == selected_type %}selected{% endif %}>{{ type }}</option>
            {% endfor %}
        </select>
        <button type="submit">Filter Images</button>
    </form>
    <hr>
    {% for img_data in images_data %}
        <div class="image-container">
            <p>{{ img_data.image_name }}</p>
            <img src="{{ url_for('render_image', filename=img_data.image_name) }}" alt="Image">
            <p class="label-text">Number of Labels: {{ img_data.number_label }}</p>
        </div>
    {% endfor %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    data_df = load_data(data_csv_path)

    # Extract unique image prefixes and types (modify as per your CSV structure)
    image_prefixes = sorted(data_df["org_name"].str.replace(r"_iter_.*$", "", regex=True).unique())
    types = sorted(data_df["type"].unique())

    # Get filter values
    selected_image_prefix = request.form.get("image_prefix")
    selected_type = request.form.get("type")

    # Filter DataFrame
    filtered_df = data_df.copy()
    if selected_image_prefix:
        filtered_df = filtered_df[
            filtered_df["org_name"].str.startswith(selected_image_prefix)
        ]
    if selected_type:
        filtered_df = filtered_df[filtered_df["type"] == selected_type]

    # Default filter (if no filters selected)
    if not selected_image_prefix and not selected_type:
        filtered_df = filtered_df[filtered_df["org_name"].str.contains("_iter_0")]

    # Prepare data for rendering
    images_data = []
    for _, row in filtered_df.iterrows():
        images_data.append(
            {
                "image_name": row["image_name"],
                "number_label": row["number_label"], 
            }
        )

    return render_template_string(
        template,
        image_prefixes=image_prefixes,
        selected_image_prefix=selected_image_prefix,
        types=types,
        selected_type=selected_type,
        images_data=images_data,
    )

@app.route('/<path:directory>/<path:filename>')
def serve_file(directory, filename):
    """Serve files from the specified directory."""
    return send_from_directory(directory, filename)

@app.route("/render_image/<path:filename>")
def render_image(filename):
    image_path = os.path.join(image_dir, filename)
    label_filename = filename.replace(".jpg", ".txt")
    label_path = os.path.join(label_dir, label_filename)

    img_bytes = draw_bounding_boxes_and_labels(image_path, label_path)
    if img_bytes:
        return Response(img_bytes, mimetype="image/jpeg")
    else:
        return "Image not found or could not be processed.", 404

if __name__ == "__main__":
    app.run(debug=True)