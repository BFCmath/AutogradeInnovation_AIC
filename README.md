# AutogradeInnovation_AIC

## Competition Information
For more details about the competition, please refer to [this notebook](info/competition_info.ipynb).

## Idea
Details of the core idea can be found in [idea.md](idea.md).  
Additionally, the inspiration for this idea is documented in [this notebook](info/approach_research.ipynb).

## Progress Updates

- **16/12/2024**  
  - Received the dataset and made initial observations on both the data and the publicly available solution.

- **17/12/2024**  
  - Created a simple web-based image reader to identify labeling errors.  
  - Formulated assumptions about these errors and validated them.

- **18/12/2004**  
  - Removed duplicate labels.

- **19/12/2024**  
  - Conducted a basic YOLOv8 fine-tuning.  
  - Began contour detection.

- **20/12/2024**  
  - Continued contour detection work.

- **21/12/2024**  
  - Completed the contour detection process.

- **22/12/2024**  
  - Began the first testing phase.  
  - Performed image cropping.

- **23/12/2024**  
  - Further removal of duplicate labels.  
  - Removed redundant labels.

- **24/12/2024**  
  - Addressed missing labels.  
  - Replaced bounding boxes that had two labels with the correct label.  
  - Cropped images for Test Set 1.  
  - Fine-tuned YOLOv8 on these cropped images.  
  - Ran inference on Test Set 1 with the fine-tuned model.

- **25/12/2004**  
  - Downloaded images for Test Set 2.

## My Solution

### Training Phase

> **Note:** All the progress is also documented in [data_analysis.ipynb](data_analysis.ipynb).

1. **Label Error Detection**  
   Used a simple web-based image reader (found in [image_reader](image_reader/)) to identify labeling errors.  

2. **Label Error Correction**  
   Cleaned the labels by removing duplicates, redundancies, and missing entries, and by replacing bounding boxes that had multiple labels. All related work is in the [data_engineering](data_engineering/) folder.

3. **Model Training**  
   Trained a YOLOv8 model on Kaggle. You can find the configurations and training details in [model_selection.ipynb](model_selection.ipynb).

### Testing Phase

1. **Contour Detection & Cropping**  
   Used the previously defined contour detection function to crop images.  
   For images where contours were not automatically detected, contours were drawn manually.

2. **Inference**  
   Cropped images were then passed through the fine-tuned model for final inference.

## File Structure

- **[data_analysis.ipynb](data_analysis.ipynb)**  
  Contains data exploration and ongoing progress notes.

- **[image_reader](image_reader/)**  
  A simple web-based viewer that displays images and their labels.

- **[data_engineering](data_engineering/)**  
  Includes label error corrections and image processing workflows:  
  - [contours_detecting.ipynb](data_engineering/contours_detecting.ipynb): Detects image contours.  
  - [contours_sorting.ipynb](data_engineering/contours_sorting.ipynb): Sorts detected contours.  
  - [image_cutting.ipynb](data_engineering/image_cutting.ipynb): Crops images based on identified contours.  
  - [label_processing.ipynb](data_engineering/label_processing.ipynb): Handles label errors (duplicates, missing entries, redundancies).  
  - [label_correting.ipynb](data_engineering/label_correting.ipynb): Fixes bounding boxes with multiple labels.

- **[model_selection.ipynb](model_selection.ipynb)**  
  Documents the model choice for the competition along with training configurations and processes.
