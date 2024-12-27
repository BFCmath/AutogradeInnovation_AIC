# AutogradeInnovation_AIC

## Idea
Details about the idea can be found in [idea.md](idea.md)

## Update

+ 16/12/2024:
  + Received the data and performed initial observations on both the dataset and the current publicly available solution.
+ 17/12/2024:
  + Make a simple image web reader to analyze the error in the labels.
  + Make assumptions on the error and confirm the error.
+ 18/12/2004:
  + Performed duplicates removing on the labels.
+ 19/12/2024:
  + Performed simple Yolo v8 fine-tuning.
  + Performed contours detecting.
+ 20/12/2024:
  + Continued with the contours detecting.
+ 21/12/2024:
  + Finished contours detecting.
+ 22/12/2024:
  + Open first testing phase.
  + Cutting images.
+ 23/12/2024:
  + Performed duplicates removing on the labels
  + Performed redundant removing on the labels
+ 24/12/2024:
  + Performed missings replacing on the labels
  + Performed 2-labeled bboxes replacing with true labels
  + Cutting the test set 1 images.
  + Yolo v8 fine-tuning on cropped images.
  + Inference the test set 1 images using fine-tuned model.
+ 25/12/2004:
  + Download the test set 2 images.

## My solution
### Training phase

NOTE: I already note all the progress in [data analysis](data_analysis.ipynb)

+ Detect the error in the labels (based on webs in [image_reader](image_reader/))
+ Clean the error in the labels (duplicates, redundants, missings, 2-labeled bboxes). You can check the work in [data_engineering](data_engineering/)
+ Train on Kaggle with Yolo v8. You can check the work in [model selection](model_selection.ipynb)

### Testing phase

+ Detect the contours using the function from the training phase, then cut the images.
+ Manually draw the contours for images that the algorithm cannot detect the contours.
+ Crop the images.
+ Inference using the fine-tuned model on the cropped images.
