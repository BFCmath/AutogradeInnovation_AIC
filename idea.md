# IDEA

## TODO

+ [X] Load data
+ [X] Observe data
+ [X] Find current solution in public (not dive deep yet)
+ [X] Save the training set into a csv
+ [X] Dive deep into public solution
+ [X] Propose solutions
+ [X] Process the error in the labels
  + [X] Detect the error
  + [X] Make a web reader to analyze the error
  + [X] Predict and confirm the error
  + [X] Cutting images
    + [X] Detect the contours
    + [X] Sort the contours
  + [X] Process the error
    + [X] Remove the duplicates
    + [X] Remove the redundants
    + [X] Replace the missings
  + [X] Replace the 2-labeled bboxes with true labels
+ [ ] Build pipeline
  + [X] Detecting the contours
  + [X] Sorting the contours
  + [X] Cutting the images
  + [ ] End-to-end pipeline
+ [ ] Finetune the model
  + [ ] Augment the data
  + [X] Yolov8
  + [ ] Analyze the error
+ [ ] Predict testing set 1
  + [X] Detect the contours
  + [X] Sort the contours
  + [X] Cut the images
  + [ ] Inference the images using fine-tuned model

## Observations

### First observation on the dataset

+ There are 5200 images in first trainging batch
+ Training set will give us:
  + raw images
  + labeled images
  + yolo labels
+ Testing set will only give us raw images
+ There are some repetitive error in the training labels (that we need to process ourselves)
+ The images are not even (probably due to scanning process)
+ Duplicate choices in the same question
+ No choice in some question

### Error in the labels

+ Missing some bounding boxes
+ Duplicate bounding boxes
+ Redundant bounding boxes
+ Bounding boxes is not even

### Second observation on the dataset

+ It seems that all the images have the same format
+ The trainging data is split into each batch of 260 iters
+ Within the same batch, the images have the same type of error

### Third observation on the dataset

+ The bounding boxes are not even
+ We can using the black squares to get relative position of each bounding box window.
+ There are yellow streaks in some images

### Cutting images

+ Based on the black squared contours at the corners, we can effectively detect them and cutting the images.
+ But there are still some error from the images so we cannot rely completely on the contour detective algorithm.
+ So we will use some default pivot to check if the algorithm detecting the contour in those regions.
+ After using default pivot, if there are still some undetected contours, we will use stack appoarch to detect the contours (without using the pivot)
+ After that, if there are still some undetected contours, we will manually detect the contours based on rules.
+ And after all of that, we can finally cut the images based on some rules we have defined.
+ And also, we need to convert the position of bubbles so that it has yolov8 format for each cropped images.

### Forth observation on the dataset

+ Some duplicate bubbles have different class.

### Processing the error

+ Duplicated bounding boxes:
  + We can use the IOU to detect these error bounding boxes.
+ Redundant bounding boxes:
  + We can visuallize and define the rules to remove these error bounding boxes.
+ Missing bounding boxes:
  + We can visuallize and define the rules to replace these error bounding boxes.
  