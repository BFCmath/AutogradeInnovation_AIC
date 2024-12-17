# Idea

## TODO

+ [X] Load data
+ [X] Observe data
+ [X] Find current solution in public (not dive deep yet)
+ [X] Save the training set into a csv
+ [ ]  Process the error in the labels
  + [X] Detect the error
  + [X] Make a web reader to analyze the error
  + [ ] Predict and confirm the error
  + [ ] Process the error
+ [ ] Dive deep into public solution
+ [ ] Propose solutions
+ [ ] Build pipeline
+ [ ] Try opencv approach (augmentation)

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
+ Bounding boxes is not even
