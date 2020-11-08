# car brand classfication
HW1 for NCTU CS Selected Topics in Visual Recognition using Deep Learning

16,185 car images belonging to 196 classes (train: 11,185, test:5000)

all the code are save in the [hw1.ipynb](https://github.com/sweiichen/car-brand-classification/blob/main/hw1.ipynb) notebook.


## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- GeForce GTX 1080 Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)

## Installation
All requirements should be detailed in  requirements.txt. 
You have to create your own python virtual environment.
- python version: 3.7.7 
- cuda version: 10.1.243

```
pip install -r requirements.txt
```

## Dataset Preparation
Download the dataset from https://www.kaggle.com/c/cs-t0828-2020-hw1/data
and put training_data folder and testing_data folder into this repo


### Prepare Images
Because the original dataset provided, only give the csv file to show the class label of each image.
I make them be imagefolder which could use the pytorch ImageFolder function.
using the python code in notebook to mount the images in their label_index folder.
you need to use the [training_label.csv](https://github.com/sweiichen/car-brand-classification/blob/main/training_label.csv) file in our repo to run the code.
"car_dataset" folder are empty of images befor you run the code.
you could see the detail in the jupyter note book code.
After make images to imagefolder that torchvision.datasets.ImageFolder() can use, the data directory is structured as:
```
car_dataset
  +- train
  |  +- 0
  |  +- 1
  |  +- 2
  |  +- .
  |  +- .
  |  +- .
  +- test
  |  +- 0
  |  +- 1
  |  +- 2
  |  +- .
  |  +- .
  |  +- .

```




### Split Dataset
you can also use the function split_data() in the notebook to split the training images to train set and validation set by different proportion.



## Training
Follow the notebook steps and comments, you can successsfully start to train the model.
you might chage the bath size, according to you GPU memory size.
The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet152 | 1x TitanX | 400 | 10 | 1.5 hours

the test accuracy after 10 epochs could reach about 0.92

### Pretrained models
I use the pretrained resnet152 model provide by pytorch to do this task.

### load trained parameters
you can alsodirectly load the trained model parameters without retraining again.
```python=
checkpoint = torch.load('model/resnet152.pth')
model_ft.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

## Inference
Follow to the note book comment and predict the class of the test image from the kaggle dataset. 
- To inference, [classes.txt](https://github.com/sweiichen/car-brand-classification/blob/main/classes.txt) file will be used to map the label_indext to car brand.



