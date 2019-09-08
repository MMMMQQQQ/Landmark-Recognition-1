# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import shutil
import os
import keras
# from collections import Counter
# import numpy as np
# import matplotlib.pyplot as plt
# from IPython.display import Image
# from IPython.core.display import HTML 

# %matplotlib inline

# Essential working directories
WORK_DIR = '/Users/matil/Google Drive/images/'
IMG_DIR = WORK_DIR + 'images/'
TRAIN_DIR = IMG_DIR + 'train/'
VALIDATE_DIR = IMG_DIR + 'validate/'
TEST_DIR = IMG_DIR + 'test/'

def splitDataset(datasetPath):
        # Load dataset CSV which contains the landmark IDs and the image IDs of the landmarks
        df = pd.read_csv(datasetPath, delimiter=';')

        # define the image ID as the feature
        x = df['id']
        # and the landmark ID as the label
        y = df['landmark_id']

        # Split dataset to train/validate & test
        from sklearn.model_selection import StratifiedShuffleSplit

        testSplitter = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=0)

        for trainIndex, testIndex in testSplitter.split(x, y):
                xTrainVal, xTest = x[trainIndex], x[testIndex]
                yTrainVal, yTest = y[trainIndex], y[testIndex]

        xTrainVal = xTrainVal.reset_index(drop=True)
        yTrainVal = yTrainVal.reset_index(drop=True)

        # Further split train/validate dataset to seperate train & validate
        validationSplitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        for trainIndex, validationIndex in validationSplitter.split(xTrainVal, yTrainVal):
                xTrain, xValidate = xTrainVal[trainIndex], xTrainVal[validationIndex]
                yTrain, yValidate = yTrainVal[trainIndex], yTrainVal[validationIndex]

        print('x shape: {}'.format(x.shape))
        print('y shape: {}'.format(y.shape))
        print()

        print('xTrainVal shape: {}'.format(xTrainVal.shape))
        print('yTrainVal shape: {}'.format(yTrainVal.shape))
        print()

        print('xTrain shape: {}'.format(xTrain.shape))
        print('yTrain shape: {}'.format(yTrain.shape))
        print()

        print('xValidate shape: {}'.format(xValidate.shape))
        print('yValidate shape: {}'.format(yValidate.shape))

        print()
        print('xTest shape: {}'.format(xTest.shape))
        print('yTest shape: {}'.format(yTest.shape))

        return xTrain, yTrain, xValidate, yValidate, xTest, yTest




# After splitting the dataset, move the files to train/validate/test directories. 
# In order to use keras.flow_from_directory we must further split the 3 datasets into directories
# per class (or label) 
def CreateImageDirectories(xTrain, yTrain, xValidate, yValidate, xTest, yTest):
        print('Moving train images')
        for idx, item in xTrain.iteritems():
                classDir = TRAIN_DIR + str(yTrain[idx]) + '/'
                if not os.path.exists(TRAIN_DIR):
                        os.mkdir(TRAIN_DIR)
                if not os.path.exists(classDir):
                        os.mkdir(classDir)
                try:
                        # print('{}{}.jpg'.format(IMG_DIR, item), '{}{}.jpg'.format(classDir, item))
                        shutil.move('{}{}.jpg'.format(IMG_DIR, item), '{}{}.jpg'.format(classDir, item))
                except Exception as ex:
                        print('Failed to move {}: {}'.format(item + '.jpg', ex))

        print('Moving validation images')
        for idx, item in xValidate.iteritems():
                classDir = VALIDATE_DIR + str(yValidate[idx]) + '/'
                if not os.path.exists(VALIDATE_DIR):
                        os.mkdir(VALIDATE_DIR)
                if not os.path.exists(classDir):
                        os.mkdir(classDir)
                try:
                        # print('{}{}.jpg'.format(IMG_DIR, item), '{}{}.jpg'.format(classDir, item))
                        shutil.move('{}{}.jpg'.format(IMG_DIR, item), '{}{}.jpg'.format(classDir, item))
                except Exception as ex:
                        print('Failed to move {}: {}'.format(item + '.jpg', ex))

        print('Moving test images')
        for idx, item in xTest.iteritems():
                classDir = TEST_DIR + str(yTest[idx]) + '/'
                if not os.path.exists(TEST_DIR):
                        os.mkdir(TEST_DIR)
                if not os.path.exists(classDir):
                        os.mkdir(classDir)
                try:
                        # print('{}{}.jpg'.format(IMG_DIR, item), '{}{}.jpg'.format(classDir, item))
                        shutil.move('{}{}.jpg'.format(IMG_DIR, item), '{}{}.jpg'.format(classDir, item))
                except Exception as ex:
                        print('Failed to move {}: {}'.format(item + '.jpg', ex))


if False:
        print('Splitting dataset')
        xTrain, yTrain, xValidate, yValidate, xTest, yTest = splitDataset(WORK_DIR + 'dataset.csv')
        print('Creating directory hiearchy')
        CreateImageDirectories(xTrain, yTrain, xValidate, yValidate, xTest, yTest)


# Encode labels into categories via the one hot encoding method
# from sklearn.preprocessing import OneHotEncoder
# NOT NEEDED!! - It is done by keras

# yEncoder = OneHotEncoder(sparse=False, categories='auto')
# yTrainCategories = yEncoder.fit_transform(yTrain.values.reshape(-1, 1))
# yValidateCategories = yEncoder.fit_transform(yValidate.values.reshape(-1, 1))
# yTestCategories = yEncoder.fit_transform(yTest.values.reshape(-1, 1))

# Load all the images into numpy arrays via keras
from keras.preprocessing.image import ImageDataGenerator

# Training ImageDataGenerator will do some on-the-fly data augmentation to further enrich the sample
trainDataGenerator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test and validation ImageDataGenerators don't need data augmentation
testDataGenerator = ImageDataGenerator(rescale=1./255)

# Load the data from the directories. 
# By setting class_mode='categorical' the data generator does the one-hot encoding for us
print('Loading training images')
trainGenerator = trainDataGenerator.flow_from_directory(TRAIN_DIR, 
                                                        target_size=(150, 150), 
                                                        batch_size=32, 
                                                        class_mode='categorical')
print('Loading validation images')
validateGenerator = trainDataGenerator.flow_from_directory(VALIDATE_DIR, 
                                                        target_size=(150, 150), 
                                                        batch_size=32, 
                                                        class_mode='categorical')
                                                        
print('Loading test images')
testGenerator = trainDataGenerator.flow_from_directory(TEST_DIR, 
                                                        target_size=(150, 150), 
                                                        batch_size=1,
                                                        class_mode=None)
