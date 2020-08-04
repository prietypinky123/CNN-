import warnings
warnings.filterwarnings("ignore")

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
TRAIN_DIR = 'train/train'
TEST_DIR = 'test/test'
IMG_SIZE = 50
LR = 1e-3
'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img[0]
  
    if word_label == 'h': return [1,0,0,0]
    
    elif word_label == 'b': return [0,1,0,0]
    elif word_label == 'v': return [0,0,1,0]
    elif word_label == 'l': return [0,0,0,1]

def create_train_data():                        #cretes a function create dataset
        # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
            # tqdm is only used for interactive loading 
    # loading the training data 
         #running a for loop. os.listdir returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order.
        label = label_img(img)
          # labeling the images 
        path = os.path.join(TRAIN_DIR,img)
        # Join one or more path components intelligently. The return value is the concatenation of TRAIN_DIR and any members of img with exactly
        #one directory separator following each non-empty part except the last, meaning that result will only end in a separator if the last part is empty.
        # If a component is an absolute path, all previous components are thrown away and joining continues from the absolute path component.
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        #resizes the image that we read earlier to the size specified. Here, we have passed IMG_SIZE as parameters, which we had declared in starting as. 50.
        # This will resize the image to 50 by 50 pixels.
        training_data.append([np.array(img),np.array(label)])
        # final step-forming the training data list with numpy array of the images
    shuffle(training_data)
     # shuffling of the training data to preserve the random state of our data
    np.save('train_data.npy', training_data)
     # saving our trained data for further uses if required 
    return training_data

def process_test_data():
    '''Processing the given test data'''
# Almost same as processing the training data but 
# we dont have to label it.
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
'''Running the training and the testing in the dataset for our model'''

# If you have already created the dataset:
#train_data = np.load('train_data.npy')

'''Creating the neural network using tensorflow'''
# Importing the required libraries 
import tflearn
"""TFlearn is a modular and transparent deep learning library built on top of Tensorflow.
It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it."""
from tflearn.layers.conv import conv_2d, max_pool_2d
"""TFLearn brings "layers" that represent an abstract set of operations to make building neural networks more convenient. For example, a convolutional layer will:

Create and initialize weights and biases variables
Apply convolution over incoming tensor
Add an activation function after the convolution etc"""
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
"""Clears the default graph stack and resets the global default graph.

NOTE: The default graph is a property of the current thread. This function applies only to the current thread. """
#What we have here is a nice, 2 layered convolutional neural network, with a fully connected layer, and then the output layer. 

""" input_data is a placeholder for input features. The array you mention holds first None (always), then IMG width and height (seems the image is squared since width=height) and channels
(in this case is 1; ex.: in case of RGB you would get 3 channels). This way the net gets to know the dimensions of input features."""
#Now we are going to begin building the convolutional neural network, starting with the input layer:
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

#Now to create the model:
model = tflearn.DNN(convnet, tensorboard_dir='log')

"""Max pooling is used to reduce the image size by mapping the size of a given window into a single result by taking the maximum value of the elements in the window."""

"""
Activation Function: It's just a thing(node) that you add to the output end of any neural network.
It is also known as Transfer Function. Activation functions are used to determine the output of neural network like yes or no.
It maps the resulting values in between 0 to 1 or -1 to 1 etc. depending on the function.

ReLU (Rectified Linear Unit) Activation Function: The ReLU is the most used activation function on the world right now.
Since, it is used in almost all the convolutional neural networks or deep learning. T
he ReLU is half rectified. f(z) is zero when z is less than zero and f(z) is z w

"""
#to save and reload changes everytime
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    #os.path.exists() method in Python is used to check whether the specified path exists or not. This method can be also used to check whether the given path refers to an open file descriptor or not.
    model.load(MODEL_NAME)
    print('model loaded!')
    
# Splitting the testing data and training data 
train = train_data[:-500]
test = train_data[-500:]

'''Setting up the features and lables'''
# X-Features & Y-Labels 
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

'''Fitting the data into our model'''
# epoch = 8 taken 
model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)




"""Next, a convolutional layer with 32 filters and stride = 8 is created. Stride is the number of pixels shifts over the input matrix. When the stride is 1 then we move the filters to 1 pixel at a time.  The activation function is ReLU. Right after that, a max pool layer is added.
That same trickery is repeated again with 64 filters.
A convolution is how the input is modified by a filter. In convolutional networks, multiple filters are taken to slice through the image and map them one by one and learn different portions of a
n input image. ... Each time a match is found, it is mapped out onto an output image.“filters” what they're referring to is the learned weights of the convolutions.


The tf.reshape does not change the order of or the total number of elements
in the tensor, and so it can reuse the underlying data buffer. This makes it a fast operation independent of how big of a tensor it is operating on.
Next, a fully-connected layer with 1024 neurons is added. Finally, a dropout layer with keep probability of 0.8 is used to finish our model
The reason why softmax is useful is because it converts the output of the last layer in your neural network into what is essentially a probability distribution..

epoch = no of iteration


Convolutions have been used for a long time in image processing to blur and
sharpen images, and perform other operations, such as, enhance edges and emboss.
 The area where the filter is on the image is called the receptive field.

 We use Adam as optimizer with learning rate set to 0.001
 . Our loss function is categorical cross entropy.
 Finally, we train our Deep Neural Net for 10 epochs.

Working: Conv2D filters extend through the three channels in an image (Red, Green, and Blue). The filters may be different for each channel too.
After the convolutions are performed individually for each channels, they are added up to get the final convoluted image.
The output of a filter after a convolution operation is called a feature map."""










        
