---
layout: single
permalink: capstone/
title: &title "Capstone Project"
author_profile: true
toc: true
toc_label: ON THIS PAGE
---
<img src="/assets/images/capstone_banner.jpg" width="100%">

## 1. Background

We've all done our shopping online. From electronics, to household items, and even pantry items. I would believe that it would be safe to say that many of us are well versed buying online from websites, both local and foreign.

Every time we go looking for something, there are seems to be a variety of what you are looking for. How does one decide which item to buy, since item A and B do the same thing, and are at the same price?

We no longer make our purchasing decisions in the blind. We rely on product reviews. Any shopper reads approximately 10 reviews before deciding if they would buy it. A review is a testimony. It's a proven track record of the product by previous buyer's recommendations or warnings.

<img src="/assets/images/capstone_review.jpg" width="100%">

The value that product reviews bring on any e-commerce platform:
* For the **buyer**: Better decision making. Merchant/Product trust. Recommended.
* For the **merchant**: Helps sell more. Better credibility. Informative experience.
* For the **platform**: More business and traction. Platform becomes the 'Go-to' place for shoppers.
* Adds to the **customer experience**

Many products receive reviews both good and bad. However it is the 'Helpful' reviews that are the decision influencing factors.

### 1.1. Understanding Reviews

Reviews are often separated into 2 groups. Helpful or Unhelpful. Helpful reviews help buyers decide if they should purchase. Reviews also shared experiences from a previous buyer. Unhelpful reviews have little or no relevance to decision making, infact they often ignored.

Most e-commerce platforms allow the reviews to be voted on.

<img src="/assets/images/capstone_eg.jpg" width="100%">

## 2. Business Case
**Actionable Insights**
What can we do if we could determine that a review would be:
<img src="/assets/images/capstone_helpful.jpg" width="100%">
_**Helpful**_:
* It could be put in the front of buyers on a product page to help potential buyers make a decision.
* Potential buyers could still vote if that review was helpful to them or not.
<img src="/assets/images/capstone_unhelpful.jpg" width="100%">
_**Unhelpful**_:
* The merchant or platform could follow up with the reviewer, seeking recourse or more information.
* Alternative trigger for customer satisfaction issues.

**Ultimately, we can provide a better customer experience on and off the platform.**


### 2.2 Non-Linear Activation

Activation functions are an extremely important elements of neural networks. They basically decide whether a pixel in a feature should be passed on deeper in a network or "killed". One of the most commonly used activation function is the Rectified Linear Unit (ReLU), which is an element wise operation (applied per pixel) and replaces all negative values in the feature map by zero. 

ReLU's popularity stems from the fact it is computationally easier to calculate, resulting in faster training time, and acheiving the objective of introducing non-linearity in the model. I will not go into the significance of non-linearity here, if you are interested this [page]() has provided quite a good explanation. 

### 2.3 Pooling

Image matrices tend to be large, and the situation quickly becomes unmanageable as image count and sizes grow larger. 

To counter this, pooling techniques are often applied to reduce the dimension of our feature maps, while retaining the most important information. The most common pooling technique used is max pooling due to its performance. 

In max pooling, a N x N matrix is slided through our feature map, and only the largest element is retained for every step. The figure below illustrates how max pooling is performed with a 2x2 matrix:

<img src="/assets/images/maxpool.gif" width="70%">

### 2.4 Fully Connected Layer
The final part to a CNN is always a Fully Connected Layer. The term “fully connected” implies that every neuron in the previous layer is connected to every neuron on the next. The Fully Connected layer is a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer. This softmax activation function works by assigning a probability to each of the training category, such that it adds up to 1. The category with the highest probability will then be the prediction.  

Think of this part like a puzzle. Prior to this, the features extracted can be thought of as pieces to the puzzle. One then has to join all of these pieces before you can get and comprehend the image.

Now that we have an understanding on how CNN works, its time to move on to our star today Joe. Read on to find out how he is built!

## 3. Methodology
The approach to building Joe can be split into 4 main parts:
1. Mining the dataset
2. Pre-process the images
3. Buildning and training the CNN
4. Evaluating results

### 3.1 Dataset
The dataset used for this project are obtained from 2 sources:
* [VegFru](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_VegFru_A_Domain-Specific_ICCV_2017_paper.pdf) database from University of Science and Technology China.  
* Web scraped from the internet (Primary Source)

A total of 15 fruits and vegetables are selected for training Joe:
1. Apple
2. Asparagus
3. Banana
4. Broccoli
5. Cabbage
6. Capsicums
7. Cherry
8. Cherry Tomato
9. Chilli
10. Grapes
11. Mango
12. Pak Choi
13. Soursop
14. Spinach
15. Strawberry

Each category contains at least 2,000 images to ensure sufficient representation, for a grand total of 37,000 images used. 

It is highly recommended that your images contain a good mix of clean and noisy images. The rationale behind is clean images will train the machine to pinpoint the object of interest, whereas noisy images will train the machine for real world application.

Sample Images:
<img src="/assets/images/Sample.png" width="100%">

Main packages used for Joe:
1. Numpy
2. PIL
3. os, shutil
5. matplotlib
6. sci-kit learn
7. keras with tensorflow-gpu backend


### 3.2 Preprocessing Images
The preprocessing stage can be split into 3 main steps:
1. Manual review and removal of irrelevant images
2. Cropping and renaming of images with Python for easy reference
3. Train-validate-test split with Python

For point 3, 100 images are randomly selected from each category for testing. A 80/20 split is done randomly on the remaining images for training and validation. I shall not go into details here, and the codes to above can be found on my github.


## 4. Building Joe with Keras

Now its time to go into the architecture of Joe!

### 4.1 Defining Joe's Architecture
 
```python
input_shape = (128, 128, 3)
num_classes = 15

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
```

To understand the model above, each Convolution2D() defines a single convolution layer. Using the first layer as an example:
* 32 refers to the number of filters used
* (3,3) refers to the size of the filter
* Activation function to use is ReLU
* Input shape of (128,128,3) refers to the width, height, number of colour channels of the input image

Notice that there is a parameter called padding. Recall the earlier illustration on filters in section 2.1, the resulting feature map is always smaller than that of the original. A 'same' padding will ensure the input and output of the convolution layer to be of the same size through a technique called zero padding.

We have also added various Dropout layers in our Keras model. At each dropout layer, a portion of the features extracted will be randomly discarded based on the fraction defined. The Dropout layer acts like a regulariser to prevent overfitting of Joe.


### 4.2. Training Joe

Keras has provided a very useful ImageDataGenerator class that defines the configuration for image data preparation and augmentation. This allows images to be augmented real time on the fly before being pass into the CNN, which has an effect of artificially increasing your dataset. 

Each specification defined this class can potentially increase your dataset by 100%, which is especially useful when your dataset is small. For a full list of options, you may refer to [Keras documentation](https://keras.io/preprocessing/image/).

```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 40,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validate_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(Training_images,
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validate_set = validate_datagen.flow_from_directory(Validation_images,
                                                    target_size = (128, 128),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
```

Joe is then trained by calling the fit generator method from Keras. We train Joe over 200 epochs and a batch size of 32. To put it simply, an epoch is considered completed when the machine has run through your entire dataset once.

```python
hist = model.fit_generator(training_set,
                           steps_per_epoch = (training_count//batch_size),
                           epochs = 200,
                           validation_data = validation_set,
                           validation_steps = (validation_count//batch_size),  
                           callbacks = [csv_log, checkpointer])
```

### 4.3 Model Evaluation
After training, we will test our model on the 1,500 test images selected during the pre-processing stage, which are images Joe has never seen before. This is crucial as it checks if our model is good at generalising new images and not just on images it is trained on i.e. overfitting.

We will choose the model weights at epoch 125 for the test as it has the best loss and accuracy, and use the f1-score as the key metric to evaluate Joe.

**Confusion Matrix:**
![png](cap/output_18_0.png)

**Classification report**

                   precision    recall  f1-score   support
    
            Apple       1.00      0.93      0.96       100
        Asparagus       0.99      0.95      0.97       100
           Banana       0.99      0.99      0.99       100
         Broccoli       0.97      0.99      0.98       100
          Cabbage       0.93      0.99      0.96       100
        Capsicums       0.96      0.96      0.96       100
           Cherry       0.97      0.98      0.98       100
    Cherry Tomato       0.94      0.99      0.97       100
           Chilli       0.99      0.94      0.96       100
           Grapes       0.93      0.99      0.96       100
            Mango       0.98      0.99      0.99       100
          Pakchoi       0.94      0.96      0.95       100
          Soursop       0.98      0.95      0.96       100
          Spinach       0.99      0.93      0.96       100
       Strawberry       0.96      0.98      0.97       100
    
      avg / total       0.97      0.97      0.97      1500


Not bad at all! Joe scored a astonishing 97% in test accuracy! I have taken a step further to evaluate the top 3 error rate. Simply put, if the correct label is predicted within the top 3 probabilities, it will be considered as correct. This further improves our accuracy to 99.9% wow!

## 5. Transfer learning

We will next attempt to build Joe via transfer learning to see if a better f1-score can be obtain. Transfer learning is the process of utilising knowledge from a particular task/domain to model for another task/domain, in our case through the use of pre-trained model.

Luckily for us, Keras has a [list](https://keras.io/applications/) of pre-trained models for our implementation. In this project, we will utilise Google's Inception_V3 model and retrain the last layer of the model to classify our 15 groceries. 

### 5.1 Defining the architecture
Retraining the model is as simple as loading up the model in your terminal, and specifying the images that you wish to classify.

```python
img_width, img_height = 299, 299 #fixed size for InceptionV3
batch_size = 32

# create the base pre-trained model
inception_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = inception_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and an output layer with 15 classes
predictions = Dense(15, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=inception_model.input, outputs=predictions)

# training only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in inception_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

```

Likewise, we call on the ImageDataGenerator class from Keras before training the model using the fit generator method.

### 5.2 Model Evaluation
**Confusion Matrix:**
![png](cap/output_15_0.png)

**Classification report**

                   precision    recall  f1-score   support
    
            Apple       0.87      0.83      0.85       100
        Asparagus       1.00      0.76      0.86       100
           Banana       0.94      0.96      0.95       100
         Broccoli       0.96      0.99      0.98       100
          Cabbage       1.00      0.93      0.96       100
        Capsicums       0.90      0.95      0.92       100
           Cherry       0.88      0.80      0.84       100
    Cherry Tomato       0.74      0.97      0.84       100
           Chilli       0.94      0.75      0.83       100
           Grapes       0.90      0.89      0.89       100
            Mango       0.97      0.90      0.93       100
          Pakchoi       0.89      0.97      0.93       100
          Soursop       0.96      1.00      0.98       100
          Spinach       0.85      0.95      0.90       100
       Strawberry       0.94      1.00      0.97       100
    
      avg / total       0.92      0.91      0.91      1500

Sadly, this model only gave us a f1-score of 91%.


## 6. Key Insights
1. **Keras Model**
    - Reviewing the wrongly classified images of the model, some are due to abnormalities like being in a package, some due to distortion of the image
    - For abnormalities, further training with images of such abnormalities will allow us to classify them correctly going forward  
    - In regards to distortion, all images should be cropped first to a square dimension before passing through a CNN
    
2. **Transfer learning**   
    - Being a CNN with much more layers and complexity than our Keras model, one may be puzzled why a lower score was achieved
    - Upon investigation, although Inception V3 was trained on millions of images, only 5 of our classes were represented.  
    - One way to overcome this will be to train more layers instead of only the last in the model to ensure representation of the other classes

3. **Business use case**
    - Both the Transfer Learning model and Keras model have their pros and cons. Some of their advantages are as follows: 

        * **Keras Model**
            - Full control of hyper-parameters
            - Full control of trainable features, depth and complexity of the model  

        * **Transfer learning**
            - Easy to apply
            - Models from different domains may be applicable to business use cases from differing domains  
    
   From a business perspective, it is important to consider whether there is sufficient time to develop the model, or whether accuracy is preferred over speed and utility.
    
4. **Other models**
    Support Vector Machines, Fuzzy measures and Genetic Algorithms exist to classify images. But due to the time constraints of the capstone project, we are only able to explore deep learning, which is by far, out performing the other algorithms according to various research papers. It is still important to give the other models a try for a more in-depth comparison.

## 7. Final words and future work
That's all for now. I do hope you had enjoy reading this and manage to have some takeaways.

Some future work on this that I am currently exploring:
1. Explore object detection techniques where multiple items can be classified in a single image
2. Build an application for Joe (WIP)
