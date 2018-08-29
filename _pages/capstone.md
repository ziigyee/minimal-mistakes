---
layout: single
permalink: capstone/
title: &title "Capstone Project"
author_profile: true
toc: true
toc_label: ON THIS PAGE
---
<img src="/assets/images/capstone_banner.jpg">

## 1. Background

We've all done our shopping online. From electronics, to household items, and even pantry items. I would believe that it would be safe to say that many of us are well versed buying online from websites, both local and foreign.

Every time we go looking for something, there are seems to be a variety of what you are looking for. How does one decide which item to buy, since item A and B do the same thing, and are at the same price?

We no longer make our purchasing decisions in the blind. We rely on product reviews. Any shopper reads approximately 10 reviews before deciding if they would buy it. A review is a testimony. It's a proven track record of the product by previous buyer's recommendations or warnings.

<img src="/assets/images/capstone_review.jpg">

The value that product reviews bring on any e-commerce platform:
* For the **buyer**: Better decision making. Merchant/Product trust. Recommended.
* For the **merchant**: Helps sell more. Better credibility. Informative experience.
* For the **platform**: More business and traction. Platform becomes the 'Go-to' place for shoppers.
* Adds to the **customer experience**

Many products receive reviews both good and bad. However it is the 'Helpful' reviews that are the decision influencing factors.

### 1.1. Understanding Reviews

Reviews are often separated into 2 groups. Helpful or Unhelpful. Helpful reviews help buyers decide if they should purchase. Reviews also shared experiences from a previous buyer. Unhelpful reviews have little or no relevance to decision making, infact they often ignored.

Most e-commerce platforms allow the reviews to be voted on.

<img src="/assets/images/capstone_eg.jpg">

## 2. Business Case for Actionable Insights

What can we do if we could determine that a review would be:

<img src="/assets/images/capstone_helpful.jpg">

_**Helpful**_:
* It could be put in the front of buyers on a product page to help potential buyers make a decision.
* Potential buyers could still vote if that review was helpful to them or not.

<img src="/assets/images/capstone_unhelpful.jpg">

_**Unhelpful**_:
* The merchant or platform could follow up with the reviewer, seeking recourse or more information.
* Alternative trigger for customer satisfaction issues.

**Ultimately, we can provide a better customer experience on and off the platform.**


## 3. The Capstone
<img src="/assets/images/capstone_goal.jpg">

### 3.1 The Dataset

The dataset that was used for this project was from Amazon.com and can be [found here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)
I used the Amazon Product Reviews of ‘Tools’ Category.

<img src="/assets/images/capstone_dataset.jpg">

### 3.1 Columns of Interest
<img src="/assets/images/capstone_interest.jpg"> 

Before we proceed, we need to understand which columns we would be using as our main features. Those in green, are our main focus features. We need the text data from those columns.
For the yellow, we can use them as support features, that may or may not work for our model in predicting if a review is helpful or not.

### 3.2 The Ground Truth
In the dataset, for each review, there are Total Votes and Helpful Votes.

We accept the ground truth of a review being helpful by considering the amount of helpful or unhelpful votes the review has.
An initial approach
```
Helpful Votes / Total Votes
```
However we start to realize that there is a fundumental issue with this.
```
1 Helpful Vote  /  1 Total Vote  =  100%
99 Helpful Votes  /  100 Total Votes  =  99%
```
The problem here is that a review with a single helpful vote (ratio = 1.0) will beat a comment with 99 helpful votes and 1 unhelpful vote (ratio = 0.999), but clearly the latter comment is more likely to be “more helpful.”
We needed to take in account of the number of votes.

**How do we determine how helpful a review is?**

Using Bayesian statistics - [read here](http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter4_TheGreatestTheoremNeverTold/Ch4_LawOfLargeNumbers_PyMC2.ipynb)

We can take into account the number of votes:
```
1 Helpful Vote  /  1 Total Vote  =  0.277758
49 Helpful Votes  /  50 Total Votes  =  0.917
71 Helpful Votes  /  100 Total Votes  =  0.632
999 Helpful Votes  /  1000 Total Votes  =  0.996
```
Now this represents how helpful a review is, in a much better form.

**Alas, we need to decide what will our ground truth be**
This may seem like going one big round, however the bayesian calculation would help further on in the actionable business side to rank and display the reviews according to a score (In this case, we shall call it the Helpfulness Score)

For the project; I have set a hard cut-off point for the ground truth.
```
1 Helpful Vote  /  1 Total Vote  =  0.277758 Helpfulness Score
```
The cut-off for a review to be deemed 'Helpful' is that more than 1 person must have voted for it to be 'Helpful'
```
IF Helpfulness Score > 0.277758:
    Is_Helpful = 1
ELSE
    Is_Helpful = 0
```

### 3.3 Working Dataset
The original dataset was 1.7M rows. In order for us to build the model, we need to manage our data, for training, testing and predicting on. 

<img src="/assets/images/capstone_workingdata.jpg"> 

The dataset was split into 2 sets.
1. Has Votes - Which we will use for training and building our model. (consist of 42% of the main dataset)
2. No Votes - Which is our 'hold-out' data to predict on. (consist of 58% of the main dataset)

The logic is that we use the Has Votes, because we have knowledge about it. We predict on the No Votes because these reviews have not been determined if they are Helpful or Not, since they may not even be displayed to the potential buyers.

**From the Has Votes set, we sampled 50,000 rows as our working dataset.**



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
