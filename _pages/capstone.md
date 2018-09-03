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
```python
Helpful Votes / Total Votes
```
However we start to realize that there is a fundumental issue with this.
```python
1 Helpful Vote  /  1 Total Vote  =  100%
99 Helpful Votes  /  100 Total Votes  =  99%
```
The problem here is that a review with a single helpful vote (ratio = 1.0) will beat a comment with 99 helpful votes and 1 unhelpful vote (ratio = 0.999), but clearly the latter comment is more likely to be “more helpful.”
We needed to take in account of the number of votes.

**How do we determine how helpful a review is?**

Using Bayesian statistics - [read here](http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter4_TheGreatestTheoremNeverTold/Ch4_LawOfLargeNumbers_PyMC2.ipynb)

We can take into account the number of votes:
```python
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
```python
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

> **From the Has Votes set, we sampled 50,000 rows as our working dataset.**



## 4. Before Model Building
Before we go any further, we need to define the approach and metrics that we will use to evaluate our model(s).
**We are tackling this as a Classification problem. Where we will classify reviews as Is_Helpful = 1 or 0.**

1. We start off with Data Preparation - cleaning the text data, fixing of HTML embeddings, emoticons, non-UTF-8 characters.
2. Next, we proceed with Model Selection.
3. Train and Test the model.
4. Analyse the results - with respect to the business outcome.

<img src="/assets/images/capstone_approachmetrics.jpg"> 

The 2 main metrics that we are looking at would be:
1. Precision: *When it's actually 'Helpful', how often does it predict 'Helpful'.*
2. Recall: *When it predicts 'Helpful', how often is it correct?*

### 4.1 Baseline model
As this is a binary classification model, we start off with a Logistic Regression model as the baseline model.
A pipeline was also built to analyse text features by groups of words and characters together with their frequencies.
By iteration, there were 3 main features that contributed to the best logistic regression score.

<img src="/assets/images/capstone_baseline.jpg"> 

Now that we have a baseline model to start from, we began trying out different models.
### 4.2 Model Selection
<img src="/assets/images/capstone_compare.jpg"> 

I did a comparison with models that work differently. From [Naive Bayes models](http://scikit-learn.org/stable/modules/naive_bayes.html), [Decision Tree models](http://scikit-learn.org/stable/modules/tree.html), [Boosted Tree models](https://xgboost.readthedocs.io/en/latest/) and the [Support Vector Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)

I found that the LinearSVC model could classify the results the best, in accordance to my needs.

<img src="/assets/images/capstone_linearSVC.jpg"> 
**Linear Support Vector Classification**
The core concept behind the success and the powerful nature of Support Vector Machines is that of margin maximisation. More specifically, SVMs attempt to build a decision boundary that accurately separates the training samples of different classes by maximising the margin between them. The margin is the (perpendicular) distance from the decision surface to the closest data points of each class.


### 4.3. Alternate Model
Given the nature of a this starting off from a NLP (Natural Language Processing) root, I decided to try an alternate approach. Instead of decision boundary approach (LinearSVC), i tried a Recurrent neural network, with a Long Short Term Memory. By this approach, the neural network would be learning via the sequence of words and characters.

This is a simplified way of describing a Recurrent neural network, with a Long Short Term Memory.
- A simple neural network does'nt have any memory. It process information straight through.
- A Recurrent neural network, has *some* memory. Letting it remember just a little. Remembering and learning (adding information) at the same time.
- A Recurrent neural network with a Long Short Term Memory has the ability to remember more. Therefore learning and remembering and greater capacity.

<img src="/assets/images/capstone_rnnLSTM.jpg"> 


### 4.4 Model Conclusion & Evaluation

<img src="/assets/images/capstone_conclude.jpg"> 


## 5. Risk and Limitations

We will next attempt to build Joe via transfer learning to see if a better f1-score can be obtain. Transfer learning is the process of utilising knowledge from a particular task/domain to model for another task/domain, in our case through the use of pre-trained model.

Luckily for us, Keras has a [list](https://keras.io/applications/) of pre-trained models for our implementation. In this project, we will utilise Google's Inception_V3 model and retrain the last layer of the model to classify our 15 groceries. 


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
