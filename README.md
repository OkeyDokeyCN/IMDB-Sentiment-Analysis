# IMDB-Sentiment-Analysis

In this project, I performed Sentiment Analysis on 50k labeled IMDB movie reviews, build a classifier to classify the overall review as positive or negative.

## Introduction
Movie review is a very intuitive way to evaluate a movie performance. The numerical rating is a part of the overall quality of a movie, but the movie reviews provide a deeper insight of the audience's response. People can express their emotions through the text, it is usually more complex and detailed than numerical ratings. 

Sentiment Analysis involve with Natural Language Processing, it is used to find the sentiment of the person with respect to a given source of content.


## Data
The dataset consists of 50k IMDB movie reviews: 25k labeled training instances, 25k labeled test intances. Each revie consists of several sentences of text, the labels are two category: Positive and Negative.

https://www.kaggle.com/utathya/imdb-review-dataset

## Results
* Logistic Regression
After using GridSearchCV, obtained accuracy 87.34%, with C = 0.1

* Random Forest Classifier
After using GridSearchCV, obtained accuracy 84.88%, with max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 300

* SVM
After using GridSearchCV, obtained accuracy 87.48%, C = 0.01, Kernel = Linear

* Naive Bayes
After using GridSearchCV, obtained accuracy 83.25%, MultinomialNB, alpha = 1

## Application
* This is useful in cases when the producer wants to measure its overall performance using reviews, also improve the quality of the movie by learning the unfavourable aspect of the movie.

* The outcome of this project can also be used to create a recommendation system, by analyzing and categorizing the viewers’ opinion according to their preferences and interests, the system can predict which movie should be recommended, which one should not.

* Another application of this project would be to find a group of viewers with similar movie tastes(likes or dislikes).


## Reference
Sentiment Analysis, Wikipedia, author: https://en.wikipedia.org/wiki/Sentiment_analysis
Bag of Words Meets Bags of Popcorn,author:Kaggle, https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words
Sentiment-Recognition-on-IMDB, author:karimamd
Sentiment Analysis of Movie Reviews using Machine Learning Techniques,2017,author:Palak Baid,Apoorva Gupta,Neelam Chaplot
Sentiment Analysis for Movie Reviews, 2015, author: Ankit Goyal,Amey Parulekar

