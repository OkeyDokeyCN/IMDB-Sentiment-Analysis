

#Import Libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder
import operator
from nltk.stem.porter import PorterStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, ensemble, metrics
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB

#Read Files
#"header=0" indicates the first line of the file contains the column names,"delimiter=\t"
#indicates that the fields are separated by tabs,"quoting=3" tells Python to ignore doubled quotes.       
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
train.shape
train.columns.values
print(train['review'][0])

test = pd.read_csv("testData.tsv", header=0, \
                    delimiter="\t", quoting=3)

test['rate'] = test['id'].apply(lambda x: int(x.replace('"', "").split('_')[1]))

test['sentiment'] = test['rate'].apply(lambda x : 1 if x >= 7 else 0)

print((test.sentiment == 1).sum())
print((test.sentiment == 0).sum())



#Preprocessing
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return meaningful_words 


#Clean the training set
num_reviews = train["review"].size
clean_train_reviews = []
print("Cleaning and parsing the training set movie reviews...\n")
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_reviews ))                                                                 
    clean_train_reviews.append( review_to_words( train["review"][i] ))
    
#Clean the test set

num_reviews = test["review"].size
clean_test_reviews = []
print("Cleaning and parsing the test set movie reviews...\n")
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_reviews ))                                                                 
    clean_test_reviews.append( review_to_words( test["review"][i] ))



#Normalization
#Stemming, replace every word with its root word
def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review]) for review in corpus]

#stemme the train data
stemmed_train_reviews = get_stemmed_text(clean_train_reviews)    

#stemme the test data
stemmed_test_reviews = get_stemmed_text(clean_test_reviews)    



    
#Feature Extraction
#After we cleaned the data, we need to convert the text into numerical form.
#CountVectorizer converts a collection of text documents to a sparse matrix of token counts.
vectorizer = CountVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             ngram_range=(1,1),     #ngram_range can be(1,1),(1,2),(1,3)
                             max_features = 5000)   #                1 word,2 words, 3 words


#Get the training features
train_data_features = vectorizer.fit_transform(stemmed_train_reviews)

train_data_features = train_data_features.toarray()

print(train_data_features.shape)

#Get the testing features
test_data_features = vectorizer.transform(stemmed_test_reviews)

test_data_features = test_data_features.toarray()
    
print(test_data_features.shape)

#Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)

#print the counts of each words in the vocabulary
dist = np.sum(train_data_features, axis = 0)
for tag, count in zip(vocab, dist):
    print(count, tag)




def print_feature(df):
    
    # create an instance for tree feature selection
    tree_clf = ExtraTreesClassifier()

    # first create arrays holding input and output data

    # Vectorizing Train set
    cv = CountVectorizer(analyzer='word')
    x_train = cv.fit_transform(df['review'])

    # Creating an object for Label Encoder and fitting on target strings
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    # fit the model
    tree_clf.fit(x_train, y)

    # Preparing variables
    importances = tree_clf.feature_importances_
    feature_names = cv.get_feature_names()
    feature_imp_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(20):
        print("feature %d : %s (%f)" % (indices[f], sorted_features[f][0], sorted_features[f][1]))

    # Plot the feature importances of the forest
    plt.figure(figsize = (20,20))
    plt.title("Feature importances")
    plt.bar(range(100), importances[indices[:100]],
           color="r", align="center")
    plt.xticks(range(100), sorted_features[:100], rotation=90)
    plt.xlim([-1, 100])
    plt.show()

    return()



d = {'review': stemmed_train_reviews, 'sentiment': train['sentiment']}
imdb_train = pd.DataFrame(data=d)

#Print the top 20 most frequent features 
print_feature(imdb_train)
'''
Feature ranking:
feature 3010 : bad (0.006765)
feature 49639 : worst (0.006680)
feature 48570 : wast (0.006117)
feature 2802 : aw (0.004718)
feature 18167 : great (0.004356)
feature 14343 : excel (0.004291)
feature 5184 : bore (0.004163)
feature 49632 : wors (0.002894)
feature 26095 : love (0.002886)
feature 44170 : terribl (0.002812)
feature 42640 : stupid (0.002609)
feature 30992 : noth (0.002599)
feature 13739 : enjoy (0.002572)
feature 34188 : poor (0.002473)
feature 20490 : horribl (0.002228)
feature 43081 : suppos (0.002184)
feature 36991 : ridicul (0.002164)
feature 4143 : best (0.002159)
feature 28670 : minut (0.002070)
feature 34191 : poorli (0.001924)
'''

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=300,
        max_font_size=40, 
        scale=3,
        random_state=0 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(stemmed_train_reviews)


#Model
#Using different classification models to compare results (Logistic Regression,
#RandomForest Classifie, SVM, Naive Bayes)

roc_auc_scorer = metrics.make_scorer(metrics.roc_auc_score)

#Select 500 features to tune the parameters, instead of using the full data, this could save a lot of time
#After comparing the best acuracy scores, choose the best parameter to train the model
X_train = train_data_features[:500]

Y_train = train['sentiment'][:500]



#Logistic Regression

params = { 'C':[ 0.001, 0.01, 0.1, 1, 10]}

lr = LogisticRegression(random_state = 0)

#Implement GridSearch to optimize the best parameter for the model
clf = GridSearchCV(lr, param_grid = params, scoring = roc_auc_scorer, cv = 5)

clf.fit(X_train, Y_train)

#print the best score with the optimal parameter
print('Best score: {}'.format(clf.best_score_))
#Best score: 0.7901546695317313

print('Best parameters: {}'.format(clf.best_params_))
#Best parameters: {'C': 0.1}


#create logistic regression with the best C value
lr_best = LogisticRegression(C = 0.1, random_state = 0)

lr_best.fit(train_data_features,train['sentiment'])

pred = lr_best.predict(test_data_features)

#accuracy score
accuracy = accuracy_score(pred,test['sentiment'])
print(accuracy) #0.871

#Below are the accuracy scores from different ngram features
#The highest accuracy scores is 0.8734 for logistic regression, with ngram = (1,2) and C = 0.1
#ngram = (1,1), C = 0.1, accuracy = 0.871
#ngram = (1,2), C = 0.1, accuracy = 0.87344
#ngram = (1,3), C = 0.1, accuracy = 0.87312
#ngram = (1,4), C = 0.1, accuracy = 0.87212

# plot the best and worst top 25 features.
import mglearn
mglearn.tools.visualize_coefficients(clf.best_estimator_.coef_, vocab, n_top_features=25)
plt.show()



#Random Forest Classifier
forest = RandomForestClassifier(random_state = 1)

#Initialize the parameters to optimize
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)

bestF = gridF.fit(X_train, Y_train)

print('Best score: {}'.format(bestF.best_score_))
#Best score: 0.7780342928600631

print('Best parameters: {}'.format(bestF.best_params_))
#Best parameters: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}

#Create randomforest classifier with the best parameters
rf_best = RandomForestClassifier(max_depth = 25,
                                 min_samples_leaf = 1, 
                                 min_samples_split = 2,
                                 n_estimators = 500,
                                 random_state = 1)

rf_best.fit(train_data_features,train['sentiment'])

rf_pred = rf_best.predict(test_data_features)

rf_accuracy = accuracy_score(rf_pred, test['sentiment'])

print(rf_accuracy)

#ngram=(1,1), hyperF=(max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 300)
# 0.84888
#ngram=(1,2), hyperF=(max_depth = 25, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 500)
#0.84732




#SVM
#Parameters for SVM model
params = {'kernel':['linear', 'rbf'], 'C':[0.1, 0.5, 1, 5, 10]}

svc = SVC(probability = True, random_state = 0)

gridSVC = GridSearchCV(svc, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

bestSVC = gridSVC.fit(X_train, Y_train)

print('Best score: {}'.format(bestSVC.best_score_))
#Best score: 0.7613371098919072

print('Best parameters: {}'.format(bestSVC.best_params_))    
#Best parameters: {'C': 5, 'kernel': 'rbf'}

#Crate the model with the best parameters
svc_best = SVC(C = bestSVC.best_params_['C'], 
               kernel = bestSVC.best_params_['kernel'], 
               probability = True, 
               random_state = 0)

svc_best.fit(train_data_features, train['sentiment'])

svc_pred = svc_best.predict(test_data_features)

svc_accuracy = accuracy_score(svc_pred, test['sentiment'])

print(svc_accuracy)
#ngram = (1,1) C = 0.01, Kernel = linear
#0.87488
#second time C = 5, Kernel = rbf
#0.86788




#Naive Bayes
params = {"var_smoothing" : [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1]}

nb = GaussianNB()

gridNB = GridSearchCV(nb, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

gridNB.fit(X_train, Y_train)

print('Best score: {}'.format(gridNB.best_score_))
print('Best parameters: {}'.format(gridNB.best_params_))

NB_best = GaussianNB(var_smoothing = 0.0001)

NB_best.fit(train_data_features, train['sentiment'])

NB_pred = NB_best.predict(test_data_features)

NB_accuracy = accuracy_score(NB_pred, test['sentiment'])

print(NB_accuracy)
#var_smoothing = 0.0001
#0.6474


#MultinomialNB
params = {"alpha" : [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1]}

nb = MultinomialNB()

gridNB = GridSearchCV(nb, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

gridNB.fit(X_train, Y_train)

print('Best score: {}'.format(gridNB.best_score_))
print('Best parameters: {}'.format(gridNB.best_params_))

NB_best = MultinomialNB(alpha = 1)

NB_best.fit(train_data_features, train['sentiment'])

NB_pred = NB_best.predict(test_data_features)

NB_accuracy = accuracy_score(NB_pred, test['sentiment'])

print(NB_accuracy)
#alpha = 1
#0.83252






