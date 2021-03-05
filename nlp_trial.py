import io
import numpy as np
import pandas as pd
import PIL
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import urllib.request

from const import SENT_VALUES
import matplotlib.pyplot as plt
import re
import nltk
import streamlit as st
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


@st.cache
def load_data():
    df = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    # Data cleaning
    corpus = []
    for i in range(0, 1000):                                    # treat each review separately
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])   # replace things that aren't letters by a space
        review = review.lower()                                   # make everything lowercase
        review = review.split()                                   # split into different words
        ps = PorterStemmer()
        allStopwords = stopwords.words('english')
        allStopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(allStopwords)] # stem all words that are not stop words, e.g. 'loved' -> 'loved'
        review = ' '.join(review)                                 # rejoin and add spaces to each word in review
        corpus.append(review)
    return df, corpus

df = load_data()[0]
corpus = load_data()[1]

@st.cache
def split_data():
    # Creating bag of words model
    cv = CountVectorizer(max_features = 1500)     # max_features allows you to ignore words that rarely appear
    X = cv.fit_transform(corpus).toarray()        # create matrix of features
    y = df.iloc[:, -1].values

    # Split dataset into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    return cv, X_train, X_test, y_train, y_test

cv = split_data()[0]
X_train = split_data()[1]
X_test = split_data()[2]
y_train = split_data()[3]
y_test = split_data()[4]

# Naive Bayes
def nlpNaiveBayes():
    # Train the Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predict the test set results
    y_predNB = classifier.predict(X_test)

    accuracyNB = accuracy_score(y_test, y_predNB)
    precisionNB = precision_score(y_test, y_predNB)
    recallNB = recall_score(y_test, y_predNB)
    cmNB = confusion_matrix(y_test, y_predNB)

    return(accuracyNB, precisionNB, recallNB, cmNB, classifier)#, y_pred_1, y_pred_2)

# logistic regression
def nlpLogisticReg():
    # Train the logistic regression model on the training set
    logReg = LogisticRegression(random_state = 0)
    logReg.fit(X_train, y_train)

    # Predict the test set results
    y_predLR = logReg.predict(X_test)

    accuracyLR = accuracy_score(y_test, y_predLR)
    precisionLR = precision_score(y_test, y_predLR)
    recallLR = recall_score(y_test, y_predLR)
    cmLR = confusion_matrix(y_test, y_predLR)

    return(accuracyLR, precisionLR, recallLR, cmLR)


# k-nearest neighbours
def nlpKNearestNeighb():
    # Train the k-nn model on the training set
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, y_train)

    # Predict the test set results
    y_predKNN = knn.predict(X_test)

    accuracyKNN = accuracy_score(y_test, y_predKNN)
    precisionKNN = precision_score(y_test, y_predKNN)
    recallKNN = recall_score(y_test, y_predKNN)
    cmKNN = confusion_matrix(y_test, y_predKNN)

    return(accuracyKNN, precisionKNN, recallKNN, cmKNN)

# SVM
def nlpSVM():
    # Train the SVM model on the training set
    svm = SVC(kernel='linear', random_state=0)
    svm.fit(X_train, y_train)

    # Predict the test set results
    y_predSVM = svm.predict(X_test)

    accuracySVM = accuracy_score(y_test, y_predSVM)
    precisionSVM = precision_score(y_test, y_predSVM)
    recallSVM = recall_score(y_test, y_predSVM)
    cmSVM = confusion_matrix(y_test, y_predSVM)

    return(accuracySVM, precisionSVM, recallSVM, cmSVM)

# kernel SVM
def nlpKernelSVM():
    # Train the kernel SVM model on the training set
    kernelsvm = SVC(kernel='rbf', random_state=0)
    kernelsvm.fit(X_train, y_train)

    # Predict the test set results
    y_predKernelSVM = kernelsvm.predict(X_test)

    accuracyKernelSVM = accuracy_score(y_test, y_predKernelSVM)
    precisionKernelSVM = precision_score(y_test, y_predKernelSVM)
    recallKernelSVM = recall_score(y_test, y_predKernelSVM)
    cmKernelSVM = confusion_matrix(y_test, y_predKernelSVM)

    return(accuracyKernelSVM, precisionKernelSVM, recallKernelSVM, cmKernelSVM)

# decision tree
def nlpDecisionTree():
    # Train the decision tree classification model on the training set
    decisionTree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    decisionTree.fit(X_train, y_train)

    # Predict the test set results
    y_predDecisionTree = decisionTree.predict(X_test)

    accuracyDecisionTree = accuracy_score(y_test, y_predDecisionTree)
    precisionDecisionTree = precision_score(y_test, y_predDecisionTree)
    recallDecisionTree = recall_score(y_test, y_predDecisionTree)
    cmDecisionTree = confusion_matrix(y_test, y_predDecisionTree)

    return(accuracyDecisionTree, precisionDecisionTree, recallDecisionTree, cmDecisionTree)

# random forest classification
def nlpRandomForest():
    # Train the random forest classification model on the training set
    rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rfc.fit(X_train, y_train)

    # Predict the test set results
    y_predRFC = rfc.predict(X_test)

    accuracyRFC = accuracy_score(y_test, y_predRFC)
    precisionRFC = precision_score(y_test, y_predRFC)
    recallRFC = recall_score(y_test, y_predRFC)
    cmRFC = confusion_matrix(y_test, y_predRFC)

    return(accuracyRFC, precisionRFC, recallRFC, cmRFC)

def userSent(userInput, classifier):
    userInput = re.sub('[^a-zA-Z]', ' ', userInput)
    userInput = userInput.lower()
    userInput = userInput.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    userInput = [ps.stem(word) for word in userInput if not word in set(all_stopwords)]
    userInput = ' '.join(userInput)
    corpus_1 = [userInput]
    X_test_1 = cv.transform(corpus_1).toarray()
    y_pred_1 = classifier.predict(X_test_1)
    y_pred_1 = SENT_VALUES[int(y_pred_1[0].item())]

    return y_pred_1