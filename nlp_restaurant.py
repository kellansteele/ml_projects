import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def nlpRestaurant():
    dataset = pd.read_csv('data/Restaurant_Reviews.tsv')

    corpus = []

    for i in range(0, 1000):                                    # treat each review separately
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])   # replace things that aren't letters by a space
        review = review.lower()                                   # make everything lowercase
        review = review.split()                                   # split into different words
        ps = PorterStemmer()
        allStopwords = stopwords.words('english')
        allStopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(allStopwords)] # stem all words that are not stop words, e.g. 'loved' -> 'loved'
        review = ' '.join(review)                                 # rejoin and add spaces to each word in review
        corpus.append(review)

    st.write(corpus)