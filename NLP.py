import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
def cleanText(Text):
    Text = re.sub(r'http\S+', '',Text)
    Text = re.sub(r'RT|cc', '',Text)
    Text = re.sub(r'@\S+','',Text)
    Text = re.sub(r'#\S+','',Text)
    Text = re.sub(r'[^A-Za-z0-9]', ' ',Text)

    return Text.lower()
df = pd.read_csv('Dataset.csv', encoding='latin1')
df['Text'] = df['Text'].apply(cleanText)
# print(df.head())
# print(df.shape)
# print(df['Category'].value_counts())
# print(df['Category'][0])
# print(df['Text'][0])


# print(Text("hii http://hwlloworld.com and nidisg@gmail.com or #akas This Or yes! you *> "))

from sklearn.feature_extraction.text import TfidfVectorizer

Tfid = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)

X= Tfid.fit_transform(df['Text'])
y= df['Category']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#---------------------------
# from sklearn.naive_bayes import MultinomialNB

# model= MultinomialNB()

#------------------------------

from sklearn.svm import LinearSVC
model= LinearSVC()

#--------------------------------
# from sklearn.linear_model import LogisticRegression

# model= LogisticRegression(max_iter=1000)

#-------------------------------------
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred= model.predict(X_test)
print("Accuracy score:", accuracy_score(y_test, y_pred))

from sklearn.metrics import precision_score
print("Precision score:", accuracy_score(y_test, y_pred))

import pickle

# save trained model
pickle.dump(model, open("resume_model.pkl", "wb"))

# save TF-IDF vectorizer
pickle.dump(Tfid, open("vectorizer.pkl", "wb"))