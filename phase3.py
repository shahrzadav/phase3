# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split

import codecs
import time

start = time.time()


file1=codecs.open("quran-simple.txt",'r',encoding='utf-8')
for line in file1:
    vectorizer = TfidfVectorizer(lowercase=False, max_df=0.8)
    fs = vectorizer.fit_transform(line)


    fs_train, fs_test = train_test_split(
           fs, test_size=0.4, random_state=0
    )

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(fs_train)

    predict = kmeans.predict(fs_test)
print predict

end = time.time()
print end - start
