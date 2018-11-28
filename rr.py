import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

dataset =  pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')
#nltk.download('stopwords')

corpus = []
for i in range(1000):
	review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review
				if not word in set(stopwords.words('english'))]
	review = ' '.join(review)
	corpus.append(review)

cv = CountVectorizer(1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model = RandomForestClassifier(n_estimators = 501, criterion = 'entropy')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)
