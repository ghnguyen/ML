# Use seaborn to plot Linear and Logistic Rgeression
# https://seaborn.pydata.org/tutorial/regression.html

#https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data

# -------------------------------
# plot Linear Regression
# https://stackoverflow.com/questions/43152529/python-matplotlib-adding-regression-line-to-a-plot-given-its-intercept-and-slop
# ---------------

# ----------------
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
# ----------------


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
print(type(iris))
X = iris.data[:, [2,3]]
y = iris.target
#print(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
#print("Mis classified %d" % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
#print("ACCURACY %.2f" % accuracy_score(y_test, y_pred))

#print("This is my first PyCharm test")

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X,y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    test_set_color = [cl for cl in colors if cl not in cmap.colors][0]

    # plot the decision surface
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


    for idx, cl in enumerate(np.unique(y)):
        xx = X[y==cl,0]
        yy = X[y==cl,1]
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8,c=cmap(idx),
                    marker="x", label=cl)

    if test_idx:
        #print("xtf")
        X_test, y_test = X[test_idx,:], y[test_idx]
        xx = 1
        for idx, cl in enumerate(np.unique(y)):
            my_y = y_test==cl
            plt.scatter(X_test[y_test==cl,0], X_test[y_test==cl,1],
                        c=cmap(idx), alpha=1.0, linewidths=1,
                        marker='^', s=100, label='test '+str(cl))


X_comined_std = np.vstack((X_train_std, X_test_std))

#
#y_test = [3 for yt in y_test]
#
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000, random_state=0)
lr.fit(X_train_std, y_train)


plot_decision_regions(X=X_comined_std, y=y_combined, classifier=lr,test_idx=range(105,150))
plt.xlabel("petal length standardized")
plt.ylabel("petal width standardized")
plt.legend(loc="upper left")
#plt.show()

# what is a vector norm ?
v = [1.6, 2,1.29,1.29,1.29,2,1.29]
#print(np.linalg.norm(v))

# IMDB



"""
import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(50000)
labels = {"pos":1, "neg":0}
df = pd.DataFrame()
for s in ("test", "train"):
    for l in ("pos", "neg"):
        path = "C:/Users/hahien/PycharmProjects/aclImdb/{}/{}".format(s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf8") as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
df.columns=["review", "sentiment"]
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv("C:/Users/hahien/PycharmProjects/aclImdb/movie_data.csv", index=False)
"""
import pandas as pd
df = pd.read_csv("C:/Users/hahien/PycharmProjects/aclImdb/movie_data.csv", encoding="ISO-8859-1")
#print(df.head())
#print("OK")
#print(df.loc[0, 'review'][-50:])

# Cleaning
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

#print(preprocessor(df.loc[0, 'review'][-50:]))
#print(preprocessor("</a> This :) is :( a test :-) !"))
df["review"] = df["review"].apply(preprocessor)
#print(df.head())


#
def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter():
    return [porter(w) for w in text.split()]



X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidif = TfidfVectorizer(strip_accents=None, lowercase=False,preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [None],
               'vect__tokenizer': [tokenizer],
               'clf__penalty':["l2"],
               'clf__C': [1]
               }
              ]

""" 
{'vect__ngram_range': [(1,1)],
'vect__stop_words': [None],
'vect__tokenizer': [tokenizer, tokenizer_porter],
'vect__use_idf': [False],
'vect__norm': [None],
'clf__penalty':["l1", "l2"],
'clf__C': [1,10,100]
}
"""
lr_tfidf = Pipeline([("vect", tfidif), ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=2, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print("Best parameters set %s " % gs_lr_tfidf.best_params_)

print("CV accuracy: %3f" % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test accuracy %3f' % clf.score(X_test, y_test))
