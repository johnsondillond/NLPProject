'''
• KNN Classifier - NearestNeighbors
• Random Forest - RandomForestClassifier
• Naive Bayes - MultinomialNB
• Support Vector Classifier - LinearSVR
• Decision Tree Classifier - DecisionTreeClassifier
• Hist Gradient Boosting Classifier - HistGradientBoostingClassifier
• Logistic Regression - LogisticRegression
• MLP Classifier - MLPClassifier
• Neural Nets
'''
'''
F1 score
recall
precision
AUC-ROC (one-vs-all)
'''
# import torch
# import torch.nn as nn
# import torch.optim as optim
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import VotingClassifier

import polars as pl
import numpy as np
import pickle

def result_generator(model_type, test, pred, probs):
  p = precision_score(test, pred, average='weighted')
  r = recall_score(test, pred, average='weighted')
  f1 = f1_score(test, pred, average='weighted')
  ar = roc_auc_score(test, probs, average='weighted', multi_class='ovr')

  print("PRECISION: ", p)
  print("RECALL: ", r)
  print("F1: ", f1)
  print("AUC-ROC: ", ar)

  results.append(f'{model_type.upper()} - Precision: {p} Recall: {r} F1: {f1} AUC_ROC: {ar}')

# Train/Test split 
train_df = pl.read_csv('data/train.csv')
test_df = pl.read_csv('data/valid.csv')

X_train = train_df['Body'] # Change for other features
X_test = test_df['Body'] # Change for other features
y_train = train_df['Y']
y_test = test_df['Y']

#*### Preprocessing ####
# Tokenize
print('Tokenizing')
token_X_train = [word_tokenize(line)[:100] for line in X_train]
token_X_test = [word_tokenize(line)[:100] for line in X_test]

# Filter out non-English
print('Filtering out non-English')
token_train_cleaned = []
token_test_cleaned = []
for t in token_X_train:
  temp = []
  for i in t:
    if i.isalpha():
      temp.append(i)
  token_train_cleaned.append(temp)

for t in token_X_test:
  temp = []
  for i in t:
    if i.isalpha():
      temp.append(i)
  token_test_cleaned.append(temp)

# Lemmatize
print('Lematizing')
lemmatizer = WordNetLemmatizer()
lemma_X_train = [[lemmatizer.lemmatize(word) for word in line] for line in token_train_cleaned]
lemma_X_test = [[lemmatizer.lemmatize(word) for word in line] for line in token_test_cleaned]

# Remove stop words
print('Removing stop words')
clear_X_train = [[word for word in line if word not in stopwords.words('english')] for line in lemma_X_train]
clear_X_test = [[word for word in line if word not in stopwords.words('english')] for line in lemma_X_test]

#*### TF-IDF ####
print('TF-IDF')
# Initiate a tfidf vectorizer object
tfidf_vectorizer = TfidfVectorizer(preprocessor=' '.join)

# Fitting the training dataset and transform it
X_train_vec = tfidf_vectorizer.fit_transform(clear_X_train)

# Apply the vectorizer to the test dataset
X_test_vec = tfidf_vectorizer.transform(clear_X_test)

#*### Encoding Labels ####
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.fit_transform(y_test)
results = []

#?### Random Forest - RandomForestClassifier ####
print('\nRANDOM FOREST')
print('Training')
n_estim_list = [50, 100, 200]

for estim in n_estim_list:
  random_forest = RandomForestClassifier(n_estimators=estim, random_state=42)
  random_forest.fit(X_train_vec, y_train_encoded)

  y_pred = random_forest.predict(X_test_vec)
  y_pred_probs = random_forest.predict_proba(X_test_vec)

  result_generator('random forest (%s)' %estim, y_test_encoded, y_pred, y_pred_probs)

  with open('random_forest_test_%s.pkl' %estim, 'wb') as file:
      pickle.dump(random_forest, file)


#?### Support Vector Classifier - LinearSVC
print('\nSVC')
print('Training')
clf = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False, max_iter=300), cv=5)
clf.fit(X_train_vec, y_train_encoded)

y_pred = clf.predict(X_test_vec)
y_pred_probs = clf.predict_proba(X_test_vec)

result_generator('linear svc', y_test_encoded, y_pred, y_pred_probs)

with open('svc_test.pkl', 'wb') as file:
    pickle.dump(clf, file)

#?### Hist Gradient Boosting Classifier - HistGradientBoostingClassifier
print('\nHIST GRADIENT BOOSTING CLASSIFIER')
print('compressing for denser vectors')
learn_rate_list = [0.05, 0.1, 0.2]
max_depth_list = [None, 5, 10]
for lr in learn_rate_list:
  for md in max_depth_list:

    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_vec)
    X_test_reduced = svd.transform(X_test_vec)

    print('Training')
    hist = HistGradientBoostingClassifier(max_bins=255, max_iter=300, random_state=42, max_depth=md, learning_rate=lr)
    hist.fit(X_train_reduced, y_train_encoded)

    y_pred = hist.predict(X_test_reduced)
    y_pred_probs = hist.predict_proba(X_test_reduced)

    result_generator('hist gradient boosting (%s, %s)' %(lr, md), y_test_encoded, y_pred, y_pred_probs)

    with open('hist_gradient_test_%s_%s.pkl' %(lr, md), 'wb') as file:
        pickle.dump(hist, file)

#?### Logistic Regression - LogisticRegression
print('\nLOGISTIC REGRESSION')
print('Training')
# Train a model of logistic regression
clf = LogisticRegression().fit(X_train_vec, y_train_encoded)

y_pred = clf.predict(X_test_vec)
y_pred_probs = clf.predict_proba(X_test_vec)

result_generator('logistic regression', y_test_encoded, y_pred, y_pred_probs)

with open('logistic_regression_test.pkl', 'wb') as file:
  pickle.dump(clf, file)

'''
#?### MLP Classifier - MLPClassifier
print('\nMLP')
print('Training')
layer_sizes_list = [(64, 32), (100, 50), (128, 64)]

for ls in layer_sizes_list:
  mlp = MLPClassifier(hidden_layer_sizes=ls, activation='relu', solver='adam', max_iter=300, random_state=42)
  mlp.fit(X_train_vec, y_train_encoded)

  y_pred = mlp.predict(X_test_vec)
  y_pred_probs = mlp.predict_proba(X_test_vec)

  result_generator('mlp', y_test_encoded, y_pred, y_pred_probs)

  with open('mlp_test_%s_%s.pkl' %(ls[0], ls[1]), 'wb') as file:
      pickle.dump(mlp, file)
'''


#?### Ensemble Classifier - Logistic Regression, Linear SVC, and Random Forest
print('\nEnsemble')
print('Training')

log_reg = LogisticRegression()
clf = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False, max_iter=300), cv=5)
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)

ensemble = VotingClassifier(estimators=[("log-reg", log_reg), ("lin-svc", clf), ("rand-for", random_forest)], voting='soft')
ensemble.fit(X_train_vec, y_train_encoded)

y_pred = ensemble.predict(X_test_vec)
y_pred_probs = ensemble.predict_proba(X_test_vec)

result_generator('ensemble', y_test_encoded, y_pred, y_pred_probs)

with open('ensemble_test.pkl', 'wb') as file:
  pickle.dump(ensemble, file)

#*### Results File ####
with open('test_model_body_results_hyper.txt', "w") as file:
   for item in results:
    file.write(item + '\n')