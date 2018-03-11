from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class Parse_file:
    def __init__(self, categories, file, train):
        self.target_names = categories
        self.data = []
        self.filenames = []
        self.train = train
        file_train = open(file, 'rt', encoding="utf8")
        if train:
            self.target = []
            for line in file_train:
                doc = line.split('\t')
                self.data.append(doc[2])
                self.filenames.append(doc[1])
                self.target.append(self.target_names.index(doc[0]))
        else:
            for line in file_train:
                doc = line.split('\t')
                self.data.append(doc[1])
                self.filenames.append(doc[0])

    def print(self):
        print('docs number: ' + str(len(self.data)) + '\nfilenames number: ' + str(len(self.filenames)) + '\ncategories: ' + str(self.target_names))
        if self.train:
            print(self.target[:10])
        print()

docs_train = Parse_file(['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport'],
                        'news_train.txt',True)
docs_test = Parse_file(['science', 'style', 'culture', 'life', 'economics', 'business', 'travel', 'forces', 'media', 'sport'],
                        'news_test.txt', False)

docs_train.print()
docs_test.print()

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     #('clf', MultinomialNB())])
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, n_iter=5, random_state=50))])
print("training...")
text_clf = text_clf.fit(docs_train.data, docs_train.target)

print("predicting...")
print(np.mean(text_clf.predict(docs_train.data)== docs_train.target))

print("writing to file...")
out = open("news_output.txt", 'w')
for cat in text_clf.predict(docs_test.data):
  out.write("%s\n" % docs_test.target_names[cat])

'''
#WITHOUT PIPELINE

#build a dictionary of features and transform documents to feature vectors
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs_train.data)
#compute tf–idf
#fit() - fit estimator to the data, transform() - transform count-matrix to a tf-idf representation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#train a classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

X_new_counts = count_vect.transform(docs_test.data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_test.data, predicted):
    print('%r => %s' % (doc, docs_test.target_names[category]))

    from sklearn.grid_search import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(docs_train.data[:400], docs_train.target[:400])

print(np.mean(gs_clf.predict(docs_train.data) == docs_train.target))
'''
