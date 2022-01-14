import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse.lil import lil_matrix

from nltk.corpus import words as nltk_words
# from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer


plt.style.use('seaborn')
sns.set(font_scale=2)

ENGLISH_WORDS = set(nltk_words.words())


def tokenize(s, english_only=False, stopwords=None):
    regexp_tokenizer = RegexpTokenizer(r'[\w+\-\']+')
    words = [w.lower()
             .replace("'s", '')
             .replace("in'", 'ing')
             for w in regexp_tokenizer.tokenize(s.strip())]
    # if english_only:
    #     new_words = []
    #     for w in words:
    #         lemma = wordnet.morphy(w)
    #         if lemma is not None:
    #             if lemma in ENGLISH_WORDS:
    #                 new_words.append(w)
    #         elif w in ENGLISH_WORDS:
    #             new_words.append(w)
    #     words = new_words
    if stopwords is not None:
        words = [w for w in words if w not in stopwords]
    return words


def tokenizer(s):
    tokens = tokenize(s.strip(), english_only=True)
    tokens = [t for t in tokens if len(t) >= 4]
    return tokens


def create_keras_model(input_dim=None):
    from keras.models import Sequential
    from keras import layers
    model = Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


class NLPipeline:
    """Pipeline for NLP classification with vectorization and resampling

    Parameters
    ----------
    vectorizer : transformer object
        Object with `fit_transform` and `transform` methods for vectorizing
        corpus data.

    resampler : resampler object
        Object with fit_resample method for resampling training data in
        `Pipeline.fit`. This can be any under/oversampler from `imblearn`
        for binary or multiclass classification, or `MLSOL` from
        https://github.com/diliadis/mlsol/blob/master/MLSOL.py for multi-label
        classification.

    classifier : estimator object
        Binary, multi-class, or multi-label classifier with a `predict`
        or `predict_proba` method.

    Methods
    -------
    fit(X, y)
        Fit vectorizer and resampler, then train classifier on transformed data.

    predict(X)
        Return classification probabilities (if `self.classifier` has a
        `predict_proba` method, otherwise return predictions using `predict`).
    """
    def __init__(self, vectorizer, resampler, classifier, pad_features=False):
        self.vectorizer = vectorizer
        self.resampler = resampler
        self.classifier = classifier
        self.pad_features = pad_features
        self.padding = 0
        self.threshold = None
        self.labels = None

    @property
    def features(self):
        feature_names = self.vectorizer.get_feature_names()
        if self.pad_features:
            feature_names + [''] * self.padding
        return

    def fit(self, X, y, labels=None):
        self.labels = labels
        X_v = self.vectorizer.fit_transform(X).toarray()
        if self.pad_features:
            self.padding = self.vectorizer.max_features - len(self.vectorizer.get_feature_names())
            if self.padding > 0:
                padding_array = np.zeros((X_v.shape[0], self.padding))
                X_v = np.concatenate((X_v, padding_array), axis=1)
        X_r, y_r = self.resampler.fit_resample(X_v, y)
        self.classifier.fit(X_r, y_r)
        return self

    def predict(self, X):
        X_v = self.vectorizer.transform(X).toarray()
        if self.padding > 0:
            padding_array = np.zeros((X_v.shape[0], self.padding))
            X_v = np.concatenate((X_v, padding_array), axis=1)
        try:
            y_p = self.classifier.predict_proba(X_v)
        except AttributeError:
            y_p = self.classifier.predict(X_v)
        if (
                isinstance(y_p, csr_matrix) or
                isinstance(y_p, lil_matrix)
        ):
            y_p = y_p.toarray()
        return y_p

    def set_threshold(self, threshold):
        self.threshold = threshold
        return

    def classify_text(self, text, verbose=False):
        X_test = np.array([' '.join(text.lower().split())])
        prob = self.predict(X_test)[0]
        if self.threshold is not None:
            pred = prob > self.threshold
        else:
            pred = prob > 0.5
        if self.labels is not None:
            labels = self.labels
        else:
            labels = range(len(pred))
        results = [(label, prob[i], pred[i]) for i, label in enumerate(labels)]
        results.sort(key=lambda x: 1 - x[1])
        if verbose:
            print("Classification:")
            if results[0][2] < 1:
                print("NONE")
            else:
                print(", ".join([res[0].upper() for res in results if res[2] > 0]))
            print("\nIndividual label probabilities:")
            for res in results:
                print("{:<10s}{:>5.2g}%".format(res[0], 100 * res[1]))
        return results


# def plot_feature_importances(clf, vocab):
#     if hasattr(clf, 'feature_importances_'):
#         fi = clf.feature_importances_
#     elif hasattr(clf, 'coef_'):
#         fi = clf.coef_[0]
#     else:
#         raise AttributeError(f"Object {clf.__name__} has no feature importance attribute")
#     fi_top = fi.argsort()[-10:]
#     x_vals = range(len(fi_top))
#     fig = plt.figure(figsize=(8, 5))
#     plt.bar(x_vals, fi[fi_top])
#     plt.xticks(x_vals, vocab[fi_top], rotation=45)
#     return fig
