from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score


class SVMModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = OneVsRestClassifier(SVC(kernel="linear", probability=True))

    def train(self, X_train, y_train):
        X = self.vectorizer.fit_transform(X_train)
        self.model.fit(X, y_train)

    def predict(self, X_test):
        X = self.vectorizer.transform(X_test)
        return self.model.predict(X)

    def predict_proba(self, X_test):
        X = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        X = self.vectorizer.transform(X_test)
        return self.model.score(X, y_test)

    def cross_validate(self, X, y, cv=5):
        X_transformed = self.vectorizer.fit_transform(X)
        scores = cross_val_score(self.model, X_transformed, y, cv=cv)
        return scores
