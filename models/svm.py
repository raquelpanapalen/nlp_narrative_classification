from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class SVMModel:
    def __init__(self, max_features=1000, kernel="linear"):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = OneVsRestClassifier(SVC(kernel=kernel, probability=True))

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

    def grid_search_cv(self, X, y, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "clf__estimator__kernel": ["linear", "rbf", "poly"],
                "clf__estimator__C": [0.1, 1, 10, 100],
                "clf__estimator__gamma": ["scale", "auto"],
                "vectorizer__max_features": [None, 500, 1000, 1500, 2000, 5000],
                "vectorizer__max_df": (
                    0.5,
                    0.75,
                    1.0,
                ),  # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
                "vectorizer__ngram_range": (
                    (1, 1),
                    (1, 2),
                    (1, 3),
                ),
            }

        pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("clf", OneVsRestClassifier(SVC(probability=True))),
            ]
        )

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, n_jobs=-1, scoring="f1_samples"
        )
        grid_search.fit(X, y)

        print(grid_search.best_estimator_.named_steps["clf"])

        self.vectorizer = grid_search.best_estimator_.named_steps["vectorizer"]
        self.model = grid_search.best_estimator_.named_steps["clf"]

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")

        return grid_search.best_params_, grid_search.best_score_

    def cross_validate(self, X, y, cv=5):
        X_transformed = self.vectorizer.fit_transform(X)
        scores = cross_val_score(self.model, X_transformed, y, cv=cv)
        return scores
