import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RNG = 42
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RNG
)

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None, random_state=RNG),
    n_estimators=200, bootstrap=True, random_state=RNG, n_jobs=-1
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=RNG),
    n_estimators=200, learning_rate=0.8, random_state=RNG
)

models = [("Bagging (Trees)", bag), ("AdaBoost (Decision Stumps)", ada)]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
scoring = {"acc": "accuracy", "f1m": "f1_macro"}

for name, model in models:
    cvres = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n=== {name} ===")
    print(f"CV Accuracy : {cvres['test_acc'].mean():.3f} ± {cvres['test_acc'].std():.3f}")
    print(f"CV Macro F1 : {cvres['test_f1m'].mean():.3f} ± {cvres['test_f1m'].std():.3f}")

for name, model in models:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\n--- Hold-out Test: {name} ---")
    print(f"Accuracy : {acc:.3f} | Macro F1 : {f1m:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=iris.target_names))
