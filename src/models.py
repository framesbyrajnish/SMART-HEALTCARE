from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def train_random_forest(X, y, save_path=None, n_estimators=200, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    try:
        print('ROC AUC:', roc_auc_score(y_test, probs))
    except Exception:
        pass

    if save_path:
        joblib.dump(model, save_path)

    return model
