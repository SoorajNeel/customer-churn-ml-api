import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, precision_recall_curve
from preprocessing import load_data, clean_data, split_data, build_preprocessor


def train():
    df = load_data("data/teleco.csv")
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    # y_pred = pipeline.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # joblib.dump(pipeline, "models/model.pkl")

    probability_threshold = 0.4
    probs = pipeline.predict_proba(X_test)[:, 1]
    y_preds = (probs > probability_threshold).astype(int)
    print(classification_report(y_test, y_preds))

    # # test for different threshold
    # probs = pipeline.predict_proba(X_test)[:, 1]
    # from sklearn.metrics import precision_recall_curve

    # precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # for t in [0.5, 0.4, 0.35, 0.3]:
    #     preds = (probs >= t).astype(int)
    #     from sklearn.metrics import recall_score, precision_score

    #     print(f"Threshold {t}")
    #     print("Recall:", recall_score(y_test, preds))
    #     print("Precision:", precision_score(y_test, preds))
    #     print()


if __name__ == "__main__":
    train()
