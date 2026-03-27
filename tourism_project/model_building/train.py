import os
import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import hf_hub_download

HF_DATASET_REPO = "treddy333/visit-with-us-predict"
HF_MODEL_REPO = "treddy333/tourism-wellness-model"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("tourism-mlops-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

def load_csv(repo, filename):
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    return pd.read_csv(path)

Xtrain = load_csv(HF_DATASET_REPO, "Xtrain.csv")
Xtest = load_csv(HF_DATASET_REPO, "Xtest.csv")
ytrain = load_csv(HF_DATASET_REPO, "ytrain.csv").squeeze("columns")
ytest = load_csv(HF_DATASET_REPO, "ytest.csv").squeeze("columns")

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

numeric_features = [c for c in numeric_features if c in Xtrain.columns]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("randomforestclassifier", rf)]
)

param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [6, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 2, 4],
}

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, refit=True
    )
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True, zero_division=0
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True, zero_division=0
    )

    def _pos(rep):
        return rep.get("1", rep.get(1, {}))

    tr1, te1 = _pos(train_report), _pos(test_report)
    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_precision": tr1.get("precision", 0),
            "train_recall": tr1.get("recall", 0),
            "train_f1-score": tr1.get("f1-score", 0),
            "test_accuracy": test_report["accuracy"],
            "test_precision": te1.get("precision", 0),
            "test_recall": te1.get("recall", 0),
            "test_f1-score": te1.get("f1-score", 0),
        }
    )

    model_path = "best_tourism_prod_taken_model_v1.joblib"
    joblib.dump(best_model, model_path, compress=3)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as MLflow artifact: {model_path}")

try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Model repo '{HF_MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo '{HF_MODEL_REPO}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)
print("Best model uploaded to Hugging Face Model Hub.")
