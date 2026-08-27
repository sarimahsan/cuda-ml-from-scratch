import pandas as pd
import numpy as np
import urllib.request
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from logistic_regression import CUDALogisticRegression


def load_titanic_dataset() -> pd.DataFrame:
    """
    Downloads and loads the real Titanic dataset from GitHub raw mirror.
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    local_path = os.path.join(os.path.dirname(__file__), "titanic.csv")

    if not os.path.exists(local_path):
        print(f"[INFO] Downloading Titanic dataset from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print("[INFO] Download complete!")

    df = pd.read_csv(local_path)
    return df


def preprocess_titanic(df: pd.DataFrame):
    """
    Preprocess Titanic dataset:
    - Handle missing values (Age -> median, Embarked -> mode, Fare -> median)
    - Feature engineering (FamilySize)
    - Categorical encoding (Sex, Embarked, Pclass)
    - Standard scaling
    """
    df = df.copy()

    # Drop non-predictive identifier columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Impute missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Binary and One-Hot Encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"], drop_first=True)

    # Separate Features and Target
    target_col = "Survived"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    return X, y, feature_cols


def main():
    print("================================================================")
    print("       CUDA ML Models: Logistic Regression on Titanic Data      ")
    print("================================================================\n")

    # 1. Load and Preprocess Data
    raw_df = load_titanic_dataset()
    print(f"[INFO] Raw Dataset shape: {raw_df.shape}")

    X, y, feature_names = preprocess_titanic(raw_df)
    print(f"[INFO] Processed Features ({len(feature_names)}): {feature_names}")

    # 2. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"[INFO] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")

    # 3. Instantiate and Train CUDA Logistic Regression Model
    model = CUDALogisticRegression(
        lr=0.1,
        epochs=3000,
        device="cuda"
    )

    model.fit(X_train, y_train, verbose=True)

    # 4. Evaluation on Test Set
    print("\n----------------------------------------------------------------")
    print("                   Test Evaluation Metrics                      ")
    print("----------------------------------------------------------------")

    y_test_proba = model.predict_proba(X_test).cpu().numpy()
    y_test_pred = model.predict(X_test).cpu().numpy()

    acc = accuracy_score(y_test, y_test_pred) * 100.0
    prec = precision_score(y_test, y_test_pred) * 100.0
    rec = recall_score(y_test, y_test_pred) * 100.0
    f1 = f1_score(y_test, y_test_pred) * 100.0
    roc_auc = roc_auc_score(y_test, y_test_proba)

    print(f"  Test Accuracy:  {acc:.2f}%")
    print(f"  Precision:      {prec:.2f}%")
    print(f"  Recall:         {rec:.2f}%")
    print(f"  F1-Score:       {f1:.2f}%")
    print(f"  ROC-AUC Score:  {roc_auc:.4f}")

    # 5. Learned Feature Weights
    print("\n----------------------------------------------------------------")
    print("                  Learned Model Parameters                      ")
    print("----------------------------------------------------------------")
    print(f"  Bias (Intercept): {model.bias:.4f}")
    print("  Feature Coefficients:")
    weights = model.weights
    for name, weight in zip(feature_names, weights):
        print(f"    {name:<15}: {weight:+.4f}")

    print("\n================================================================")
    print("                     Training Successful!                       ")
    print("================================================================")


if __name__ == "__main__":
    main()
