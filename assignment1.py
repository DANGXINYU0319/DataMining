import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)

# Data Exploration and Cleaning

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Drop rows with missing label
train = train.dropna(subset=["label"])
test = test.dropna(subset=["label"])

X_train = train.drop(columns="label")
X_test = test.drop(columns="label")

y_train = (train["label"] == "yes").astype(int)
y_test = (test["label"] == "yes").astype(int)

# Split columns into numeric and categorical
numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
categorical_columns = [a for a in X_train.columns if a not in numeric_columns]


# Exploratory Data Analysis

print("Class proportion in the training set")
print(train["label"].value_counts(normalize=True))

print("Missing values in training set:", X_train.isna().sum().sum())
print("Missing values in test set:", X_test.isna().sum().sum())

print("Numeric column summary")
print(X_train[numeric_columns].describe().loc[["mean", "50%", "min", "max"]])

# Plot: left: class distribution bar chart; right: box plots of numeric features
normalised = (X_train[numeric_columns] - X_train[numeric_columns].mean()) / X_train[numeric_columns].std()
figure, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(["no", "yes"], [(y_train == 0).sum(), (y_train == 1).sum()])
axes[0].set_title("Class distribution")

axes[1].boxplot([normalised[a].dropna() for a in numeric_columns])
axes[1].set_xticklabels(numeric_columns, rotation=45, ha="right")
axes[1].set_title("Box plots of numeric features (z-scored)")

plt.tight_layout()
plt.savefig("eda.png")

# Data Cleaning

# Missing Value Imputation and Feature Engineering

tree_preprocessing = ColumnTransformer([
    ("numeric_part", SimpleImputer(strategy="median"), numeric_columns),
    ("categorical_part", Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(min_frequency=0.01)),
    ]), categorical_columns),
])

knn_preprocessing = ColumnTransformer([
    ("numeric_part", Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("standardise", StandardScaler()),
    ]), numeric_columns),
    ("categorical_part", Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(min_frequency=0.01)),
    ]), categorical_columns),
])

# Model Training (Hyperparameter tuning via grid search and 5-fold cross validation)

# Decision Tree hyperparameters: max depth and min samples leaf
tree_param_grid = {
    "model__max_depth": [3, 5, 10, 15],
    "model__min_samples_leaf": [20, 50, 100, 200],
}

# KNN hyperparameters: k and p
knn_param_grid = {
    "model__n_neighbors": [5, 21, 51, 101],
    "model__p": [1, 2],
}

tree_pipeline = Pipeline([
    ("prepare", tree_preprocessing),
    ("model", DecisionTreeClassifier(criterion="gini", random_state=42)),
])

knn_pipeline = Pipeline([
    ("prepare", knn_preprocessing),
    ("model", KNeighborsClassifier()),
])

# F-measure is used as the scoring metric for hyperparameter selection
tree_search = GridSearchCV(tree_pipeline, tree_param_grid, scoring="f1", cv=5)
tree_search.fit(X_train, y_train)

knn_search = GridSearchCV(knn_pipeline, knn_param_grid, scoring="f1", cv=5)
knn_search.fit(X_train, y_train)

print("Decision Tree best parameters:", tree_search.best_params_)
print("Decision Tree cross-validation mean F-measure:", round(tree_search.best_score_, 4))

print("KNN best parameters:", knn_search.best_params_)
print("KNN cross-validation mean F-measure:", round(knn_search.best_score_, 4))

# Evaluation and Comparison

final_models = {
    "Decision Tree": tree_search.best_estimator_,
    "KNN": knn_search.best_estimator_,
}

for model_name, model in final_models.items():
    predicted_labels = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, predicted_labels
    ).ravel()

    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    roc_auc = roc_auc_score(y_test, probability)
    pr_auc = average_precision_score(y_test, probability)

    print(f"{model_name} performance on the test set:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F-measure:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Confusion matrix: TN =", true_negative, " FP =", false_positive,
          " FN =", false_negative, " TP =", true_positive)
    Z = 1.96
    N = len(y_test)
    common_part = 2 * N * accuracy + Z ** 2
    sqrt_part = Z * math.sqrt(Z ** 2 + 4 * N * accuracy - 4 * N * accuracy ** 2)
    denominator = 2 * (N + Z ** 2)
    lower_bound = (common_part - sqrt_part) / denominator
    upper_bound = (common_part + sqrt_part) / denominator
    print(f"95% confidence interval for accuracy: "f"[{lower_bound:.4f}, {upper_bound:.4f}]")