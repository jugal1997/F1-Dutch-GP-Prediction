import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load prepared data
df = pd.read_csv('dutch_gp_processed.csv')
feature_cols = [col for col in [
    'Year', 'GridPosition', 'Points', 'Laps', 'Finished', 'Driver_encoded', 'Team_encoded'
] if col in df.columns]
X = df[feature_cols]
y = df['Winner']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Model and parameter grid definitions
models_and_params = [
    ('Logistic Regression', LogisticRegression(solver='liblinear'), {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }),
    ('Random Forest', RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3, 5, 10]
    }),
    ('SVM (RBF kernel)', SVC(probability=True), {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1]
    }),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8],
        'learning_rate': [0.01, 0.1, 0.2]
    })
]

# Loop through models
for name, model, param_grid in models_and_params:
    print(f"\n===== {name} =====")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

