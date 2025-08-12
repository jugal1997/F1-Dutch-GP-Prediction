import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the preprocessed dataset
df = pd.read_csv('dutch_gp_processed.csv')

# Features and target
feature_cols = [col for col in [
    'Year', 'GridPosition', 'Points', 'Laps', 'Finished', 'Driver_encoded', 'Team_encoded'
] if col in df.columns]
X = df[feature_cols]
y = df['Winner']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Fit the (baseline) model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Attach predictions back to the test set
X_test_copy = X_test.copy()
X_test_copy['Predicted'] = y_pred

# Add driver and year info from the original df
X_test_copy['Driver_encoded'] = df.loc[X_test.index, 'Driver_encoded']
X_test_copy['Year'] = df.loc[X_test.index, 'Year']

# Show predicted winners
predicted_winners = X_test_copy[X_test_copy['Predicted'] == 1][['Year', 'Driver_encoded']]
print("\nPredicted winners in the test set:")
print(predicted_winners)

# Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nOverall Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

# (Optional) Feature importance
import matplotlib.pyplot as plt
importances = clf.feature_importances_
feat_importance = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
feat_importance.plot(kind='bar')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
