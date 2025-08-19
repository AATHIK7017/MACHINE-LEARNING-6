import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------
# Step 1: Create Dataset
# --------------------------
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Sleep_Hours':   [7, 6, 8, 5, 6, 5, 4, 3],
    'Result':        ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

# --------------------------
# Step 2: Encode Target Variable
# --------------------------
le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])  # Fail=0, Pass=1

# Features (X) and Target (y)
X = df[['Hours_Studied', 'Sleep_Hours']]
y = df['Result']

# --------------------------
# Step 3: Split Data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --------------------------
# Step 4: Train Logistic Regression Model
# --------------------------
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# --------------------------
# Step 5: Make Predictions
# --------------------------
y_pred = log_reg.predict(X_test)

# --------------------------
# Step 6: Evaluate Model
# --------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
