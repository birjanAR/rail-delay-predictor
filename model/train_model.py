import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# -----------------------------
# 1. CREATE SIMPLE SYNTHETIC DATA
# -----------------------------

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "hour": np.random.randint(0, 24, n),
    "day_of_week": np.random.randint(0, 7, n),  # 0 = Monday
    "is_peak": np.random.randint(0, 2, n),
    "weather_delay_risk": np.random.randint(0, 3, n)  # 0=low,1=med,2=high
})

# Create a simple rule-based target (delayed or not)
data["delayed"] = (
    (data["is_peak"] * 0.4) +
    (data["weather_delay_risk"] * 0.4) +
    (np.where((data["hour"] >= 16) & (data["hour"] <= 19), 0.3, 0))
)

data["delayed"] = (data["delayed"] > 0.5).astype(int)

# -----------------------------
# 2. SPLIT DATA
# -----------------------------

X = data.drop("delayed", axis=1)
y = data["delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 4. EVALUATE MODEL
# -----------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 5. SAVE MODEL
# -----------------------------

joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")