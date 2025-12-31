import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import joblib

# -----------------------------
# 1️⃣ Load datasets
# -----------------------------
train_df = pd.read_parquet("dataset/UNSW_NB15_training-set.parquet")
test_df = pd.read_parquet("dataset/UNSW_NB15_testing-set.parquet")

train_df = train_df.dropna(subset=["attack_cat"])
test_df = test_df.dropna(subset=["attack_cat"])

# -----------------------------
# 2️⃣ Encode categorical features
# -----------------------------
categorical_cols = ["proto", "service", "state"]
for col in categorical_cols:
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    encoder = LabelEncoder()
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# -----------------------------
# 3️⃣ Encode target column
# -----------------------------
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df["attack_cat"])
y_test = target_encoder.transform(test_df["attack_cat"])

# -----------------------------
# 4️⃣ Prepare features
# -----------------------------
X_train = train_df.drop(columns=["attack_cat", "label"])
X_test = test_df.drop(columns=["attack_cat", "label"])

# -----------------------------
# 5️⃣ SMOTE to balance minority classes
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE, X_train shape: {X_train_res.shape}, y_train shape: {y_train_res.shape}")

# -----------------------------
# 6️⃣ Compute class weights (extra help for rare attacks)
# -----------------------------
classes = np.unique(y_train_res)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_res)
weight_dict = {i: w for i, w in zip(classes, class_weights)}
sample_weights = np.array([weight_dict[i] for i in y_train_res])

# -----------------------------
# 7️⃣ Initialize XGBoost with tuned params
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=1000,      # more trees
    max_depth=9,            # slightly deeper
    learning_rate=0.03,     # smaller learning rate
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train_res, y_train_res, sample_weight=sample_weights)

# -----------------------------
# 8️⃣ Train XGBoost (without early stopping for old version)
# -----------------------------
print("🚀 Training XGBoost model...")
xgb_model.fit(X_train_res, y_train_res, sample_weight=sample_weights)

# -----------------------------
# 9️⃣ Evaluate model
# -----------------------------
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 🔟 Save model & encoder
# -----------------------------
joblib.dump(xgb_model, "xgb_cyber_model.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
print("\n💾 Model and encoder saved for real-time prediction!")
