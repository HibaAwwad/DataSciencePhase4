import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
df = pd.read_csv('c:/Users/hp/Desktop/phase4_datascience/clean_data.csv')

# Drop completely empty columns or ones not useful for prediction
cols_to_drop = ['FLT_DATE', 'APT_NAME']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Features and target
df['avg_delay'] = df['DLY_APT_ARR_1'] / df['FLT_ARR_1'].replace(0, 1)
df['delayed_flag'] = (df['avg_delay'] > 15).astype(int)

print("Preparing features...")
features = ['YEAR', 'MONTH_NUM', 'STATE_NAME', 'season', 'weekday', 'FLT_ARR_1', 'APT_ICAO']
X = df[features].copy()

# Encode categorical
for col in ['STATE_NAME', 'season', 'APT_ICAO']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

y_reg = df['avg_delay']
y_clf = df['delayed_flag']

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

print("Training Decision Tree Regressor...")
dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_reg.fit(X_train, y_train_reg)
y_pred_reg = dt_reg.predict(X_test)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Decision Tree R^2 Score (Regression on avg_delay): {r2:.4f}")

print("Training Decision Tree Classifier...")
dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_clf.fit(X_train, y_train_clf)
y_pred_clf = dt_clf.predict(X_test)
acc = accuracy_score(y_test_clf, y_pred_clf)
print(f"Decision Tree Accuracy (Classification on delayed_flag): {acc:.4f}")
print(classification_report(y_test_clf, y_pred_clf))
