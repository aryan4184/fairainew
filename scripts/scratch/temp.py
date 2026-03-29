import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("data/adult_reconstruction.csv")



def income_class(x):
    if x < 25000:
        return "Low"
    elif x <= 60000:
        return "Mid"
    else:
        return "High"

df["income_class"] = df["income"].apply(income_class)

le = LabelEncoder()
df["income_class_encoded"] = le.fit_transform(df["income_class"])

print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df["income_class"].value_counts())


X = df.drop(columns=["income", "income_class", "income_class_encoded"])
y = df["income_class_encoded"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), 
        LogisticRegression(max_iter=5000, multi_class="multinomial", n_jobs=-1, class_weight='balanced')
    ),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42, max_iter=200)
}


results = []
trained_models = {}

for name, model in models.items():
    print(f"\n===== {name} =====")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc})
    trained_models[name] = model
    
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, digits=4))


print("\n===== Voting Classifier =====")
voting_clf = VotingClassifier(
    estimators=[
        ('hgb', trained_models['Hist Gradient Boosting']),
        ('rf', trained_models['Random Forest']),
        ('lr', trained_models['Logistic Regression'])
    ],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_test)
acc_vote = accuracy_score(y_test, y_pred_vote)
results.append({"Model": "Voting Classifier", "Accuracy": acc_vote})
print("Accuracy:", acc_vote)


results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n=== Model Comparison ===")
print(results_df)



print("\nTotal features:", X_train.shape[1])
