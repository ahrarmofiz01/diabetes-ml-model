import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Dataset load karo
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Features aur target alag karo
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Data split karo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model banao aur train karo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Model ko save karo .pkl file mein
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model train ho gaya aur save bhi ho gaya ✅")
