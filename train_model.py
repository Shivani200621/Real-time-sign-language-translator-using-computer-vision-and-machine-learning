import csv
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

X = []
y = []

for file in os.listdir("data"):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")

        with open("data/" + file, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                if row:
                    X.append([float(value) for value in row])
                    y.append(label)

print("Samples:", len(X))

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("AI model trained successfully!")