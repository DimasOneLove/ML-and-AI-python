## 1. Убрать из данных iris часть точек (на которых обучаемся) и убедиться, что на предсказание влияют только опорные вектора

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "virginica") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_virginica = data_df[data_df["species"] == "virginica"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

delete_percent = [[0, 10, 20, 30], [50, 60, 70, 80]]
support_indices = []
n_already_deleted = 0
DF_LEN = len(data_df)
print(DF_LEN)
fig, ax = plt.subplots(2,4, sharex="col", sharey="row")

for i in range(2):
    for j in range(4):
        n_to_remove = int(DF_LEN * delete_percent[i][j] / 100 - n_already_deleted)
        if n_to_remove:
            rng = np.random.RandomState(1)
            non_support_indices = np.setdiff1d(np.array(data_df.index), support_indices)
            indices_to_remove = rng.choice(non_support_indices, size=n_to_remove, replace=False)
            data_df = data_df.drop(index=indices_to_remove)
            n_already_deleted += len(indices_to_remove)

        X = data_df[["sepal_length", "petal_length"]]
        y = data_df["species"]

        data_df_virginica = data_df[data_df["species"] == "virginica"]
        data_df_versicolor = data_df[data_df["species"] == "versicolor"]


        ax[i,j].scatter(data_df_virginica["sepal_length"], data_df_virginica["petal_length"])
        ax[i,j].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

        model = SVC(kernel="linear", C=1000)
        model.fit(X, y)

        # support_indices = model.support_
        support_indices = data_df.iloc[model.support_].index.to_numpy()
        print(model.support_)
        ax[i,j].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=400, facecolors='none', edgecolors='b')

        x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
        x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)
        X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"])
        # print(X_p.head())

        y_p = model.predict(X_p)

        X_p["species"] = y_p

        X_p_virginica = X_p[X_p["species"] == "virginica"]
        X_p_versicolor = X_p[X_p["species"] == "versicolor"]
        # print(X_p.head())

        ax[i,j].scatter(X_p_virginica["sepal_length"], X_p_virginica["petal_length"], alpha=0.1)
        ax[i,j].scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.1)


plt.show()
