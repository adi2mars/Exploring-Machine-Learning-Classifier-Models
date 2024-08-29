import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree as treeViz
import graphviz
from IPython.display import display


def organize_data(df):
    df["Q5"] = df["Q5"].fillna('')
    df["Q6"] = df["Q6"].fillna('')
    df["Q10"] = df["Q10"].fillna('')

    # Convert Q5 categories into binary features
    q5_categories = ["Partner", "Friends", "Siblings", "Co-worker"]
    for category in q5_categories:
        df[f"Q5_{category}"] = df["Q5"].apply(
            lambda s: 1 if category in s else 0)

    # Convert Q6 categories into binary features
    q6_categories = ["Skyscrapers", "Sport", "Art and Music", "Carnival",
                     "Cuisine", "Economic"]
    for category in q6_categories:
        numbers = [int(re.findall(f"{category}=>(\d+)", s)[0]) if re.findall(
            f"{category}=>(\d+)", s) else 0 for s in
                   df["Q6"]]
        df[f"Q6_{category}"] = numbers

    # Drop original columns
    df.drop(["Q5", "Q6"], axis=1, inplace=True)

    # fix string values input to Q7 and Q9
    df['Q7'] = df['Q7'].str.replace(',', '').fillna(0).astype(float)
    df['Q9'] = df['Q9'].str.replace(',', '').fillna(0).astype(float)

    df["Q10"] = df["Q10"].astype(str)
    df["Q10"] = df["Q10"].apply(lambda s: re.sub(r'[^a-zA-Z\s]', ' ', s))
    df["Q10"] = df["Q10"].apply(lambda s: ' '.join(s.split()))

    vocab_list = list()
    for text in df["Q10"]:
        words = text.split()
        vocab_list.extend(words)
    vocab_list = list(set(word.lower() for word in vocab_list))
    for word in vocab_list:
        df[word] = 0

    for i, text in enumerate(df["Q10"]):
        word_counter = Counter(text.split())
        for word, freq in word_counter.items():
            df.at[i, word.lower()] = freq

    return df, vocab_list


# MODEL 1 - kNN
def kNN(X, t, X_valid, t_valid, X_test, t_test):
    d = {}
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    epsilon = 0.0001
    X_norm = (X - X_mean) / (X_std + epsilon)
    X_valid_norm = (X_valid - X_mean) / (X_std + epsilon)
    X_test_norm = (X_test - X_mean) / (X_std + epsilon)
    for k in range(1, 10):
        for metric in ['euclidean', 'cosine']:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_norm, t)

            t_pred = knn.predict(X_norm)
            accuracy_train = accuracy_score(t_pred, t)

            t_pred_valid = knn.predict(X_valid_norm)
            accuracy_valid = accuracy_score(t_pred_valid, t_valid)

            t_pred_test = knn.predict(X_test_norm)
            accuracy_test = accuracy_score(t_pred_test, t_test)
            d[(k, metric)] = (accuracy_train, accuracy_valid, accuracy_test)

    return d


# MODEL 2 - MLP
def MLP(X_train, t_train, X_valid, t_valid, X_test, t_test):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    epsilon = 0.0001
    X_norm = (X_train - X_mean) / (X_std + epsilon)
    X_valid_norm = (X_valid - X_mean) / (X_std + epsilon)
    X_test_norm = (X_test - X_mean) / (X_std + epsilon)

    d = {}
    layers = [1, 2, 4]
    neurons = [5, 20, 50, 100]
    for i in ['identity', 'logistic', 'tanh', 'relu']:
        for j in layers:
            for k in neurons:
                hiddenLayer = [k]*j
                hiddenLayer = tuple(hiddenLayer)
                clf = MLPClassifier(hidden_layer_sizes=hiddenLayer, activation=i, random_state=1, max_iter=5000).fit(X_norm, t_train)

                accuracy_train = clf.score(X_norm, t_train)
                accuracy_valid = clf.score(X_valid_norm, t_valid)
                accuracy_test = clf.score(X_test_norm, t_test)

                d[(i, j, k)] = (accuracy_train, accuracy_valid, accuracy_test)

    return d

# MODEL 3 - Decision Tree
def DecisionTree(X_train, t_train, X_valid, t_valid, X_test, t_test):
    res = {}
    criterions = ["entropy", "gini", "log_loss"]
    max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
    min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for c in criterions:
        for d in max_depths:
            for s in min_samples_split:
                tree = DecisionTreeClassifier(criterion=c, max_depth=d,
                                              min_samples_split=s,
                                              random_state=1)
                tree.fit(X_train, t_train)

                accuracy_train = tree.score(X_train, t_train)
                accuracy_valid = tree.score(X_valid, t_valid)
                accuracy_test = tree.score(X_test, t_test)

                res[(c, d, s)] = (accuracy_train, accuracy_valid, accuracy_test)

    return res


def predict_all(filename):
    df = pd.read_csv(filename)
    df, vocab = organize_data(df)

    df = df[~df['Q1'].isnull()]
    df = df[~df['Q2'].isnull()]
    df = df[~df['Q3'].isnull()]
    df = df[~df['Q8'].isnull()]

    t = df["Label"]
    X = df.drop(["Q10", "Label"], axis=1)


    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2,
                                                        random_state=21)
    X_train, X_valid, t_train, t_valid = train_test_split(X_train, t_train,
                                                          test_size=0.2,
                                                          random_state=21)

    # run models and get a dictionary with hyperparameter key and (training accuracy, validation accuracy, testing accuracy) tuple

    # kNN scores, dict with key as (k, metric) --> tuple
    kNN_scores = kNN(X_train, t_train, X_valid, t_valid, X_test, t_test)

    bestModel = None
    highestAccuracy = -1
    for key, val in kNN_scores.items():
        if val[1] > highestAccuracy:
            highestAccuracy = val[1]
            bestModel = key
    print("key:   " + str(bestModel) + "   val:  " + str(kNN_scores[bestModel]))
    #
    #
    #
    #
    # # MLP scores, dict with key as max_iter --> int
    MLP_scores = MLP(X_train, t_train, X_valid, t_valid, X_test, t_test)
    bestModel = None
    highestAccuracy = -1
    for key, val in MLP_scores.items():
        if val[1] > highestAccuracy:
            highestAccuracy = val[1]
            bestModel = key
    print("key:   " + str(bestModel) + "   val:  " + str(MLP_scores[bestModel]))

    # DecisionTree scores, dict with key as (criterion, max_depth, min_sample_split) --> tuple
    DecisionTree_scores = DecisionTree(X_train, t_train, X_valid, t_valid, X_test, t_test)
    bestModel = None
    highestAccuracy = -1
    for key, val in DecisionTree_scores.items():
        if val[1] > highestAccuracy:
            highestAccuracy = val[1]
            bestModel = key
    print("key:   " + str(bestModel) + "   val:  " + str(DecisionTree_scores[bestModel]))


# Example usage:
if __name__ == "__main__":
    filename = "clean_dataset.csv"  # Provide the filename of your CSV data file
    predict_all(filename)
