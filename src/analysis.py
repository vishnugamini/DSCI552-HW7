from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

LABELS = ["Family", "Genus", "Species"]


def triplet_metrics(actual, predicted):
    mismatches = np.asarray(actual) != np.asarray(predicted)
    distance = mismatches.sum(axis=1)
    loss = mismatches.mean()
    return {
        "exact_match": (distance == 0).mean(),
        "hamming_distance": distance.mean(),
        "hamming_loss": loss,
        "hamming_score": 1 - loss,
        "set_jaccard": np.mean((3 - distance) / (3 + distance)),
    }


def svm_search(kind, folds):
    if kind.startswith("RBF"):
        scaled = kind == "RBF standardized"
        gamma = [.01, .05, .25, 1.] if scaled else [.1, .5, 2., 10.]
        estimator = SVC(kernel="rbf", cache_size=200)
        grid = {
            "svm__estimator__C": [.1, 1., 10., 100.],
            "svm__estimator__gamma": gamma,
        }
    else:
        scaled = True
        estimator = LinearSVC(
            penalty="l1", loss="squared_hinge", dual=False,
            tol=1e-4, max_iter=30000, random_state=42,
        )
        grid = {"svm__estimator__C": [.001, .01, .1, 1., 10.]}

    pipeline = Pipeline([
        ("scale", StandardScaler() if scaled else "passthrough"),
        ("smote", SMOTE(random_state=42) if kind == "L1 SMOTE"
         else "passthrough"),
        ("svm", OneVsRestClassifier(estimator)),
    ])
    return GridSearchCV(
        pipeline, grid, cv=folds, scoring="accuracy", n_jobs=4,
        error_score="raise",
    )


def fit_svms(kind, X_train, y_train, X_test, y_test, folds, output):
    predictions = pd.DataFrame(index=y_test.index)
    rows, grids = [], []
    for label in LABELS:
        search = svm_search(kind, folds).fit(X_train, y_train[label])
        predicted = search.predict(X_test)
        predictions[label] = predicted
        params = search.best_params_
        gamma = params.get("svm__estimator__gamma", np.nan)
        rows.append({
            "model": kind, "label": label,
            "C": params["svm__estimator__C"], "gamma": gamma,
            "sigma": np.sqrt(1 / (2 * gamma)),
            "cv_accuracy": search.best_score_,
            "test_accuracy": np.mean(predicted == y_test[label]),
            "macro_recall": balanced_accuracy_score(y_test[label], predicted),
        })
        cv = pd.DataFrame(search.cv_results_)
        columns = ["param_svm__estimator__C", "mean_test_score",
                   "std_test_score", "rank_test_score"]
        if kind.startswith("RBF"):
            columns.insert(1, "param_svm__estimator__gamma")
        columns += [f"split{i}_test_score" for i in range(len(folds))]
        grids.append(cv[columns].assign(label=label))
    stem = kind.lower().replace(" ", "_")
    pd.concat(grids, ignore_index=True).to_csv(output / f"{stem}_cv.csv",
                                              index=False)
    predictions.to_csv(output / f"{stem}_predictions.csv", index_label="row")
    return pd.DataFrame(rows), predictions


def cluster_trials(X, labels, output, repeats=50):
    trials, candidates, majorities, assignments = [], [], [], []
    for seed in range(repeats):
        best_score = -np.inf
        for k in range(1, 51):
            model = KMeans(n_clusters=k, n_init=1, random_state=seed)
            membership = model.fit_predict(X)
            score = calinski_harabasz_score(X, membership) if k > 1 else np.nan
            candidates.append((seed, k, model.inertia_, score))
            if score > best_score:
                best_score, best_k, best_membership = score, k, membership
        grouped = labels.groupby(best_membership, sort=True)
        modes = grouped.agg(lambda values: values.mode().iloc[0])
        assigned = modes.loc[best_membership].to_numpy()
        trials.append({
            "seed": seed, "k": best_k, "CH": best_score,
            **triplet_metrics(labels, assigned),
        })
        majorities.append(modes.assign(seed=seed, size=grouped.size()))
        assignments.append(pd.DataFrame({
            "seed": seed, "row": labels.index, "cluster": best_membership
        }))
    trials = pd.DataFrame(trials)
    trials.to_csv(output / "clustering_trials.csv", index=False)
    pd.DataFrame(candidates, columns=["seed", "k", "inertia", "CH"]).to_csv(
        output / "cluster_selection.csv", index=False)
    pd.concat(majorities).to_csv(output / "cluster_majorities.csv",
                                 index_label="cluster")
    pd.concat(assignments).to_csv(output / "cluster_assignments.csv",
                                  index=False)
    return trials, majorities[0]


def linkage_figures(distances):
    condensed = squareform(distances)
    complete = linkage(condensed, method="complete")
    single = linkage(condensed, method="single")
    swapped = complete.copy()
    swapped[:, :2] = swapped[:, :2][:, ::-1]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, tree, title in zip(
        axes, [complete, single, swapped],
        ["(a) Complete", "(b) Single", "(e) Complete, reversed"],
    ):
        dendrogram(tree, labels=["1", "2", "3", "4"], ax=ax)
        ax.set(title=title, xlabel="Observation", ylabel="Dissimilarity",
               ylim=(0, .9), yticks=[0, .3, .4, .45, .8])
    fig.tight_layout()
    plt.show()
    return [fcluster(tree, 2, criterion="maxclust")
            for tree in [complete, single, swapped]]
