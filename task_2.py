import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from pygam import LogisticGAM, s

def split_to_random_train_test_with_stratify(data):

    X = data[:, 0 : -1]
    Y = data[:, -1]

    x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size=0.33, shuffle=True, stratify=Y)

    return x_train, x_test, y_train, y_test

def calculate_accuracy(y, y_hat): 
    return (y == y_hat).sum() / y.shape[0]

def task_2(x_train, x_test, y_train, y_test, X, Y, save_prefix):

    # Task 2 a)

    k_fold_dataset = KFold(n_splits=5, shuffle=True)


    k_values = list(range(1, 50))
    test_accuracy = np.zeros(len(k_values))

    for i, k in enumerate(k_values): 

        accuracy = np.zeros(5)
        index = 0
        for train_i, test_i in k_fold_dataset.split(X):
            x_cv_train, x_cv_test, y_cv_train, y_cv_test = X[train_i], X[test_i], Y[train_i], Y[test_i]

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_cv_train, y_cv_train)

            y_hat = knn.predict(x_cv_test)

            accuracy[index] = calculate_accuracy(y_cv_test, y_hat)
            index += 1


        test_accuracy[i] = np.mean(accuracy)

    # Leave one out 
    test_accuracy_leave_one_out = np.zeros(len(k_values))

    k_fold_dataset = KFold(n_splits=Y.shape[0], shuffle=True)

    for i, k in enumerate(k_values): 

        accuracy = np.zeros(Y.shape[0])
        index = 0
        for train_i, test_i in k_fold_dataset.split(X):
            x_cv_train, x_cv_test, y_cv_train, y_cv_test = X[train_i], X[test_i], Y[train_i], Y[test_i]

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_cv_train, y_cv_train)

            y_hat = knn.predict(x_cv_test)

            accuracy[index] = calculate_accuracy(y_cv_test, y_hat)
            index += 1


        test_accuracy_leave_one_out[i] = np.mean(accuracy)

    
    plt.figure()
    plt.plot(k_values, test_accuracy, label="CV-k=5")
    plt.plot(k_values, test_accuracy_leave_one_out, label="LOOCV")
    plt.xlabel("k")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.savefig(f"{save_prefix}_task_2_a.pdf", format="pdf")


    # Task 2 b
    
    gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7))
    backward_feature_selector = SequentialFeatureSelector(gam, n_features_to_select="auto", direction="backward")
    backward_feature_selector.fit(x_train, y_train)

    print("Features selected")
    print(backward_feature_selector.get_support())
    
    gam.fit(x_train, y_train)

    new_gam = LogisticGAM(s(4) + s(5) + s(6) + s(7))
    new_gam.fit(x_train, y_train)

    y_hat = gam.predict(x_test)
    y_hat_new = new_gam.predict(x_test)

    accuracy = calculate_accuracy(y_test, y_hat)
    accuracy_new = calculate_accuracy(y_test, y_hat_new)

    print("GAM accuracies")
    print(f"Accuracy with all features: {accuracy}")
    print(f"Accuracy with selected features: {accuracy_new}")

    # Task 2 c

    decision_tree = DecisionTreeClassifier()
    bagging_tree = BaggingClassifier(bootstrap=True)
    random_forest = RandomForestClassifier(bootstrap=True)
    
    decision_tree.fit(x_train, y_train)
    bagging_tree.fit(x_train, y_train)
    random_forest.fit(x_train, y_train)

    y_hat_decision = decision_tree.predict(x_test)
    y_hat_bagging = bagging_tree.predict(x_test)
    y_hat_forest = random_forest.predict(x_test)

    y_hat_decision_train = decision_tree.predict(x_train)
    y_hat_bagging_train = bagging_tree.predict(x_train)
    y_hat_forest_train = random_forest.predict(x_train)

    decision_accuracy_test = calculate_accuracy(y_test, y_hat_decision)
    bagging_accuracy_test = calculate_accuracy(y_test, y_hat_bagging)
    forest_accuracy_test = calculate_accuracy(y_test, y_hat_forest)

    decision_accuracy_train = calculate_accuracy(y_train, y_hat_decision_train)
    bagging_accuracy_train = calculate_accuracy(y_train, y_hat_bagging_train)
    forest_accuracy_train = calculate_accuracy(y_train, y_hat_forest_train)

    print(f"Decision tree train accuracy: {decision_accuracy_train}")
    print(f"Bagging tree train accuracy: {bagging_accuracy_train}")
    print(f"Random forest train accuracy: {forest_accuracy_train}")

    print(f"Decision tree test accuracy: {decision_accuracy_test}")
    print(f"Bagging tree test accuracy: {bagging_accuracy_test}")
    print(f"Random forest test accuracy: {forest_accuracy_test}")

if __name__ == "__main__": 
    np.random.seed(1234)

    headers = ["pregnant","glucose","pressure","triceps","insulin","mass","pedigree","age","diabetes"]
    data = np.genfromtxt("pimaindiansdiabetes.csv", delimiter=",", dtype=np.float32)

    X = data[:, 0 : -1]
    Y = data[:, -1]

    x_train, x_test, y_train, y_test = split_to_random_train_test_with_stratify(data)

    print("--- Original dataset ---")
    task_2(x_train, x_test, y_train, y_test, X, Y, "dataset_1")

    data = np.genfromtxt("pimaindiansdiabetes2.csv", delimiter=",", dtype=np.float32)
    
    X = data[:, 0 : -1]
    Y = data[:, -1]

    x_train, x_test, y_train, y_test = split_to_random_train_test_with_stratify(data)

    print("--- New dataset ---")
    task_2(x_train, x_test, y_train, y_test, X, Y, "dataset_2")

    plt.show()