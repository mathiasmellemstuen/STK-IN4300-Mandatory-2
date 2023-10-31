import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from pygam import GAM, s

def dataset_one_hot(data): 
    # Setting the count variables to one-hot encoding
    # Count variables to change: 
    # - H050 - index 2
    # - nN - index 6 
    # C040 - index 7

    data = data.copy()

    data[np.where(data[:, 2] != 0.0)] = 1
    data[np.where(data[:, 6] != 0.0)] = 1
    data[np.where(data[:, 7] != 0.0)] = 1

    return data

def calculate_mse(y, y_hat): 
    return (1 / y.shape[0]) * np.sum((y - y_hat) ** 2)

def calculate_aic(y_hat, y): 

    mse = calculate_mse(y, y_hat)
    N = y_hat.shape[0]
    num_features = 8

    return (-(2.0 / N) * np.log(mse)) + (2 * (num_features / N))

def calculate_bic(y_hat, y): 

    mse = calculate_mse(y, y_hat)
    N = y_hat.shape[0]
    num_features = 8
    return (-2.0 * np.log(mse)) + (num_features * np.log(y.shape[0]))

def split_to_random_train_test(data): 
    data = data.copy()
    
    train_dataset, test_dataset = train_test_split(data, test_size=0.33, shuffle=True)


    X_train = train_dataset[:, 0 : -1]
    Y_train = train_dataset[:, -1]

    X_test = test_dataset[:, 0 : -1]
    Y_test = test_dataset[:, -1]

    return X_train, Y_train, X_test, Y_test

def run_one_prediction(data): 
    # Splitting the datasets
    X_train, Y_train, X_test, Y_test = split_to_random_train_test(data)

    X_train_one_hot = dataset_one_hot(X_train)
    X_test_one_hot = dataset_one_hot(X_test)

    # Fitting the models
    lr_model = LinearRegression()
    lr_model.fit(X_train, Y_train)

    lr_model_one_hot = LinearRegression()
    lr_model_one_hot.fit(X_train_one_hot, Y_train)

    # Predicting and computing test accuracy
    y_hat = lr_model.predict(X_test)
    y_hat_one_hot = lr_model_one_hot.predict(X_test_one_hot)

    mse = calculate_mse(Y_test, y_hat)
    mse_one_hot = calculate_mse(Y_test, y_hat_one_hot)

    #print(calculate_aic(y_hat, Y_test, X_train.shape[1]))
    return mse, mse_one_hot

def plot_distribution(mse, title, save_name): 
    amount_of_bins = 20
    max_value = np.max(mse)
    min_value = np.min(mse)

    bins = np.linspace(min_value, max_value, amount_of_bins)
    frequencies = np.zeros(bins.shape[0])

    for value in mse:
        frequencies[np.argmin(np.abs(value - bins))] += 1
    
    plt.figure()
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("MSE")
    plt.bar(bins, frequencies)
    plt.savefig(save_name, format="pdf")

def task_1_c(X_train, Y_train):

    aic_scorer = make_scorer(calculate_aic, greater_is_better=False)
    bic_scorer = make_scorer(calculate_bic, greater_is_better=False)

    lr_model_aic_forward = SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", direction="forward", scoring = aic_scorer)
    lr_model_bic_forward = SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", direction="forward", scoring = bic_scorer)
    lr_model_aic_backward = SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", direction="backward", scoring = aic_scorer)
    lr_model_bic_backward = SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", direction="backward", scoring = bic_scorer)

    lr_model_aic_forward.fit(X_train, Y_train)
    lr_model_bic_forward.fit(X_train, Y_train)
    lr_model_aic_backward.fit(X_train, Y_train)
    lr_model_bic_backward.fit(X_train, Y_train)

    features_selected = np.array([
        lr_model_aic_forward.get_support(),
        lr_model_bic_forward.get_support(),
        lr_model_aic_backward.get_support(),
        lr_model_bic_backward.get_support()
    ])

    return features_selected

if __name__ == "__main__": 

    np.random.seed(1234)
    header = ["TPSA", "SAcc", "H050", "ML0GPM", "RDCHI", "GATS1p", "nN", "C040"]
    data = np.genfromtxt("qsar_aquatic_toxicity.csv", delimiter=";", dtype=np.float32)

    num_repetitions = 200

    # Task 1 b)
    mse = np.zeros(num_repetitions)
    mse_one_hot = np.zeros(num_repetitions)

    for i in range(num_repetitions):
        mse[i], mse_one_hot[i] = run_one_prediction(data)

    plot_distribution(mse, "MSE", "task_1_b_distribution_mse.pdf")
    plot_distribution(mse_one_hot, "MSE: One hot", "task_1_b_distribution_one_hot_mse.pdf")

    run_one_prediction(data)

    # Task 1 c)

    X_train, Y_train, X_test, Y_test = split_to_random_train_test(data)

    X_train_one_hot = dataset_one_hot(X_train)
    
    print("Features selected:")
    features_selected = task_1_c(X_train, Y_train)
    print(features_selected)

    print("Features selected one-hot:")
    features_selected_one_hot = task_1_c(X_train_one_hot, Y_train)
    print(features_selected_one_hot)

    # Task 1 d)
    bootstrap_data = np.copy(data)
    lambda_paramters_to_test = np.linspace(0, 10, 100)

    results = np.zeros((4, lambda_paramters_to_test.shape[0]))

    bootstrap_n_iterations = 1024

    cv_k_folds = 10

    for i in range(lambda_paramters_to_test.shape[0]): 
        current_lambda = lambda_paramters_to_test[i]
        
        # Bootstrapping 
        bootstrap_mse = np.zeros((2, bootstrap_n_iterations))

        for j in range(bootstrap_n_iterations):
            data_bs = resample(bootstrap_data, replace=True)

            x_bs_train, y_bs_train, x_bs_test, y_bs_test = split_to_random_train_test(data_bs)

            model = Ridge(alpha = current_lambda)

            model.fit(x_bs_train, y_bs_train)

            y_hat_train = model.predict(x_bs_train)
            y_hat_test = model.predict(x_bs_test)

            bootstrap_mse[0, j] = calculate_mse(y_bs_test, y_hat_test)
            bootstrap_mse[1, j] = calculate_mse(y_bs_train, y_hat_train)

        results[0, i] = np.mean(bootstrap_mse[0, :])
        results[1, i] = np.mean(bootstrap_mse[1, :])

        # Crossvalidation
        k_fold_dataset = KFold(n_splits=cv_k_folds, shuffle=True)

        cv_data = np.copy(data)
        x_cv = cv_data[:, 0 : -1]
        y_cv = cv_data[:, -1]

        cv_mse = np.zeros((2, cv_k_folds))

        index = 0
        for train_i, test_i in k_fold_dataset.split(cv_data):
            x_cv_train, x_cv_test, y_cv_train, y_cv_test = x_cv[train_i], x_cv[test_i], y_cv[train_i], y_cv[test_i]

            model = Ridge(alpha = current_lambda)

            model.fit(x_cv_train, y_cv_train)

            y_hat_train = model.predict(x_cv_train)
            y_hat_test = model.predict(x_cv_test)

            cv_mse[0, index] = calculate_mse(y_cv_test, y_hat_test)
            cv_mse[1, index] = calculate_mse(y_cv_train, y_hat_train)

            index += 1

        results[2, i] = np.mean(cv_mse[0, :])
        results[3, i] = np.mean(cv_mse[1, :])

    plt.figure(figsize=(6,6))
    plt.plot(lambda_paramters_to_test, results[0, :], label="Bootstrap test")
    plt.plot(lambda_paramters_to_test, results[1, :], label="Bootstrap train")
    plt.plot(lambda_paramters_to_test, results[2, :], label="Cross-validation test")
    plt.plot(lambda_paramters_to_test, results[3, :], label="Cross-validation train")
    plt.xlabel(r'$\lambda$')
    plt.ylabel("MSE")
    plt.xticks(np.linspace(0, 10, 25))
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("task_1_d.pdf", format="pdf")

    lambda_parameter = 0.91
    model = Ridge(alpha = lambda_parameter)
    X_train, Y_train, X_test, Y_test = split_to_random_train_test(data)

    model.fit(X_train, Y_train)

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    mse_train = calculate_mse(Y_train, y_hat_train)
    mse_test = calculate_mse(Y_test, y_hat_test)

    print("Ridge Results:")
    print(f"MSE train: {mse_train}")
    print(f"MSE test: {mse_test}")

    # Task 1 e)
    X_train, Y_train, X_test, Y_test = split_to_random_train_test(data)

    spline_orders = np.array(list(range(2, 7)))

    mse_train = np.zeros(spline_orders.shape[0])
    mse_test = np.zeros(spline_orders.shape[0])

    for i in range(spline_orders.shape[0]):

        current_spline_order = spline_orders[i]

        gam_model = GAM(s(0, spline_order=current_spline_order) + s(1, spline_order=current_spline_order) + s(2, spline_order=current_spline_order) + s(3, spline_order=current_spline_order) + s(4, spline_order=current_spline_order) + s(5, spline_order=current_spline_order) + s(6, spline_order=current_spline_order) + s(7, spline_order=current_spline_order))
        gam_model.fit(X_train, Y_train)

        y_hat_train = gam_model.predict(X_train)
        y_hat_test = gam_model.predict(X_test)

        mse_train[i] = calculate_mse(Y_train, y_hat_train)
        mse_test[i] = calculate_mse(Y_test, y_hat_test)
    
    plt.figure()
    plt.plot(spline_orders, mse_train, label="Train")
    plt.plot(spline_orders, mse_test, label="Test")
    plt.xticks(spline_orders)
    plt.xlabel("Number of splines per feature")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("task_1_e.pdf", format="pdf")
    
    selected_order = 7
    gam_model = GAM(s(0, spline_order=selected_order) + s(1, spline_order=selected_order) + s(2, spline_order=selected_order) + s(3, spline_order=selected_order) + s(4, spline_order=selected_order) + s(5, spline_order=selected_order) + s(6, spline_order=selected_order) + s(7, spline_order=selected_order))
    gam_model.fit(X_train, Y_train)

    y_hat_train = gam_model.predict(X_train)
    y_hat_test = gam_model.predict(X_test)

    mse_train = calculate_mse(Y_train, y_hat_train)
    mse_test = calculate_mse(Y_test, y_hat_test)

    print("GAM results")
    print(f"Train MSE: {mse_train}")
    print(f"Test MSE : {mse_test}")

    # Task 1 f)

    X_train, Y_train, X_test, Y_test = split_to_random_train_test(data)

    model_tree = DecisionTreeRegressor(random_state=0)
    model_tree.fit(X_train, Y_train)

    y_hat_train = model_tree.predict(X_train)
    y_hat_test = model_tree.predict(X_test)

    alphas = model_tree.cost_complexity_pruning_path(X_train, Y_train)["ccp_alphas"]

    mse_train = np.zeros(len(alphas))
    mse_test = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas): 
        tree = DecisionTreeRegressor(ccp_alpha=alpha)

        tree.fit(X_train, Y_train)

        y_hat_train = tree.predict(X_train)
        y_hat_test = tree.predict(X_test)

        mse_train[i] = calculate_mse(Y_train, y_hat_train)
        mse_test[i] = calculate_mse(Y_test, y_hat_test)

    plt.figure()
    plt.plot(alphas, mse_train, label="Train")
    plt.plot(alphas, mse_test, label="Test")

    plt.xlabel(r'$\alpha$')
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("task_1_f.pdf", format="pdf")

    model_tree = DecisionTreeRegressor(ccp_alpha=0.038, random_state=0)
    model_tree.fit(X_train, Y_train)

    plt.figure()
    plot_tree(model_tree)
    plt.savefig("task_1_f_tree.pdf", format="pdf")

    y_hat_train = model_tree.predict(X_train)
    y_hat_test = model_tree.predict(X_test)

    mse_train = calculate_mse(Y_train, y_hat_train)
    mse_test = calculate_mse(Y_test, y_hat_test)

    print("Regression tree results")
    print(f"Train MSE: {mse_train}")
    print(f"Test MSE : {mse_test}")

    plt.show()