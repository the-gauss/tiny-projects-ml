from sklearn import metrics, preprocessing, model_selection, linear_model
import numpy as np
import matplotlib.pyplot as plt
from scratch_models import linear_regression


def learning_curve(model, X, y):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_val_pred = model.predict(X_val)
        train_errors.append(metrics.mean_squared_error(y_train[:m], y_train_pred))
        val_errors.append(metrics.mean_squared_error(y_val, y_val_pred))

    plt.plot(np.sqrt(train_errors), 'r-', label='train')
    plt.plot(np.sqrt(val_errors), 'b--', label='validation')
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")


def early_stopping_SGD(X, y, n_epochs=500, warm_start=True, max_iter=1, **sgd_args):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2)
    train_errors, val_errors, models = [], [], []

    for epoch in range(n_epochs):
        sgd = linear_model.SGDRegressor(max_iter=max_iter, warm_start=warm_start, **sgd_args)

        sgd.fit(X, y)
        y_train_predict = sgd.predict(X_train)
        y_val_predict = sgd.predict(X_val)
        train_errors.append(metrics.mean_squared_error(y_train, y_train_predict))
        val_errors.append(metrics.mean_squared_error(y_val, y_val_predict))
        models.append(sgd)

    best_epoch = np.argmin(val_errors)
    best_val_rmse = np.sqrt(val_errors[best_epoch])
    best_model = models[best_epoch]

    plt.annotate('Best model',
                 xy=(best_epoch, best_val_rmse),
                 xytext=(best_epoch, best_val_rmse + 1),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=16,
                 )

    best_val_rmse -= 0.03  # just to make the graph look better
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.show()

    return best_model


def early_stopping_BGD(X, y, n_epochs, **bgd_args):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2)
    train_errors, val_errors, models = [], [], []

    for epoch in range(n_epochs):
        bgd = linear_regression.GradientDescent(gd_type='batch', n_epochs=n_epochs, **bgd_args)

        bgd.fit(X, y)
        y_train_predict = bgd.predict(X_train)
        y_val_predict = bgd.predict(X_val)
        train_errors.append(metrics.mean_squared_error(y_train, y_train_predict))
        val_errors.append(metrics.mean_squared_error(y_val, y_val_predict))
        models.append(bgd)

    best_epoch = np.argmin(val_errors)
    best_val_rmse = np.sqrt(val_errors[best_epoch])
    best_model = models[best_epoch]

    best_val_rmse -= 0.03  # just to make the graph look better
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.show()

    return best_model