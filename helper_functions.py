import io


def df_metadata(df) -> tuple:
    """
    return a tuple of (size, shape, info)
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    return (df.size, df.shape, buffer.getvalue())


def df_print_metadata(df) -> None:
    """
    print metadata of dataframe
    """
    size, shape, info = df_metadata(df)
    print("DATAFRAME METADATA")
    print(f"Size: {size}")
    print()
    print(f"Shape: {shape[0]} x {shape[1]}")
    print()
    print("Info:")
    print(info, end="")


def df_peek(df) -> tuple:
    """
    return head and tail of df
    """
    return df.head().append(df.tail())


def df_print_summary(df):
    """
    columns is a sequence of columns whose IQR and range will be calculated
    """
    print("SUMMARY")
    description = df.describe()
    print("Description:")
    print(description)
    print()
    print("IQR:")
    for col in df.columns:
        print(
            f"\t{col}: {description.loc['75%', col] - description.loc['25%', col]}"
        )
    print()
    print("Range:")
    for col in df.columns:
        print(f"\t{col}: {df[col].max() - df[col].min()}")


def series_is_whole_nums(series):
    try:
        return (series % 1 == 0).all()
    except TypeError:
        return False


def df_float_to_int(df):
    to_coerce = {}
    for col in df.columns:
        if series_is_whole_nums(df[col]):
            to_coerce[col] = int
    return df.astype(to_coerce)


def df_print_missing_vals(df):
    # any missing values?
    print("\nMissing Values:\n")
    null_counts = df.isnull().sum()
    if len(null_counts[null_counts > 0]) == 0:
        print("No missing values")
    else:
        print(null_counts[null_counts > 0])


def df_percent_missing_vals(df):
    return (df.isnull().sum() / df.shape[0]) * 100


def df_drop_cols(df, to_drop):
    return df.drop(columns=to_drop)


def df_remove_outliers(df, cols, zscore_limit):
    df_clean = None
    for col in cols:
        df_clean = df[
            (np.abs(stats.zscore(df[cols])) < zscore_limit).all(axis=1)
        ]
    return df_clean


# define function that will combine variables; one argument can be an operator function. like operator.add


def df_print_r_and_p_values(X, y):
    r_and_p_values = {col: stats.pearsonr(X[col], y) for col in X.columns}
    print("PEARSON'S R")
    for k, v in r_and_p_values.items():
        col = k
        r, p = v
        print(f"{col}:")
        print(
            f"\tPearson's R is {r:.2f} with a significance p-value of {p: .3}\n"
        )


def linreg_fit_and_predict(x_train, y_train, x_test, y_test):
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    y_label = y_train.columns[0]
    y_intercept = lm.intercept_[0]
    m = lm.coef_[0][0]
    x_label = x_train.columns[0]
    print(f"Univariate: {y_label} = {y_intercept:.2f} + {m:.3}*{x_label}")
    print()

    preds_train = lm.predict(x_train)

    # run test data through model
    preds_test = lm.predict(x_test)

    return lm, preds_train, preds_test


def evaluate_model_train(x, y, preds):
    y_label = y.columns[0]
    x_label = x.columns[0]

    print("Model Evaluation on TRAIN Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained by {x_label}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(x, y)
    print(f"\tTrain: {p_vals[0]:.3}")
    print()


def evaluate_model_test(x, y, preds):
    y_label = y.columns[0]
    x_label = x.columns[0]

    print("Model Evaluation on TEST Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained by {x_label}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(x, y)
    print(f"\tTest: {p_vals[0]:.3}")
    print()


def plot_residuals(y_test, preds_test):
    y_label = y_test.columns[0]
    plt.scatter(preds_test, preds_test - y_test, c="g", s=20)
    plt.hlines(y=0, xmin=preds_test.min(), xmax=preds_test.max())
    plt.title("Residual plot")
    plt.ylabel("Residuals")
    plt.xlabel(y_label)
    plt.show()


def linreg_model(x_train, y_train, x_test, y_test):
    lm, preds_train, preds_test = linreg_fit_and_predict(
        x_train, y_train, x_test, y_test
    )

    evaluate_model_train(x_train, y_train, preds_train)
    evaluate_model_test(x_test, y_test, preds_test)

    plot_residuals(y_test, preds_test)
    

def evaluate_multi_model_train(X, y, preds):
    y_label = y.columns[0]
    X_labels = X.columns

    print("Model Evaluation on TRAIN Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained by {X_labels}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(X, y)
    print(f"\tTrain: {p_vals[0]:.3}")
    print()
    
def evaluate_multi_model_test(X, y, preds):
    y_label = y.columns[0]
    X_labels = X.columns

    print("Model Evaluation on TEST Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained by {X_labels}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(X, y)
    print(f"\tTest: {p_vals[0]:.3}")
    print()
    
def multi_linreg_fit_and_evaluate(X_train, y_train, X_test, y_test):
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    y_label = y_train.columns[0]
    y_intercept = lm.intercept_[0]
    print("Multivariate:")
    print(f"{y_label} = ")
    print(f"{y_intercept:.3f}")
    for i, col in enumerate(X_train.columns):
        coefficient = lm.coef_[0][i]
        print(f"+ {coefficient:.3}*{col}")

    preds_train = lm.predict(X_train)
    evaluate_multi_model_train(X_train, y_train, preds_train)
    
    preds_test = lm.predict(X_test)
    evaluate_model_test(X_test, y_test, preds_test)
    
    plot_residuals(y_test, preds_test)


def normalize_cols(df_train, df_test, cols):
    df_train_norm = pd.DataFrame()
    for col in cols:
        minimum = df_train[col].min()
        maximum = df_train[col].max()
        df_train_norm[f"{col}_norm"] = (df_train[col] - minimum) / (maximum - minimum)
    
    df_test_norm = pd.DataFrame()
    for col in cols:
        minimum = df_train[col].min()  # use the min and max from the train set
        maximum = df_train[col].max()
        df_test_norm[f"{col}_norm"] = (df_test[col] - minimum) / (maximum - minimum)
    return df_train_norm, df_test_norm