def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred.tolist()