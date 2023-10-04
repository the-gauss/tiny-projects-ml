from sklearn import preprocessing


# Decorator to convert a function into sklearn FunctionTransformer.
def sklearn_transformer(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    transformer = preprocessing.FunctionTransformer(wrapper, validate=False)
    return transformer
