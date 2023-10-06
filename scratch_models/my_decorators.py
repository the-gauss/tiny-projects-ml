from sklearn import preprocessing
import functools


# Decorator to convert a function into sklearn FunctionTransformer.
def sklearn_transformer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    transformer = preprocessing.FunctionTransformer(wrapper, validate=False)
    return transformer
