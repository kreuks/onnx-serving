import onnx

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common import data_types
import xgboost as xgb


def __load_data():
    digits = load_digits()
    X, y = digits.data, digits.target  # Our train data shape is (x, 64) where x is total samples
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def __create_model():
    booster = xgb.XGBClassifier(max_depth=3,
                                booster='dart',
                                eta=0.3,
                                silent=1,
                                n_estimators=100,
                                num_class=10)
    return booster


def __train_model(booster, X_train, y_train):
    booster.fit(X_train, y_train)
    return booster


def __convert_xgboost_model_and_save(booster):
    initial_type = [('float_input', data_types.FloatTensorType([1, 64]))]
    booster_onnx = convert_xgboost(booster, initial_types=initial_type, doc_string='Input size is (x, 64)')
    onnx.save(booster_onnx, 'xgboost.onnx')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = __load_data()
    booster = __train_model(__create_model(), X_train, y_train)
    __convert_xgboost_model_and_save(booster)
