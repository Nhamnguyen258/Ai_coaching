from numpy import ndarray
from sklearn.linear_model import LassoLarsCV
from base import BaseLR
from sklearn.model_selection import train_test_split
from typing import List

class LassoLr(BaseLR):
    
    def __init__(self, path_to_model:str) -> None:
        self.path_to_model = path_to_model
    
    def train(self, x_train: ndarray, y_train: ndarray, x_dev: ndarray = None, y_dev: ndarray = None):
        if not x_dev or not y_dev:
            x_train,x_dev,y_train,y_dev = train_test_split(x_train,y_train,test_size=0.2)
        model  = LassoLarsCV(fit_intercept=True,cv=5)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_dev)
        eval = self.evaluate(y_true=y_pred,y_pred=y_pred)
        print("Eval: ",eval)
        self.save(model=model,path=self.path_to_model)
        return model


if __name__=="__main__" :
    import sys
    import os
    sys.path.append(os.getcwd())
    from ml.logistic_regression.dataset import X_digits,y_digits
    clf = LassoLr(path_to_model="ml/save/logictic.pkl")

    clf.train(X_digits,y_digits)
