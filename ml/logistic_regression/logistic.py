from numpy import ndarray
from dataset import X_digits,y_digits
from typing import List
from base import BaseLR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


class LogisticReg(BaseLR):

    def __init__(self,path2model:str) -> None:
        self.path2model = path2model
    
    def train(self, x_train: ndarray, y_train: ndarray):
        X_train,X_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2)

        self.model = LogisticRegressionCV(fit_intercept=True, penalty= "l2",multi_class="multinomial",cv=5,solver='lbfgs')
        self.model.fit(X=X_train,y=y_train)
        y_pred = self.model.predict(X_test)
        self.eval = self.evaluate(y_test,y_pred)
        self.save(model=self.model, path=self.path2model)
        print()
        return self.model

if __name__=="__main__" :
    clf = LogisticReg(path2model="ml/save/logictic.pkl")
    clf.train(X_digits,y_digits)
