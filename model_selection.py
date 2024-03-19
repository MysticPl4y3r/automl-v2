from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import time
import concurrent.futures

class ModelSelection:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def detect_task(self):
        if len(set(self.y_train)) > 10:
            return 'regression'
        else:
            return 'classification'

    def random_forest(self):
        start_time = time.time()  
        model = RandomForestClassifier(n_estimators=200, random_state=0)
        model.fit(self.x_train, self.y_train)
        end_time = time.time()    
        time_elp = end_time - start_time
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Execution time for Random Forest is {time_elp} seconds')
        print(f'Accuracy of Random Forest is {accuracy}')
        return accuracy, time_elp

    def xgb(self):
        start_time = time.time()  
        model = XGBClassifier(n_estimators=10,max_depth=3,learning_rate=0.2)
        model.fit(self.x_train, self.y_train)
        end_time = time.time()    
        time_elp = end_time - start_time
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Execution time for XGB is {time_elp} seconds')
        print(f'Accuracy of XGB is {accuracy}')
        return accuracy, time_elp

    def kneighbors(self):
        start_time = time.time()  
        model = KNeighborsClassifier()
        model.fit(self.x_train, self.y_train)
        end_time = time.time()    
        time_elp = end_time - start_time
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Execution time for KNeighbors is {time_elp} seconds')
        print(f'Accuracy of KNeighbors is {accuracy}')
        return accuracy, time_elp

    def guissannb(self):
        start_time = time.time()  
        model = GaussianNB()
        model.fit(self.x_train, self.y_train)
        end_time = time.time()    
        time_elp = end_time - start_time
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Execution time for GaussianNB is {time_elp} seconds')
        print(f'Accuracy of GaussianNB is {accuracy}')
        return accuracy, time_elp

    def logistic_regression(self):
        start_time = time.time()  
        model = LogisticRegression(max_iter = 1000)
        model.fit(self.x_train, self.y_train)
        end_time = time.time()    
        time_elp = end_time - start_time
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Execution time for Logistic Regression is {time_elp} seconds')
        print(f'Accuracy of Logistic Regression is {accuracy}')
        return accuracy, time_elp

    def evaluate_model(self, model_fn):
        accuracy, time_elp = model_fn()
        return accuracy, time_elp

    def choose_best_model(self):
        models = [self.logistic_regression, self.random_forest, self.xgb, self.kneighbors, self.guissannb]
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.evaluate_model, model) for model in models]

            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        best_accuracy = max(results, key=lambda x: x[0])[0]
        best_model_index = results.index(max(results, key=lambda x: x[0]))

        model_names = ["GaussianNB","Logistic Regression", "KNeighbors", "XGBoost", "Random Forest"]
        best_model_name = model_names[best_model_index]

        print(f"The best model is {best_model_name} with an accuracy of {best_accuracy}")
        return best_model_name

    def evaluate_model_regression(self, model_fn):
        model = model_fn()
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse

    def choose_best_model_regression(self):
        models = [RandomForestRegressor, XGBRegressor, KNeighborsRegressor]
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.evaluate_model_regression, model) for model in models]

            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        best_mse = min(results)
        best_model_index = results.index(min(results))

        model_names = ["Random Forest Regression", "XGBoost Regression", "KNeighbors Regression"]
        best_model_name = model_names[best_model_index]

        print(f"The best model for regression is {best_model_name} with an MSE of {best_mse}")
        return best_model_name

def run_model(model_name):
    if model_name=='Random Forest':
        model = RandomForestClassifier(n_estimators=200, random_state=0)
    elif model_name=='XGBoost':
        model = XGBClassifier(n_estimators=10,max_depth=3,learning_rate=0.2)
    elif model_name=='KNeighbors':
        model = KNeighborsClassifier()
    elif model_name=='GaussianNB': 
        model = GaussianNB()
    elif model_name=='Logistic Regression':
        model = LogisticRegression(max_iter = 1000)
    elif model_name=='Random Forest Regression':
        model = RandomForestRegressor(n_estimators=200, random_state=0)
    elif model_name=='XGBoost Regression':
        model = XGBRegressor(n_estimators=10,max_depth=3,learning_rate=0.2)
    elif model_name=='KNeighbors Regression':
        model = KNeighborsRegressor()
    else:
        print('No model selected')
    return model


'''model_selection = ModelSelection(x_train, x_test, y_train, y_test)
task = model_selection.detect_task()
if task == 'classification':
    model_name=model_selection.choose_best_model()
elif task == 'regression':
    model_name=model_selection.choose_best_model_regression()

model=run_model(model_name)'''
