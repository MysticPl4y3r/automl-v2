import joblib

def predict(data):
    model=joblib.load("./model/best_model.sav")
    return model.predict(data)

def predict_prob(x_test):
    model=joblib.load("./model/best_model.sav")
    return model.predict_proba(x_test)