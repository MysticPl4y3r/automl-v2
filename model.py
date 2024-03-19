import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from PreProcessor import PreProcessor
from feat_imp import Feature_Selection
from functools import partial
import copy
import joblib
from gen_alg_pre import acc_score, mse_score
from model_selection import ModelSelection,run_model
from gen_alg import genetic

accuracy={}
mse={}

def preprocess(data,target):
    pp=PreProcessor()
    data=pp.RemoveIrrelevantColumn(data)
    data=pp.HandlingMissingData(data)
    if data[target].dtype!='object':
        data=pp.encoding(data)
    else:
        for i in data.columns:
            if i!=target:
                data.loc[:,i:i]=pp.encoding(data.loc[:,i:i])
    
    for i in data.columns:
        if data[i].dtype=="object":
            if (is_float(data[i][0])==True):
                data[i]=data[i].astype(float)
            elif (is_int(data[i][0])==True):
                data[i]=data[i].astype(int)
            else:
                continue
    return data
    
def train(target,df):
    mod_type=str
    org=copy.deepcopy(df)
    data=preprocess(df,target)
    print(org[target].dtype)
    print("No. of columns =",data.shape[1])
    if org[target].dtype=='object':
        print("Classify")
        mod_type="Class"
        accuracy = acc_score(data,target)
        print(accuracy)
    else:
        print("Regressor")
        mod_type="Reg"
        accuracy = mse_score(data,target)
        print(accuracy)
    x=data.drop(columns=target)
    y=data[target]
    before_acc=max(accuracy.values, key=lambda x: x[1])[1]
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    before_feat=list(X_train.columns)
    if (len(X_train.columns)>10):
        print("Using Genetic Algorithm")
        prepared=pd.concat([X_train,Y_train],axis=1)
        GG=genetic(Data=prepared, targer_var=target)
        population, generations = GG.run_evolution(
                                popluation_size=10,
                                genome_length=len(data.columns)-1,
                                # target_var='charges',
                                generation_limit=30
                            )

        chromo_df=GG.best_feature(population,prepared.columns.tolist())
        #chromo_df=best_feature(final_result,prepared.columns.to_list())
        print("Features selected using Genetic Algorithm:",chromo_df)
        X_train=X_train[chromo_df]
        X_test=X_test[chromo_df]
    else:
        print("Using Statistics Analysis")
        X_train,X_test, Y_train, Y_test=feature_select(org,data,target)
        print("Features Selected using Statistics:",X_train.columns)
        chromo_df=X_train.columns
        print("Best Features",chromo_df)
        X_train=X_train[chromo_df]
        X_test=X_test[chromo_df]
    print("X_train features :",X_train.columns)

    if mod_type=="Class":
        le=LabelEncoder()
        Y_train=le.fit_transform(Y_train)
        Y_test=le.fit_transform(Y_test)

    model_selection = ModelSelection(X_train, X_test, Y_train, Y_test)
    task = model_selection.detect_task()
    if task == 'classification':
        model_name=model_selection.choose_best_model()
    elif task == 'regression':
        model_name=model_selection.choose_best_model_regression()

    best_model=run_model(model_name)
    best_model.fit(X_train,Y_train)
    '''if org[target].dtype=='object':
        print("Classify")
        Baccuracy = class_model(X_train,X_test,Y_train,Y_test)
        print(Baccuracy)
    else:
        print("Regressor")
        Baccuracy = reg_model(X_train,X_test,Y_train,Y_test)
        print(Baccuracy)'''

    #best_model=max(Baccuracy.values, key=lambda x: x[1])[2]
    print(X_test.shape)
    print(Y_test.shape)
    y_pred=best_model.predict(X_test)
    

    if org[target].dtype=="object":
        select_feat_scr=accuracy_score(Y_test,y_pred)
        Y_train=le.inverse_transform(Y_train)
        y_pred=le.inverse_transform(y_pred)
        Y_test=le.inverse_transform(Y_test)
        labels=le.classes_
    else:
        select_feat_scr=r2_score(Y_test,y_pred)
    #Save Model
    print(select_feat_scr)
    joblib.dump(best_model,"./model/best_model.sav")
    if mod_type=="Class":
        return before_feat,before_acc,chromo_df,select_feat_scr,X_train,X_test[chromo_df],Y_train,Y_test,y_pred,labels
    else:
        return before_feat,before_acc,chromo_df,select_feat_scr,X_train,X_test[chromo_df],Y_train,Y_test,y_pred,None


def feature_select(org,data,target):
    fs=Feature_Selection()
    data,x,y=fs.constant_variance(data,0.2,target)
    if org[target].dtype=='object':
        mode='classification'
    else:
        mode='regression'
    x=fs.k_select_best(x,y,2,mode)
    x_train,x_test,y_train,y_test=fs.corr_drop(x,y)
    #print("SFS X_Train:",x_train)
    return x_train,x_test,y_train,y_test

def is_float(value):
    try:
        float_value = float(value)
        if isinstance(float_value, float):
            return True
        else:
            return False
    except ValueError:
        return False
    
def is_int(value):
    try:
        int_value = int(value)
        if isinstance(int_value, int):
            return True
        else:
            return False
    except ValueError:
        return False




'''if os.path.isfile("./model/preprocessed.csv") and os.path.isfile("./model/target.txt"):
    data=pd.read_csv("./model/preprocessed.csv")
    with open("./model/target.txt", "r") as file:
        target_val = file.read()

    TARGET_VAR = target_val.strip()'''


#data=pd.read_csv("./data/heart.csv")
#train("target",data)