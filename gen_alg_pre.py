import numpy as np
import pandas as pd
from random import randint 
import warnings

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier,XGBRegressor

def split(df,label):
    x=df.drop(columns=label)
    #print(x)
    X_tr, X_te, Y_tr, Y_te = train_test_split(x, df[label], test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te

def acc_score(df, label):
    le=LabelEncoder()
    classifiers = ['Logistic',  'RandomForest', 
               'AdaBoost',  'DecisionTree', 
               'KNeighbors','GradientBoosting','XGBClassifier']

    models = [LogisticRegression(max_iter = 1000),
              RandomForestClassifier(n_estimators=200, random_state=0),
              AdaBoostClassifier(random_state = 0),
              DecisionTreeClassifier(random_state=0),
              KNeighborsClassifier(),
              GradientBoostingClassifier(random_state=0),
              XGBClassifier(n_estimators=10,max_depth=3,learning_rate=0.2)]

    j = 0
    acc = []
    #print(df)
    X_train, X_test, Y_train, Y_test = split(df, label)
    #print(Y_train)
    Y_train=le.fit_transform(Y_train)
    for i in models:
        model = i
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        predictions=le.inverse_transform(predictions)
        #print("Inverse Transform",predictions)
        #print("Y_test :",Y_test)
        acc.append(accuracy_score(Y_test, predictions))
        j = j + 1
    #print(acc)
    
    if len(acc) != len(classifiers):
        raise ValueError("Number of accuracy scores does not match the number of classifiers.")
    Score = pd.DataFrame({"Classifier": classifiers, "Accuracy":acc,"Model":models})
    #Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False, inplace=True)
    Score.reset_index(drop=True, inplace=True)
    return Score

def mse_score(df, label):
    regressors = ['LinearRegression', 'RandomForestRegressor', 
                  'AdaBoostRegressor', 'DecisionTreeRegressor', 
                  'KNeighborsRegressor', 'GradientBoostingRegressor','XGBRegressor']
    models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=200, random_state=0),
        AdaBoostRegressor(random_state=0),
        DecisionTreeRegressor(random_state=0),
        KNeighborsRegressor(),
        GradientBoostingRegressor(random_state=0),
        XGBRegressor(n_estimators=10,max_depth=3,learning_rate=0.2)
    ]
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = split(df, label)
    
    # Initialize lists to store MSE and R-squared scores
    mse_scores = []
    r2_scores = []
    

    for model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        #mse_scores.append(mean_squared_error(Y_test, predictions))
        r2_scores.append(r2_score(Y_test, predictions))
    
    # Create DataFrame to store results
    Score = pd.DataFrame({"Regressor": regressors,  "R2 Score": r2_scores, "Model":models})
    
    # Sort by MSE in ascending order
    Score.sort_values(by="R2 Score", ascending=True, inplace=True)
    
    # Reset index
    Score.reset_index(drop=True, inplace=True)
    
    return Score

def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=bool)     
        chromosome[:int(0.3*n_feat)]=False             
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population,logmodel,X_train,X_test,Y_train,Y_test):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],Y_train)         
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)                     
    return list(scores[inds][::-1]), list(population[inds,:][::-1]) 

def fitness_score_reg(population,logmodel,X_train,X_test,Y_train,Y_test):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],Y_train)         
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(r2_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)                                    
    return list(scores[inds][::-1]), list(population[inds,:][::-1]) 

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0,len(pop_after_sel),2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen

def mutation(pop_after_cross,mutation_rate,n_feat):   
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = [] 
        for i in range(0,mutation_range):
            pos = randint(0,n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]  
        pop_next_gen.append(chromo)
    return pop_next_gen

def generations(df,label,model,size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, Y_train, Y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    try:
        for i in range(n_gen):
            if df[label].dtype=='object':
                scores, pop_after_fit = fitness_score(population_nextgen,model,X_train,X_test,Y_train,Y_test)
                print('Best score in generation',i+1,':',scores[:1])  #2
                pop_after_sel = selection(pop_after_fit,n_parents)
                pop_after_cross = crossover(pop_after_sel)
                population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
                best_chromo.append(pop_after_fit[0])
                best_score.append(scores[0])
            else :
                scores, pop_after_fit = fitness_score_reg(population_nextgen,model,X_train,X_test,Y_train,Y_test)
                print('Best score in generation',i+1,':',scores[:1])  #2
                pop_after_sel = selection(pop_after_fit,n_parents)
                pop_after_cross = crossover(pop_after_sel)
                population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
                best_chromo.append(pop_after_fit[0])
                best_score.append(scores[0])
    except:
        print("End of generation")
    return best_chromo,best_score