from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
import statistics as stat
import time


class Feature_Selection:
    def __init__(self):
        self.start_time=time.time()
        
        
    def constant_variance(self,data,threshold,dependent_variable):
        start_time=time.time()
        #print("Finding Constant Variance...")
        x =data.drop(dependent_variable, axis=1)
        y=data[dependent_variable]
        
        constant_columns=[]
        
        var_thres=VarianceThreshold(threshold)
        var_thres.fit(x)
        
        constant_columns = [column for column in x.columns
                        if column not in x.columns[var_thres.get_support()]]
        
        # Printing the variance of the column
        
        for column in x:
            print(stat.variance(x[column]))
        print(f"No of Constant Columns are {len(constant_columns)}")
        
        print(f'{constant_columns} are being removed')
        
        
        # Dropping the constant Columns
        if constant_columns:
            x=x.drop(columns=constant_columns)
            print("Constant columns are successfully dropped.")
        else:
            print("No constant columns are found.")
            
        
        end_time=time.time()
        time_elp=end_time-start_time
        #print(f"Time Elapsed for dropping constant variance: {time_elp} seconds")
        return data,x,y
    
    def data_splitting(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        return x_train,x_test,y_train,y_test
    
    def k_select_best(self,x,y,threshold,mode):
        start_time=time.time()

        #print("Selecting Best Features...\n")
        if mode=='classification':
            ordered_ranking_features =SelectKBest(score_func=f_classif, k='all')
            ordered_feature=ordered_ranking_features.fit(x,y) 
        elif mode=='regression':
            ordered_ranking_features =SelectKBest(score_func=f_regression, k='all')
            ordered_feature=ordered_ranking_features.fit(x,y)
        features_ranked=pd.DataFrame(ordered_feature.scores_,columns=['Score'])
        dfcolumns=pd.DataFrame(x.columns)
        features_rank=pd.concat([dfcolumns,features_ranked],axis=1)
        features_rank.columns=['Features','Score']
        features_rank_mod=features_rank[features_rank['Score']>=threshold]
        feature_names = features_rank_mod['Features'].values
        #print("Features in rank order:")
        #print(features_rank.sort_values(by='Score',ascending=False))
       # print("\n\nSelected Features:")
        #print(feature_names)
        x=x[feature_names]
        
        end_time=time.time()
        
        #print("k_select_best is completed.")
        #plt.bar(x=features_rank['Features'],height=features_rank['Score'])
        #plt.show()
        #time_elp=end_time-start_time
        #print(f'\n\n Time taken to complete the k_select_best is:{time_elp}')
        return x
     
    def corr_drop(self,x,y):
        start_time=time.time()
        x_train,x_test,y_train,y_test=Feature_Selection.data_splitting(self,x,y)
        
        #plt.figure(figsize=(12,10))
        #cor = x_train.corr()
        #sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        #plt.show()
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = x_train.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i): # lower triangle of an matrix
                if (corr_matrix.iloc[i, j]) > 0.95: # we are interested in coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        #print(f'Total {len(col_corr)} are being removed and the Columns are: {col_corr}')
        #print(x_train)
        x_train=x_train.drop(col_corr,axis=1)
        x_test=x_test.drop(col_corr,axis=1)
        #print(x_train.columns)
        end_time=time.time()
        print(f"Time Elapsed for Correlation Drop: {end_time-start_time} seconds")
        
        

        return x_train,x_test,y_train,y_test