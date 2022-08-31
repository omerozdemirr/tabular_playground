import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv('train.csv',  index_col="id")
df_test =pd.read_csv('test.csv', index_col="id")


#splitting features and labels

X =df_train.drop(columns=["product_code","failure"])
X_test=df_test.drop(columns=['product_code'])
y = df_train['failure']

#train test split
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2)

#seperating categorical and numerical features 
#x_trrain
X_train_categorical =X_train.select_dtypes(include='object')
X_train_numerical =X_train.select_dtypes(exclude='object')

#X validation 
X_val_categorical =X_val.select_dtypes(include ='object')
X_val_numerical =X_val.select_dtypes(exclude='object')

#X_Test

X_test_categorical = X_test.select_dtypes(include='object')
X_test_numerical = X_test.select_dtypes(exclude='object')

#create emty data frames 

encoded_X_train =pd.DataFrame(index =X_train_categorical.index)
encoded_X_val =pd.DataFrame(index =X_val_categorical.index)
encoded_X_test =pd.DataFrame(index =X_test_categorical.index)

#encoding based on material number 


encoded_X_train["attribute_0"] = X_train_categorical["attribute_0"].str.replace("material_"," ").astype(int)
encoded_X_train["attribute_1"] = X_train_categorical["attribute_1"].str.replace("material_","").astype(int)
## X val
encoded_X_val["attribute_0"] = X_val_categorical["attribute_0"].str.replace("material_","").astype(int)
encoded_X_val["attribute_1"] = X_val_categorical["attribute_1"].str.replace("material_","").astype(int)
## X test
encoded_X_test["attribute_0"] = X_test_categorical["attribute_0"].str.replace("material_","").astype(int)
encoded_X_test["attribute_1"] = X_test_categorical["attribute_1"].str.replace("material_","").astype(int)



#Impute using KNN Imputer

imputer =KNNImputer()
imputed_X_train =pd.DataFrame(imputer.fit_transform(X_train_numerical),columns =X_train_numerical.columns,index =X_train_numerical.index)


imputed_X_val =pd.DataFrame(imputer.transform(X_val_numerical),columns =X_val_numerical.columns,index =X_val_numerical.index)

imputed_X_test =pd.DataFrame(imputer.transform(X_test_numerical),columns =X_test_numerical.columns,index =X_test_numerical.index)

#handling outliers

lof = LocalOutlierFactor(n_neighbors=5)
yhat =pd.DataFrame(lof.fit_predict(imputed_X_train),
                   columns=["outliers_d"],index =X_train_numerical.index)
outliers_index = yhat[yhat["outliers_d"]==-1].index
print(outliers_index)


#dropping outliers 
imputed_X_train.drop(outliers_index,inplace =True)
encoded_X_train.drop(outliers_index,inplace=True)
y_train.drop(outliers_index,inplace =True)


#scalling 
scaler =StandardScaler()

scaled_X_train =pd.DataFrame(scaler.fit_transform(imputed_X_train),columns =imputed_X_train.columns,index = imputed_X_train.index)
scaled_X_val =pd.DataFrame(scaler.transform(imputed_X_val),columns =imputed_X_val.columns,index=imputed_X_val.index)
scaled_X_test =pd.DataFrame(scaler.transform(imputed_X_test),columns =imputed_X_test.columns,index=imputed_X_test.index)


#combinin Categorical and numerical features

#X_train 
processed_X_train =pd.concat([encoded_X_train,scaled_X_train],axis =1,ignore_index=1)
processed_X_train.columns = list(encoded_X_train.columns) + list(scaled_X_train.columns)

#x val

processed_X_val = pd.concat([encoded_X_val,scaled_X_val], axis =1, ignore_index=1)
processed_X_val.columns = list(encoded_X_val.columns) +list(scaled_X_val.columns)

#xtest

processed_X_test = pd.concat([encoded_X_test,scaled_X_test],axis =1, ignore_index=1)
processed_X_test.columns=list(encoded_X_test.columns) + list(scaled_X_test.columns)


#combine X train and x val for cross validation 


X_train_cv = pd.concat([processed_X_train,processed_X_val],axis =0)
y_train_cv=pd.concat([y_train,y_val],axis =0)


p_grid ={}

gnb =GaussianNB()
gnb_cv =GridSearchCV(gnb,param_grid=p_grid,cv =2)
model_gnb=gnb_cv.fit(X_train_cv,y_train_cv)
'''
decision_tree_model =DecisionTreeClassifier()
param_grid = {"criterion": ["gini","entropy"],
               "max_depth": np.arange(3,15),
               "min_samples_split": np.arange(2,10,2),
               "max_leaf_nodes": np.arange(2,10,2)}

model_cv =GridSearchCV(decision_tree_model,param_grid,cv =5,scoring ="roc_auc")
model_cv.fit(X_train_cv,y_train_cv)

model_cv.best_estimator_.get_params()


model_train =DecisionTreeClassifier(criterion = 'gini', max_depth=2,max_leaf_nodes=6,min_samples_leaf=2,min_samples_split=4)
model_train.fit(processed_X_train,y_train)
'''
pro_predict =pd.DataFrame(model_gnb.predict_proba(processed_X_val)[:,1],columns =['probability_failure'],index =processed_X_val.index)

print(pro_predict["probability_failure"].max())
for i  in range(1,10,1):
    decision_criterion =i/10
    pro_predict.loc[pro_predict['probability_failure']> decision_criterion,'prediction'] =1 
    pro_predict.loc[pro_predict['probability_failure']<= decision_criterion,'prediction'] =0 
    score =roc_auc_score(y_val,pro_predict['prediction'])          


#model deployment 
output = pd.DataFrame(model_gnb.predict_proba(processed_X_test)[:,1],columns =['probability_failure'],index =processed_X_test.index)
output.loc[output["probability_failure"]>0.2,"failure"] =1
output.loc[output["probability_failure"]<=0.2,"failure"] =0
output.drop(columns=["probability_failure"],inplace =True)
output["failure"] =output["failure"].astype(int)

print(output.head())

output.to_csv("submission.csv")
