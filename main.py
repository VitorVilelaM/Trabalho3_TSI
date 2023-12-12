import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

## Database

dataBase = pd.read_csv('./Data.csv')  
dataBase = dataBase.dropna()

featuresTrain = dataBase[dataBase['Date'] < 20210000]
featuresValidation = dataBase[(dataBase['Date'] > 20210000) & ( dataBase['Date'] < 20220000)]
featuresTest = dataBase[dataBase['Date'] > 20220000]

train = featuresTrain.copy()
validation = featuresValidation.copy()
test = featuresTest.copy()

date = test['Date'].values

## Variaveis Globais

# Saidas Validaçao
random_forest = []
mlp = []
knn = []
xgboost = []
naive_bayes = []
linear_regression = []
out = []

# Saidas Teste
random_forest_test = []
mlp_test = []
knn_test = []
xgboost_test = []
naive_bayes_test = []
linear_regression_test = [] 
out_test = []

# Probabilidade Saidas Teste
random_forest_probabilidade = []
mlp_probabilidade = []
knn_probabilidade = []
xgboost_probabilidade = []
naive_bayes_probabilidade = []

# AUC de cada Modelo
auc_random_forest = [1]
auc_mlp = [1]
auc_knn = [1]
auc_xgboost = [1]
auc_naiveBayes = [1]

# MSE de cada modelo
mae_random_forest = [1]
mae_mlp = [1]
mae_knn = [1]
mae_xgboost = [1]
mae_linear_regression = [1]

## Random Forest

def rf_Regressor(num_days):

    rf_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    rf_train_output = rf_train_input.pop('R') # Target

    rf_validation_input = validation.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5', 'R'])  # Features

    rf_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    rf_test_output = rf_test_input.pop('R')

    limite = rf_test_input.shape[0]

    new_train_input = rf_train_input
    new_train_output = rf_train_output

    predicts = []

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=2)
    model.fit(new_train_input,new_train_output)    
    global random_forest
    random_forest = model.predict(rf_validation_input)
           
    while( limite > 0):
        limite = rf_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = rf_test_input.iloc[0:num_days]
        rf_test_input = rf_test_input.drop(rf_test_input.index[0:num_days])    

        new_target = rf_test_output.iloc[0:num_days]
        rf_test_output = rf_test_output.drop(rf_test_output.index[0:num_days])    
        
        if(limite > 0):
            
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    

            predict = model.predict(new_features)
            predicts.append(predict)

            global mae_random_forest
            mae_random_forest.append(mean_absolute_error(new_target, predict))
         
    
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    global random_forest_test
    random_forest_test = predicts

def rf_Classifier(num_days):

    rf_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    rf_train_output = rf_train_input.pop('B') # Target

    rf_validation_input = validation.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'B'])  # Features

    rf_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    rf_test_output = rf_test_input.pop('B')

    limite = rf_test_input.shape[0]

    new_train_input = rf_train_input
    new_train_output = rf_train_output

    predicts = []

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2)
    model.fit(new_train_input,new_train_output)    
    
    global random_forest
    random_forest = model.predict(rf_validation_input)

    prob_model =  np.array([]).reshape(0, 2)
             
    while( limite > 0):
        limite = rf_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = rf_test_input.iloc[0:num_days]
        rf_test_input = rf_test_input.drop(rf_test_input.index[0:num_days])    

        new_target = rf_test_output.iloc[0:num_days]
        rf_test_output = rf_test_output.drop(rf_test_output.index[0:num_days])    

        
        if(limite > 0):
            
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    

            predict = model.predict(new_features)
            predicts.append(predict)

            global auc_random_forest
            try:
                auc_random_forest.append(roc_auc_score(new_target, predict))
            except:
                auc_random_forest.append(0.5)
            nova_matriz = np.array(model.predict_proba(new_features))

            prob_model = np.vstack((prob_model, nova_matriz))

        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    
    global random_forest_test
    random_forest_test = predicts

    global random_forest_probabilidade
    random_forest_probabilidade = prob_model

## Rede Neural - MLP

def mlp_Regressor(num_days):

    mlp_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    mlp_train_output = mlp_train_input.pop('R') # Target

    mlp_validation_input = validation.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5', 'R'])  # Features

    mlp_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    mlp_test_output = mlp_test_input.pop('R')

    limite = mlp_test_input.shape[0]

    new_train_input = mlp_train_input
    new_train_output = mlp_train_output

    predicts = []

    model = MLPRegressor(hidden_layer_sizes=(22, 10), max_iter=20000,learning_rate_init=0.1,solver='lbfgs' ,activation='logistic', random_state=42)
    model.fit(new_train_input,new_train_output)    
    global mlp
    mlp = model.predict(mlp_validation_input)
  
    while( limite > 0):
        limite = mlp_test_input.shape[0]

        
        if(num_days > limite):
            num_days = limite

        new_features = mlp_test_input.iloc[0:num_days]
        mlp_test_input = mlp_test_input.drop(mlp_test_input.index[0:num_days])    

        new_target = mlp_test_output.iloc[0:num_days]
        mlp_test_output = mlp_test_output.drop(mlp_test_output.index[0:num_days])    
        
        if(limite > 0):
           
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global mae_mlp
            mae_mlp.append(mean_absolute_error(new_target, predict))
         
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    global mlp_test
    mlp_test = predicts

def mlp_Classifier(num_days):
    mlp_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    mlp_train_output = mlp_train_input.pop('B') # Target

    mlp_validation_input = validation.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'B'])  # Features

    mlp_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    mlp_test_output = mlp_test_input.pop('B')

    limite = mlp_test_input.shape[0]

    new_train_input = mlp_train_input
    new_train_output = mlp_train_output

    predicts = []

    model = MLPClassifier(hidden_layer_sizes=(22, 10), max_iter=1000,learning_rate_init=0.1,solver='sgd' ,activation='tanh', random_state=42)
    model.fit(new_train_input,new_train_output)    

    global mlp
    mlp = model.predict(mlp_validation_input)
  
    prob_model =  np.array([]).reshape(0, 2)

    while( limite > 0):
        limite = mlp_test_input.shape[0]

        
        if(num_days > limite):
            num_days = limite

        new_features = mlp_test_input.iloc[0:num_days]
        mlp_test_input = mlp_test_input.drop(mlp_test_input.index[0:num_days])    

        new_target = mlp_test_output.iloc[0:num_days]
        mlp_test_output = mlp_test_output.drop(mlp_test_output.index[0:num_days])    
        
        if(limite > 0):
           
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global auc_mlp
            try:
                auc_mlp.append(roc_auc_score(new_target, predict))
            except:
                auc_mlp.append(0.5)
            nova_matriz = np.array(model.predict_proba(new_features))

            prob_model = np.vstack((prob_model, nova_matriz))

        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    
    global mlp_test
    mlp_test = predicts

    global mlp_probabilidade
    mlp_probabilidade = prob_model

## XGBoost 

def xgboost_Regressor(num_days):

    xgboost_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    xgboost_train_output = xgboost_train_input.pop('R') # Target

    xgboost_validation_input = validation.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    xgboost_validation_output = xgboost_validation_input.pop('R') # Target

    xgboost_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    xgboost_test_output = xgboost_test_input.pop('R')

    limite = xgboost_test_input.shape[0]

    new_train_input = xgboost_train_input
    new_train_output = xgboost_train_output

    predicts = []

    model = XGBRegressor()
    model.fit(new_train_input,new_train_output)    
    global xgboost
    xgboost = model.predict(xgboost_validation_input)

    while( limite > 0):
        limite = xgboost_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = xgboost_test_input.iloc[0:num_days]
        xgboost_test_input = xgboost_test_input.drop(xgboost_test_input.index[0:num_days])    

        new_target = xgboost_test_output.iloc[0:num_days]
        xgboost_test_output = xgboost_test_output.drop(xgboost_test_output.index[0:num_days])    
        
        if(limite > 0):
            
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global mae_xgboost
            mae_xgboost.append(mean_absolute_error(new_target, predict))
         
    
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    global xgboost_test
    xgboost_test = predicts

def xgboost_Classifier(num_days):

    xgboost_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    xgboost_train_output = xgboost_train_input.pop('B') # Target

    xgboost_validation_input = validation.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'B'])  # Features

    xgboost_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    xgboost_test_output = xgboost_test_input.pop('B') # Target

    limite = xgboost_test_input.shape[0]

    new_train_input = xgboost_train_input
    new_train_output = xgboost_train_output

    predicts = []

    model = XGBClassifier(max_depth= 3, min_child_weight= 1, subsample= 0.8, colsample_bytree= 0.8, gamma= 0, alpha= 1)
    model.fit(new_train_input,new_train_output)    
    global xgboost
    xgboost = model.predict(xgboost_validation_input)

    prob_model =  np.array([]).reshape(0, 2)

    while( limite > 0):
        limite = xgboost_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = xgboost_test_input.iloc[0:num_days]
        xgboost_test_input = xgboost_test_input.drop(xgboost_test_input.index[0:num_days])    

        new_target = xgboost_test_output.iloc[0:num_days]
        xgboost_test_output = xgboost_test_output.drop(xgboost_test_output.index[0:num_days])    
        
        if(limite > 0):
            
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global auc_xgboost
            try:
                auc_xgboost.append(roc_auc_score(new_target, predict))
            except:
                auc_xgboost.append(0.5)
            nova_matriz = np.array(model.predict_proba(new_features))

            prob_model = np.vstack((prob_model, nova_matriz))

        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]

    global xgboost_test
    xgboost_test = predicts

    global xgboost_probabilidade
    xgboost_probabilidade = prob_model

## KNN

def knn_Regressor(num_days):

    knn_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    knn_train_output = knn_train_input.pop('R') # Target

    knn_validation_input = validation.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    knn_validation_output = knn_validation_input.pop('R') # Target

    knn_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    knn_test_output = knn_test_input.pop('R')

    limite = knn_test_input.shape[0]

    new_train_input = knn_train_input
    new_train_output = knn_train_output

    predicts = []

    model = KNeighborsRegressor()
    model.fit(new_train_input,new_train_output)    
    global knn
    knn = model.predict(knn_validation_input)

    while( limite > 0):
        limite = knn_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = knn_test_input.iloc[0:num_days]
        knn_test_input = knn_test_input.drop(knn_test_input.index[0:num_days])    

        new_target = knn_test_output.iloc[0:num_days]
        knn_test_output = knn_test_output.drop(knn_test_output.index[0:num_days])    
        
        if(limite > 0):

            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global mae_knn
            mae_knn.append(mean_absolute_error(new_target, predict))
         
    
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    global knn_test
    knn_test = predicts

def knn_Classifier(num_days):

    knn_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    knn_train_output = knn_train_input.pop('B') # Target

    knn_validation_input = validation.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'B'])  # Features

    knn_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    knn_test_output = knn_test_input.pop('B')
    
    limite = knn_test_input.shape[0]

    new_train_input = knn_train_input
    new_train_output = knn_train_output

    predicts = []

    model = KNeighborsClassifier()
    model.fit(new_train_input,new_train_output)    
    global knn
    knn = model.predict(knn_validation_input)

    prob_model =  np.array([]).reshape(0, 2)

    while( limite > 0):
        limite = knn_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = knn_test_input.iloc[0:num_days]
        knn_test_input = knn_test_input.drop(knn_test_input.index[0:num_days])    

        new_target = knn_test_output.iloc[0:num_days]
        knn_test_output = knn_test_output.drop(knn_test_output.index[0:num_days])    
        
        if(limite > 0):

            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global auc_knn
            try:
                auc_knn.append(roc_auc_score(new_target, predict))
            except:
                auc_knn.append(0.5)
            nova_matriz = np.array(model.predict_proba(new_features))

            prob_model = np.vstack((prob_model, nova_matriz))

        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    
    global knn_test
    knn_test = predicts

    global knn_probabilidade
    knn_probabilidade = prob_model
  
## Naive Bayes  

def naive_bayes_Classifier(num_days):

    nb_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    nb_train_output = nb_train_input.pop('B') # Target

    nb_validation_input = validation.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5', 'B'])  # Features

    nb_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    nb_test_output = nb_test_input.pop('B') # Target

    limite = nb_test_input.shape[0]

    new_train_input = nb_train_input
    new_train_output = nb_train_output

    predicts = []

    model = GaussianNB()
    model.fit(new_train_input,new_train_output)    
    
    global naive_bayes
    naive_bayes = model.predict(nb_validation_input)

    prob_model =  np.array([]).reshape(0, 2)


    while( limite > 0):
        limite = nb_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = nb_test_input.iloc[0:num_days]
        nb_test_input = nb_test_input.drop(nb_test_input.index[0:num_days])    

        new_target = nb_test_output.iloc[0:num_days]
        nb_test_output = nb_test_output.drop(nb_test_output.index[0:num_days])    
        
        if(limite > 0):
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global auc_naiveBayes
            try:
                auc_naiveBayes.append(roc_auc_score(new_target, predict))
            except:
                auc_naiveBayes.append(0.5)
            nova_matriz = np.array(model.predict_proba(new_features))

            prob_model = np.vstack((prob_model, nova_matriz))


        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    
    global naive_bayes_test
    naive_bayes_test = predicts

    global naive_bayes_probabilidade
    naive_bayes_probabilidade = prob_model

## Linear Regression
def linear_Regression(num_days):

    linear_regression_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    linear_regression_train_output = linear_regression_train_input.pop('R') # Target

    linear_regression_validation_input = validation.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    linear_regression_validation_output = linear_regression_validation_input.pop('R') # Target

    linear_regression_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    linear_regression_test_output = linear_regression_test_input.pop('R')
    
    model = LinearRegression()
   
    limite = linear_regression_test_input.shape[0]

    new_train_input = linear_regression_train_input
    new_train_output = linear_regression_train_output

    predicts = []

    model = LinearRegression()
    model.fit(new_train_input,new_train_output)    
    global linear_regression
    linear_regression = model.predict(linear_regression_validation_input)

    while( limite > 0):
        limite = linear_regression_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = linear_regression_train_input.iloc[0:num_days]
        linear_regression_test_input = linear_regression_test_input.drop(linear_regression_test_input.index[0:num_days])    

        new_target = linear_regression_test_output.iloc[0:num_days]
        linear_regression_test_output = linear_regression_test_output.drop(linear_regression_test_output.index[0:num_days])    
        
        if(limite > 0):
            # Treine o modelo
            model.fit(new_train_input,new_train_output)    
            predict = model.predict(new_features)

            predicts.append(predict)
            global mae_linear_regression
            mae_linear_regression.append(mean_absolute_error(new_target, predict))
         
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    global linear_regression_test
    linear_regression_test = predicts

## Ensemble e Stacking

def stackingClassifier():
    stackingTrain = []
    stackingTest = []

    for i in range(len(featuresValidation)):
        stackingTrain.append([random_forest[i],mlp[i],xgboost[i],knn[i],naive_bayes[i]])

    for i in range(len(featuresTest)):
        stackingTest.append([random_forest_test[i],mlp_test[i],xgboost_test[i],knn_test[i],naive_bayes_test[i]])

    df_stacking_train = pd.DataFrame(stackingTrain)
    df_stacking_test = pd.DataFrame(stackingTest)	

    train_target = out
    train_features = df_stacking_train

    test_target = out_test
    test_features = df_stacking_test

    model = MLPClassifier()
    model.fit(train_features, train_target)

    predict = model.predict(test_features)

    auc_model = roc_auc_score(test_target, predict)
    recall_model = recall_score(test_target, predict)
    precision_model = precision_score(test_target, predict)

    sim_Stacking = []
    for i in range(len(date)):
        sim_Stacking.append([date[i], predict[i]])

    df = pd.DataFrame(sim_Stacking)
    df.to_csv("./sim_Stacking.csv", index=False)

    print()
    print("AUC Stacking: ", auc_model)
    print("RECALL Stacking: ", recall_model)
    print("PRECISION Stacking: ", precision_model)

def ensembleClassifier():

    votacao = []
    out_votacao = []

    media = []
    media_ponderada = []

    # Votacao
    for i in range(len(featuresTest)):
        votacao.append([random_forest_test[i],mlp_test[i],xgboost_test[i],knn_test[i],naive_bayes_test[i]])
        contagem = Counter(votacao[i])
        numero_mais_frequente = contagem.most_common(1)[0][0]
        out_votacao.append(numero_mais_frequente)     

    auc_votacao = roc_auc_score(out_test, out_votacao)
    recall_votacao = recall_score(out_test, out_votacao)
    precision_votacao = precision_score(out_test, out_votacao)

    sim_Vot = []
    for i in range(len(date)):
        sim_Vot.append([date[i], out_votacao[i]])

    df = pd.DataFrame(sim_Vot)
    df.to_csv("./sim_Votacao.csv", index=False)

    print()
    print("AUC Votacao: ", auc_votacao)
    print("Recall Votacao: ", recall_votacao)
    print("Precision Votacao: ", precision_votacao)

    # Media
    for i in range(len(featuresTest)):
        valor0 = 0
        valor1 = 0

        valor0 = (random_forest_probabilidade[i][0] + mlp_probabilidade[i][0] + xgboost_probabilidade[i][0] + knn_probabilidade[i][0] + naive_bayes_probabilidade[i][0])/5
        valor1 = (random_forest_probabilidade[i][1] + mlp_probabilidade[i][1] + xgboost_probabilidade[i][1] + knn_probabilidade[i][1] + naive_bayes_probabilidade[i][1])/5

        if(valor1 > valor0):
            media.append(1)
        else:
            media.append(0)    
    
    auc_media = roc_auc_score(out_test, media)
    recall_media = recall_score(out_test, media)
    precision_media = precision_score(out_test, media)

    sim_M = []
    for i in range(len(date)):
        sim_M.append([date[i], media[i]])

    df = pd.DataFrame(sim_M)
    df.to_csv("./sim_Media.csv", index=False)

    print()
    print("AUC Media: ", auc_media)
    print("Recall Media: ", recall_media)
    print("Precision Media: ", precision_media)
        
    # Media Ponderada  
    for i in range(len(featuresTest)):
        valor0 = 0
        valor1 = 0
        
        posicao = int(i/1)

        valor0 = ((random_forest_probabilidade[i][0]*auc_random_forest[posicao]) + (mlp_probabilidade[i][0]*auc_mlp[posicao]) + (xgboost_probabilidade[i][0]*auc_xgboost[posicao]) + (knn_probabilidade[i][0]*auc_knn[posicao]) + (naive_bayes_probabilidade[i][0]*auc_naiveBayes[posicao]))/5
        valor1 = ((random_forest_probabilidade[i][1]*auc_random_forest[posicao]) + (mlp_probabilidade[i][1]*auc_mlp[posicao]) + (xgboost_probabilidade[i][1]*auc_xgboost[posicao]) + (knn_probabilidade[i][1]*auc_knn[posicao]) + (naive_bayes_probabilidade[i][1]*auc_naiveBayes[posicao]))/5

        if(valor1 > valor0):
            media_ponderada.append(1)
        else:
            media_ponderada.append(0)    
    
    auc_media_ponderada = roc_auc_score(out_test, media_ponderada)
    recall_media_ponderada = recall_score(out_test, media_ponderada)
    precision_media_ponderada = precision_score(out_test, media_ponderada)

    sim_MP = []

    for i in range(len(date)):
        sim_MP.append([date[i], media_ponderada[i]])

    df = pd.DataFrame(sim_MP)
    df.to_csv("./sim_MediaPonderada.csv", index=False)

    print()
    print("AUC Media Ponderada: ", auc_media_ponderada)
    print("Recall Media Ponderada: ", recall_media_ponderada)
    print("Precision Media Ponderada: ", precision_media_ponderada)
        
def stackingRegressor():
    stackingTrain = []
    stackingTest = []

    for i in range(len(featuresValidation)):
        stackingTrain.append([random_forest[i],mlp[i],xgboost[i],knn[i],linear_regression[i]])

    for i in range(len(featuresTest)):
        stackingTest.append([random_forest_test[i],mlp_test[i],xgboost_test[i],knn_test[i],linear_regression_test[i]])

    df_stacking_train = pd.DataFrame(stackingTrain)
    df_stacking_test = pd.DataFrame(stackingTest)	

    train_target = out
    train_features = df_stacking_train

    test_target = out_test
    test_features = df_stacking_test

    model = MLPRegressor(hidden_layer_sizes=(5, 2), max_iter=5000,learning_rate_init=0.1,solver='lbfgs' ,activation='logistic', random_state=42)
    model.fit(train_features, train_target)

    predict = model.predict(test_features) 

    mae = mean_absolute_error(test_target, predict)
    mse = mean_squared_error(test_target,predict)
    rmse = np.sqrt(mse)
    
    sim_Stacking = []
    for i in range(len(date)):
        sim_Stacking.append([date[i], predict[i]])

    df = pd.DataFrame(sim_Stacking)
    df.to_csv("./sim_Stacking.csv", index=False)

    print()
    print("MAE Stacking: ", mae)
    print("RMSE Stacking: ", rmse)

def ensembleRegressor():

    media = []
    media_ponderada = []

    # Media
    for i in range(len(out_test)):
        valor = (random_forest_test[i] + mlp_test[i] + xgboost_test[i] + knn_test[i] + linear_regression_test[i])/5
        media.append(valor)

    sim_M = []
    for i in range(len(date)):
        sim_M.append([date[i], media[i]])

    df = pd.DataFrame(sim_M)
    df.to_csv("./sim_Media.csv", index=False)

    mae = mean_absolute_error(out_test, media)
    mse = mean_squared_error(out_test, media)
    rmse = np.sqrt(mse)

    print()
    print("MAE Media: ", mae)
    print("RMSE Media: ", rmse)
        
    
    # Media Ponderada  
    for i in range(len(featuresTest)):
        posicao = int(i/1)
        valor= ((random_forest_test[i]/mae_random_forest[posicao]) + (mlp_test[i]/mae_mlp[posicao]) + (xgboost_test[i]/mae_xgboost[posicao]) + (knn_test[i]/mae_knn[posicao]) + (linear_regression_test[i]/mae_linear_regression[posicao]))/5
        media_ponderada.append(valor)

    mae = mean_absolute_error(out_test, media_ponderada)
    mse = mean_squared_error(out_test, media_ponderada)
    rmse = np.sqrt(mse)   
    
    sim_MP = []

    for i in range(len(date)):
        sim_MP.append([date[i], media_ponderada[i]])

    df = pd.DataFrame(sim_MP)
    df.to_csv("./sim_MediaPonderada.csv", index=False)

    print()
    print("MAE Media Ponderada: ", mae)
    print("RMSE Media Ponderada: ", rmse)

# Treinar Modelos

def execute_Classifiers():    
    global out
    out = validation['B'].to_list()

    global out_test
    out_test = test['B'].values

    baseline = []
    for i in range(len(out_test)):
        baseline.append(1)

    auc_baseline = roc_auc_score(out_test, baseline)
    recall_baseline = recall_score(out_test, baseline)
    precision_baseline = precision_score(out_test, baseline)

    print("AUC Baseline: ", auc_baseline)
    print("Recall Baseline: ", recall_baseline)
    print("Precision Baseline: ", precision_baseline)

    num_dias = 1

    rf_Classifier(num_dias)
    mlp_Classifier(num_dias)
    knn_Classifier(num_dias)
    xgboost_Classifier(num_dias)
    naive_bayes_Classifier(num_dias)
    
    stackingClassifier()
    ensembleClassifier()

def execute_Regressors():

    global out
    out = validation['R'].to_list()

    global out_test
    out_test = test['R'].to_list()

    baseline = [0]
    for i in range(1, len(out_test)):
        baseline.append(out_test[i - 1])

    mae_baseline = mean_absolute_error(out_test,baseline)
    mse_baseline = mean_squared_error(out_test, baseline)
    rmse_baseline = np.sqrt(mse_baseline)

    print("MAE Baseline: ", mae_baseline)
    print("RMSE Baseline: ", rmse_baseline)
    
    num_dias = 1

    mlp_Regressor(num_dias)
    rf_Regressor(num_dias)
    knn_Regressor(num_dias)
    xgboost_Regressor(num_dias)
    linear_Regression(num_dias)
    
    stackingRegressor()
    ensembleRegressor()

# Execuçao
#execute_Regressors()
execute_Classifiers()
