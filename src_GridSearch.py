import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, Normalizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, ParameterGrid, KFold, cross_val_score, LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn import metrics, mixture
#import xlsxwriter
import random
import matplotlib.pyplot as plt
import time

#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.neural_network import MLPRegressor


df = pd.read_excel('Base_datos.xlsx', decimal=',', header=7)


# Determinamos la propiedad que queremos estudiar y eliminamoes el resto

# Densidad 15º [kg/m^3],	Viscosidad 40º [mm^2/s],	Viscosidad 100º [mm^2/s],	Flash Point [C],	Combustion Point [C],	Pour Point [ºC],
# Cold filter plugging point [C],	Cloud Point [C],	Cetane Number,	Cooper corrosion,	Oxidative Estability,	Acid number [mg KOH/g],
# Indice Yodo [gI2/100g],	Saponification [mg KOH/g],	Moisture Content [%],	Heatin Value [MJ/kg]

Propiedad='Indice Yodo [gI2/100g]'

# Se almacenan todos lo índices del dataframe con el que estamos trabajando
subset = pd.Index(list(df))

# Se modifica el subset para que contenga solo las columnas que NO queremos trabajar en el archivo
positions_to_drop = list(range(13,59))
subset = subset[~subset.isin(subset[positions_to_drop])]
subset = subset.delete(subset.get_loc(Propiedad))

# Se eliminan las columnas que no queremos en el estudio
df.drop(subset,axis='columns',inplace=True)
df.dropna(subset=Propiedad,axis=0,inplace=True)

# Elegimos que atributos queremos usa para la experimentación
# Tipo 1 = Trabajamos con FAME, Others y MUFA
# Tipo 2 = Trabajamos con FAME y Others 
# Tipo 3 = Trabajamos con FAME y MUFA
# Tipo 4 = Trabajamos con FAME
# Tipo 5 = Trabajamos con parámetros de Mustafa
# Tipo 6 = Trabajamos con 
# Tipo 7 = Trabajamos con todos
# Tipo 8 = Pruebas
# Tipo 9 = otras pruebas sin FAMES
tipo = 9


filas, columnas = df.shape
if tipo==1:
    tipo_s = ['Cn','Mw','DB','NPI']                                 # Nos quedamos con todos, Others y MUFA
    descuenta = -8 + len(tipo_s) # -8+4=4 quita en la suma los 4 que hay a partir de others (3 + la clase)
elif tipo==2:
    tipo_s = ['MUFA','PUFA','FSA','Cn','Mw','DB','NPI']             # Nos quedamos solo con Others
    descuenta = -8 + len(tipo_s) # -8+7=1 quita en la suma el que hay a partir de others (la clase)

elif tipo==3:
    tipo_s = ['Others','Cn','Mw']                        # Nos quedamos solo con MUFA
    descuenta = -9 + len(tipo_s) # -9+3=6 quita en la suma los 6 que hay a partir de los Cs (6 + la clase)

elif tipo==4:
    tipo_s = ['Others','MUFA','PUFA','FSA','Cn','Mw','DB','NPI']    # Nos quedamos solo con los FAME
    descuenta = -9 + len(tipo_s) # -9+8=1 quita en la suma los 8 que hay a partir de los Cs (7 + la clase)   

elif tipo==5:
    tipo_s = []    # Parámetros de Mustafa
    descuenta = -3
    df =df.iloc[:,columnas-6:]

elif tipo==6:
    tipo_s = ['Others','FSA','MUFA','PUFA','Mw','Cn','NPI']    # Parámetros de FAME y DB
    descuenta = -9 + len(tipo_s) # -9+7=2 quita en la suma los 2 que hay a partir de los Cs (1 + la clase)

elif tipo==7:
    tipo_s = ['FSA','MUFA','PUFA','Mw','Cn']    # Solo dobles enlaces
    descuenta = -8 + len(tipo_s) # -8+5=3 quita en la suma los 3 que hay a partir de others (2 + la clase)
    df =df.iloc[:,columnas-7:]

elif tipo==8:
    tipo_s = ['Others','Mw','Cn']    # Parámetros con todos los atributos
    tipo_s = ['Others']    # todos sin Others
    descuenta = -9 + len(tipo_s)

elif tipo==9: 
    # solo se queda con DB
    tipo_s = ['C4:0', 'C6:0', 'C8:0', 'C10:0', 'C12:0', 'C13:0', 'C14:0', 'C14:1',
       'C15:0', 'C15:1', 'C16:0', 'C16:1', 'C16:2', 'C16:3', 'C16:4', 'C17:0',
       'C17:1', 'C18:0', 'C18:1', 'C18:2', 'C18:3', 'C19:0', 'C19:1', 'C19:2',
       'C19:3', 'C20:0', 'C20:1', 'C20:2', 'C20:4', 'C20:5', 'C21:0', 'C21:1',
       'C22:0', 'C22:1', 'C22:3', 'C22:5', 'C22:6', 'C24:0','Others','FSA','MUFA','PUFA','Mw','Cn','NPI']

    # quitando los carbonos y quedando con el resto: ['MUFA','PUFA','FSA','Cn','Mw','DB','NPI']
    #tipo_s = ['C4:0', 'C6:0', 'C8:0', 'C10:0', 'C12:0', 'C13:0', 'C14:0', 'C14:1',
    #   'C15:0', 'C15:1', 'C16:0', 'C16:1', 'C16:2', 'C16:3', 'C16:4', 'C17:0',
    #   'C17:1', 'C18:0', 'C18:1', 'C18:2', 'C18:3', 'C19:0', 'C19:1', 'C19:2',
    #   'C19:3', 'C20:0', 'C20:1', 'C20:2', 'C20:4', 'C20:5', 'C21:0', 'C21:1',
    #   'C22:0', 'C22:1', 'C22:3', 'C22:5', 'C22:6', 'C24:0','Others']
    
    # solo se queda con Others
    #tipo_s = ['C4:0', 'C6:0', 'C8:0', 'C10:0', 'C12:0', 'C13:0', 'C14:0', 'C14:1',
    #   'C15:0', 'C15:1', 'C16:0', 'C16:1', 'C16:2', 'C16:3', 'C16:4', 'C17:0',
    #   'C17:1', 'C18:0', 'C18:1', 'C18:2', 'C18:3', 'C19:0', 'C19:1', 'C19:2',
    #   'C19:3', 'C20:0', 'C20:1', 'C20:2', 'C20:4', 'C20:5', 'C21:0', 'C21:1',
    #   'C22:0', 'C22:1', 'C22:3', 'C22:5', 'C22:6', 'C24:0','DB','FSA','MUFA','PUFA','Mw','Cn','NPI']
    
    # nos quedamos solo con los 13 del artículo de los brasileños:
    # ['C8:0', 'C10:0', 'C12:0', 'C14:0', 'C16:0', 'C18:0', 'C18:1', 'C18:2', 'C18:3', 'C20:0', 'C20:1', 'C22:1', 'C22:3']
    tipo_s = ['C4:0', 'C6:0', 'C13:0', 'C14:1', 'C15:0', 'C15:1', 'C16:1', 'C16:2', 'C16:3', 'C16:4', 
              'C17:0', 'C17:1', 'C19:0', 'C19:1', 'C19:2', 'C19:3', 'C20:2', 'C20:4', 'C20:5', 'C21:0', 
              'C21:1', 'C22:0', 'C22:5', 'C22:6', 'C24:0', 'Others','MUFA','PUFA','FSA','Cn','Mw','DB','NPI']
    
    # solo se queda con DB, MUFA y PUFA
    tipo_s = ['C4:0', 'C6:0', 'C8:0', 'C10:0', 'C12:0', 'C13:0', 'C14:0', 'C14:1',
       'C15:0', 'C15:1', 'C16:0', 'C16:1', 'C16:2', 'C16:3', 'C16:4', 'C17:0',
       'C17:1', 'C18:0', 'C18:1', 'C18:2', 'C18:3', 'C19:0', 'C19:1', 'C19:2',
       'C19:3', 'C20:0', 'C20:1', 'C20:2', 'C20:4', 'C20:5', 'C21:0', 'C21:1',
       'C22:0', 'C22:1', 'C22:3', 'C22:5', 'C22:6', 'C24:0','Others','FSA','Mw','Cn','NPI']

    

# Se eliminan las columnas con las que no queremos trabajar
df.drop(tipo_s,axis='columns',inplace=True)
# Fijamos la semilla de los números aleatorios
random.seed(1234)
# Para probar con TODOS los atributos hay que comentar el df.drop y poner descuenta a -9 para que solo sume los carbonos 
#descuenta = -9

# Ejemplos a eliminar a dedo para mejor la performance
#index_to_drop = [26,76,158,301,302,348,383,394,273,303,404,420,569,405] # Los que mejor funcionan para DB
#index_to_drop = [26,73,76,301,302,348,383,394,158,273,404,420,405,328]
#df = df.drop(index_to_drop)


# ELIMINAR LOS QUE SUMEN MÁS DEL 100%
ELIMINAR_MAYORES_100 = False
if ELIMINAR_MAYORES_100:
    suma_por_ejemplo = df.iloc[:,:descuenta].select_dtypes(np.number).sum(axis=1)
    ejemplos_a_eliminar = suma_por_ejemplo[suma_por_ejemplo > 104.00001] # para que no elimine los de 100 por problemas de redondeo
    print("Se eliminan %d ejemplos:" % ejemplos_a_eliminar.count())
    print(ejemplos_a_eliminar)
    df = df.drop(ejemplos_a_eliminar.index, axis=0)
    filas, columnas = df.shape

# se eliminan columnas con pocos datos distintos de 0
UMBRAL_atributos = 1 # al menos deben tener estos ejemplos
if UMBRAL_atributos > 0:
    datos_por_atributo = df[df>0].count()
    atributos_a_eliminar = datos_por_atributo[datos_por_atributo < UMBRAL_atributos]
    print("Se eliminan %d atributos:" % atributos_a_eliminar.count())
    print(atributos_a_eliminar)
    df = df.drop(atributos_a_eliminar.index, axis=1)
    filas, columnas = df.shape


# se eliminan los ejemplos cuya suma de atributos no llega al umbral requerido
UMBRAL_FAME = 90 # al menos deben tener este FAME
if UMBRAL_FAME > 0 and tipo != 9: # en el 9 quito los carbonos
    # se suma el % de FAME de cada ejemplo
    suma_por_ejemplo = df.iloc[:,:descuenta].select_dtypes(np.number).sum(axis=1)
    ejemplos_a_eliminar = suma_por_ejemplo[suma_por_ejemplo < UMBRAL_FAME]
    print("Se eliminan %d ejemplos:" % ejemplos_a_eliminar.count())
    print(ejemplos_a_eliminar)
    df = df.drop(ejemplos_a_eliminar.index, axis=0)
    filas, columnas = df.shape


#eliminar los atributos que se quieren descartar


#Se rellenan todos los valores de FAME que son NaN como 0 y se eliminan filas duplicadas

df = df.fillna(0)
df = df.drop_duplicates(keep='first')



X = df.iloc[:,0:(columnas-1)]
y = df.iloc[:,(columnas-1)]

# obtenemos el número de atributos
(num_ejemplos, num_atributos) = X.shape 
print("* NUMERO DE EJEMPLOS: ", num_ejemplos)
print("* NÚMERO DE ATRIBUTOS:", num_atributos)
print(df.columns)

LOO = LeaveOneOut()

# comprobar que los hace en orden
#for i, (train_index, test_index) in enumerate(LOO.split(X)):
#	print(f"Fold {i}: Test:  index={test_index}")

predicciones = np.empty((X.shape[0], 0))
resultados = np.empty((0, 7))
sys_names = []

RUN_DummyRegressor = True
RUN_STD_RegLin     = True
RUN_DecTree        = True
RUN_RandomForest   = True
RUN_XGBoost        = True
RUN_STD_SRV_LINEAR = True
RUN_STD_SRV_POLY   = True
RUN_STD_SRV_RBF    = True
RUN_ANN            = True
RUN_ANN_brasil     = True


if RUN_DummyRegressor:
    # *************************************
    # ********** DummyRegressor ***********
    sys = DummyRegressor(strategy="mean")
    ini = time.time()
    pred = cross_val_predict(sys, X, y, cv=LOO)
    fin = time.time()
    sys_names.append('DumReg')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    sys.fit(X,y)
    pred = sys.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])


if RUN_STD_RegLin:
    # *************************************
    # ********** std + LinearRegression ***********
    sys = Pipeline([('std', StandardScaler()), ('lr', LinearRegression())])
    #sys = LinearRegression()
    ini = time.time()
    pred = cross_val_predict(sys, X, y, cv=LOO, n_jobs=-1)
    fin = time.time()
    sys_names.append('std_LinReg')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    sys.fit(X,y)
    pred = sys.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])

    


if RUN_DecTree:
    # *************************************
    # ********** DecisionTreeRegressor ***********
    sys = DecisionTreeRegressor(random_state=1234)

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_max_depth = [5, 6, 7, 8, 9, 10]
    hyperparameters = dict(max_depth=valores_max_depth)

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)
    #gs = GridSearchCV(sys, hyperparameters, scoring='r2', cv=folds5, verbose=1, n_jobs=-1)

    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('DecisionTree')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para DecisionTree:", res_gs.best_params_)
    # {'max_depth': 5}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])

if RUN_RandomForest:
    # *************************************
    # ********** Random Forest - GS ***********
    sys = RandomForestRegressor(random_state=1234)

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_n_estimators = [10, 50, 100]
    valores_max_depth = [6, 7, 8]
    hyperparameters = dict(n_estimators=valores_n_estimators, max_depth=valores_max_depth)

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)

    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('RandForest')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para RandForest:", res_gs.best_params_)
    # {'max_depth': 6, 'n_estimators': 100}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])


if RUN_XGBoost:
    # *************************************
    # ********** XGBRegressor ***********
    sys = XGBRegressor()
    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_n_estimators = [10, 50, 100]
    valores_max_depth = [2, 4, 6, 8]
    valores_learning_rate = [0.01, 0.1, 0.3, 0.5]
    hyperparameters = dict(n_estimators=valores_n_estimators, max_depth=valores_max_depth, learning_rate=valores_learning_rate)

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)
    #gs = GridSearchCV(sys, hyperparameters, scoring='r2', cv=folds5, verbose=1, n_jobs=-1)


    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('XGBoost')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para XGBoost:", res_gs.best_params_)
    # {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])



if RUN_STD_SRV_LINEAR:
    # *************************************
    # ********** STD - SVR - linear - GS ***********
    svr = SVR(kernel='linear')
    sys = Pipeline([('std', StandardScaler()), ('svr', svr)])

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_Cs = [0.01, 0.1, 1, 10]
    hyperparameters = dict(svr__C=valores_Cs)

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)
    #gs = GridSearchCV(sys, hyperparameters, scoring='r2', cv=folds5, verbose=1, n_jobs=-1)

    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('SVR-STD-linear')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para SVR-STD-linear:", res_gs.best_params_)
    #{'svr__C': 1}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])



if RUN_STD_SRV_POLY:
    # **************************************
    # ********** STD - SVR - POLY - GS ***********
    svr = SVR(kernel='poly')
    sys = Pipeline([('std', StandardScaler()), ('svr', svr)])

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_Cs = [0.01, 0.1, 1, 10]
    valores_degrees = [1, 2, 3, 4]   
    hyperparameters = dict(svr__C=valores_Cs, svr__degree=valores_degrees)

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)
    #gs = GridSearchCV(sys, hyperparameters, scoring='r2', cv=folds5, verbose=1, n_jobs=-1)

    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('SVR-STD-POLY')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para SVR-POLY:", res_gs.best_params_)
    # {'svr__C': 10, 'svr__degree': 1}
    # {'svr__C': 100, 'svr__degree': 1}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])

if RUN_STD_SRV_RBF:
    # *************************************
    # ********** STD - SVR - RBF - GS ***********
    svr = SVR(kernel='rbf')
    sys = Pipeline([('std', StandardScaler()), ('svr', svr)])

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_Cs = [0.1, 1, 10, 100]
    valores_gammas = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
    hyperparameters = dict(svr__C=valores_Cs, svr__gamma=valores_gammas)

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)
    #gs = GridSearchCV(sys, hyperparameters, scoring='r2', cv=folds5, verbose=1, n_jobs=-1)

    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('SVR-STD-RBF')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para SVR-RBF:", res_gs.best_params_)
    # {'svr__C': 100, 'svr__gamma': 0.01}
    # {'svr__C': 1000, 'svr__gamma': 0.01}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])


if RUN_ANN:
    # *************************************
    # ********** ANN  ***********
    sys = MLPRegressor(random_state=1234, solver='sgd', tol=0.1, batch_size=32, max_iter=2000)

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_capas = [(8,), (16,)]
    valores_lr_init = [0.00001, 0.0001, 0.001]
    #valores_activation = ['logistic', 'tanh', 'relu']
    #valores_reg = [0.000001, 0.00001, 0.0001]
    #hyperparameters = dict(hidden_layer_sizes=valores_capas, learning_rate_init=valores_lr_init, alpha=valores_reg)
    hyperparameters = dict(hidden_layer_sizes=valores_capas, learning_rate_init=valores_lr_init)
    #hyperparameters = dict(hidden_layer_sizes=valores_capas, learning_rate_init=valores_lr_init, activation=valores_activation)
 

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)


    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('ANN')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para ANN:", res_gs.best_params_)
    # {'hidden_layer_sizes': (16,), 'learning_rate_init': 1e-05}
    # {'activation': 'relu', 'hidden_layer_sizes': (16,), 'learning_rate_init': 1e-05}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])

if RUN_ANN_brasil:
    # *************************************
    # ********** ANN  ***********
    sys = MLPRegressor(random_state=1234, hidden_layer_sizes=(24,6), n_iter_no_change=20, solver='sgd', 
                       tol=0.1, batch_size=32, max_iter=2000)#, activation='relu')

    # se definen los valores de los hiperparámetros que se quieren probar y se introducen en un diccionario
    valores_lr_init = [0.0001, 0.001]
    valores_activation = ['logistic', 'tanh', 'relu']
    #hyperparameters = dict(learning_rate_init=valores_lr_init)
    hyperparameters = dict(learning_rate_init=valores_lr_init, activation=valores_activation)
 

    folds5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    gs = GridSearchCV(sys, hyperparameters, scoring='neg_mean_absolute_error', cv=folds5, verbose=1, n_jobs=-1)


    ini = time.time()
    pred = cross_val_predict(gs, X, y, cv=LOO, verbose=5, n_jobs=-1)
    fin = time.time()
    sys_names.append('ANN_(24,6)')
    predicciones = np.hstack([predicciones, pred.reshape(X.shape[0],1)])

    mae = metrics.mean_absolute_error(y, pred)
    mse = metrics.mean_squared_error(y, pred)
    r2 = metrics.r2_score(y, pred)
    tiempo = fin - ini  # en segundos

    # ejecutamos la búsqueda sin validación cruzada para obtener la mejor combinación
    res_gs = gs.fit(X, y)
    print("\nMejor combinación de hiperparámetros para ANN_(24,6):", res_gs.best_params_)
    # {'activation': 'logistic', 'learning_rate_init': 0.0001}
    pred = gs.predict(X)
    mae_rees = metrics.mean_absolute_error(y, pred)
    mse_rees = metrics.mean_squared_error(y, pred)
    r2_rees = metrics.r2_score(y, pred)
    print(mae_rees, mse_rees, r2_rees)
    resultados = np.vstack([resultados, [mae,mse,r2,tiempo,mae_rees,mse_rees,r2_rees]])




print('\n##########################################')
print('### exportado a excel')
print('##########################################\n')

df_predicciones = pd.DataFrame(predicciones, columns=sys_names)

df_completo = pd.concat([df.reset_index(), df_predicciones], axis=1)

df_resultados = pd.DataFrame(resultados, index=sys_names, columns=['MAE', 'MSE', 'R2', 'segundos', 'MAE_REES', 'MSE_REES', 'R2_REES'])
print(df_resultados)

df_completo.to_excel('predicciones.xlsx', index=False)
df_resultados.to_excel('resultados.xlsx')


print('\n##########################################')
print('### Detección de anomalías')
print('##########################################\n')

NUM_GAUSSIANAS = 1
gmm = mixture.GaussianMixture(n_components=NUM_GAUSSIANAS, covariance_type='full')

# se entrena la mezcla de gaussianas
gmm.fit(X)
# se obtiene el log-likelihood
val = gmm.score_samples(X)

print("LOG-LIKELIHOOD:\n\tmáximo:", max(val), "\n\tmínimo:", min(val))
print("\nBIC: %.1f (cuanto menor mejor)\n\n" % (gmm.bic(X)))

if num_atributos == 2:
    # se crean datos que cubran el espacio en forma de rejilla
    x = np.linspace(min(X.iloc[:,0])-2, max(X.iloc[:,0])+2)
    y = np.linspace(min(X.iloc[:,1])-2, max(X.iloc[:,1])+2)
    xx, yy = np.meshgrid(x, y)

    # se recolocan en formato dataset para calcular su likelihood 
    espacio_completo = np.array([xx.ravel(), yy.ravel()]).T
    df_espacio_completo = pd.DataFrame(espacio_completo, columns=X.columns)
    zz = -gmm.score_samples(df_espacio_completo)
    zz= zz.reshape(xx.shape)

    # se pintan los contornos y los puntos
    CS = plt.contour(xx, yy, zz, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8)
    #CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X.iloc[:,0], X.iloc[:,1], 0.8)
    plt.title('log-likelihood negativo entrenado utilizando GMM')
    plt.axis('tight')


# se crea un dataframe con datos, predicciones y la verosimilitud
df_todo = pd.concat([X.reset_index(), y.reset_index(), df_predicciones], axis=1)
df_todo['logProb'] = val.T

# se define el porcentaje de aoutliers y se imprimen
PCT_outliers = 5 / 100
n_outliers = round(num_ejemplos * PCT_outliers)
outliers = df_todo.nsmallest(n_outliers, 'logProb')
print(outliers)

if num_atributos == 2:
    # se pintan los outliers si es 2D
    plt.scatter(outliers.iloc[:,0].values, outliers.iloc[:,1].values, marker='^')
    plt.show()

# probabilidad de pertenencia a cada una de las gaussianas
prob = gmm.predict_proba(X)
cabecera = ['Gauss'+str(x) for x in range(1,NUM_GAUSSIANAS+1)]
df_probs = pd.DataFrame(prob, columns=cabecera)

# se vuelcan a excel todos los resultados
df_todo = pd.concat([df_todo, df_probs], axis=1)
df_todo.to_excel('prediccionesYanomalías.xlsx', index=False)