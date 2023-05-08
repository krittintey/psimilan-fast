from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import catboost as cb

models = {
    'linear_regression': LinearRegression(),
    'ridge': Ridge(random_state=42),
    'lasso': Lasso(random_state=42),
    'lasso_lars': LassoLars(random_state=42),
    'elastic_net': ElasticNet(random_state=42),
    'bayesian_ridge': BayesianRidge(),
    'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
    'linear_svr': LinearSVR(random_state=42),
    'xgb': XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=42),
    'random_forest_regressor': RandomForestRegressor(n_estimators=1000, random_state=42),
    'catboost': cb.CatBoostRegressor(loss_function='RMSE', verbose=False, random_state=42)
}

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def absolute_error(approx_value, exact_value):
    error = np.abs(approx_value - exact_value)
    return error

def absolute_percentage_error(approx_value, exact_value):
    error = np.abs((approx_value - exact_value)/exact_value)
    return error

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{columns[j]}(t-{i})') for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'{columns[j]}(t)') for j in range(n_vars)]
        else:
            names += [(f'{columns[j]}(t+{i})') for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# get keys from dict as list
def getList(dict):
    return dict.keys()

# fit an xgboost model and make a one step prediction
def model_forecast(linear_model, train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = linear_model
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    #print(yhat)
    return yhat[0]

# walk-forward validation for time series forecasting
def walk_forward_validation(data, n_test):
    model_predictions = {
        'linear_regression': list(),
        'ridge': list(),
        'lasso': list(),
        'lasso_lars': list(),
        'elastic_net': list(),
        'bayesian_ridge': list(),
        'decision_tree_regressor': list(),
        'linear_svr': list(),
        'xgb': list(),
        'random_forest_regressor': list(),
        'catboost': list()
    }
    metrics = {
        'mae': list(),
        'mse': list(),
        'rmse': list(),
        'mape': list(),
        'corr': list(),
        'p-value (corr)': list()
    }
    mape_iterations = {
        'linear_regression': list(),
        'ridge': list(),
        'lasso': list(),
        'lasso_lars': list(),
        'elastic_net': list(),
        'bayesian_ridge': list(),
        'decision_tree_regressor': list(),
        'linear_svr': list(),
        'xgb': list(),
        'random_forest_regressor': list(),
        'catboost': list()
    }
    ae_iterations = {
        'linear_regression': list(),
        'ridge': list(),
        'lasso': list(),
        'lasso_lars': list(),
        'elastic_net': list(),
        'bayesian_ridge': list(),
        'decision_tree_regressor': list(),
        'linear_svr': list(),
        'xgb': list(),
        'random_forest_regressor': list(),
        'catboost': list()
    }
    ape_iterations = {
        'linear_regression': list(),
        'ridge': list(),
        'lasso': list(),
        'lasso_lars': list(),
        'elastic_net': list(),
        'bayesian_ridge': list(),
        'decision_tree_regressor': list(),
        'linear_svr': list(),
        'xgb': list(),
        'random_forest_regressor': list(),
        'catboost': list()
    }
    # split dataset
    train, test = train_test_split(data, n_test)
    # get model name list
    model_names = getList(models)
    # loop for using each models 
    for md in model_names:
        predictions = list()
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
            # fit model on history and make a prediction
            yhat = model_forecast(models[md], history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
            # summarize progress
        # save the model to disk
        filename = f'../Documents/Time_series_forecasting/GH-Injure/Horizon{horizon}/Lag{lag}/{md}_{group}_h{horizon}_l{lag}.sav'
        pickle.dump(models[md], open(filename, 'wb'))
        # invert scaling for forecast
        inv_ypred = np.concatenate([test[:, :-1], np.array(predictions).reshape(test.shape[0], 1)], axis=1)
        inv_ypred = scaler.inverse_transform(inv_ypred)
        inv_ypred = inv_ypred[:,-1]
        # add prediction in list
        model_predictions[md] = inv_ypred
        # invert scaling for actual
        inv_y = np.concatenate([test[:, :-1], np.array(test[:, -1]).reshape(test.shape[0], 1)], axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,-1]
        # estimate prediction error
        mae = mean_absolute_error(inv_y, inv_ypred)
        mse = mean_squared_error(inv_y, inv_ypred)
        rmse = (np.sqrt(mean_squared_error(inv_y, inv_ypred)))
        mape = mean_absolute_percentage_error(inv_y, inv_ypred)
        # add error in list
        metrics['mae'].append(mae)
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mape'].append(mape)
        # collect the mape for each iteration and each model
        for i in range(len(inv_y)):
            inv_predictions = inv_ypred[:i+1]
            inv_actual = inv_y[:i+1]
            mape_step = mean_absolute_percentage_error(inv_actual, inv_predictions)
            ae_step = absolute_error(inv_ypred[i], inv_y[i])
            ape_step = absolute_percentage_error(inv_ypred[i], inv_y[i])
            # Append in dict
            mape_iterations[md].append(mape_step)
            ae_iterations[md].append(ae_step)
            ape_iterations[md].append(ape_step)
    #print(mape_iterations)
    return model_predictions, metrics, mape_iterations, ae_iterations, ape_iterations

df = pd.read_csv('../Documents/ICT_SP/selfharm_time_series_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index(keys='date', inplace=True)

# All features
# df_new = pd.concat([df.iloc[:, :-6], df.iloc[:, -3:]], axis=1) # Normalization
df_with_raw_selfharm = df.copy() # Raw selfharm
# cols = df_with_raw_selfharm.columns.tolist()
# new_columns = cols[:-2] + cols[-1:] + cols[-2:-1]
# df_new = df_with_raw_selfharm[new_columns]
df_all = df_with_raw_selfharm.iloc[:-2]
##---------------------------------------------------------------------## 
# Grouped features
# Sentiment
df_injure = df_with_raw_selfharm.iloc[:, -1]
df_sentiment = df_with_raw_selfharm.iloc[:, :4]
df_new = pd.concat([df_sentiment, df_injure], axis=1)
df_ms = df_new.iloc[:-2]

# Emotion
df_injure = df_with_raw_selfharm.iloc[:, -1]
df_emotion = df_with_raw_selfharm.iloc[:, 4:11]
df_new = pd.concat([df_emotion, df_injure], axis=1)
df_me = df_new.iloc[:-2]

# Suicidal Tendency
df_injure = df_with_raw_selfharm.iloc[:, -1]
df_suicide = df_with_raw_selfharm.iloc[:, 11:13]
df_new = pd.concat([df_suicide, df_injure], axis=1)
df_m = df_new.iloc[:-2]

# Selfharm
df_new = df_with_raw_selfharm.iloc[:, -2:]
# cols = df_new.columns.tolist()
# new_columns = cols[:-2] + cols[-1:] + cols[-2:-1]
# df_new = df_new[new_columns]
df_gh = df_new.iloc[:-2]
##---------------------------------------------------------------------## 
# Combination features
# Sentiment + Emotion
df_injure = df_with_raw_selfharm.iloc[:, -1]
df_sentiment = df_with_raw_selfharm.iloc[:, :4]
df_emotion = df_with_raw_selfharm.iloc[:, 4:11]
df_new = pd.concat([df_sentiment, df_emotion, df_injure], axis=1)
df_ms_me = df_new.iloc[:-2]

# Sentiment + Suicidal Tendency
df_injure = df_with_raw_selfharm.iloc[:, -1]
df_sentiment = df_with_raw_selfharm.iloc[:, :4]
df_suicide = df_with_raw_selfharm.iloc[:, 11:13]
df_new = pd.concat([df_sentiment, df_suicide, df_injure], axis=1)
df_ms_m = df_new.iloc[:-2]

# Sentiment + Selfharm
df_selfharm = df_with_raw_selfharm.iloc[:, -1]
df_sentiment = df_with_raw_selfharm.iloc[:, :4]
df_new = pd.concat([df_sentiment, df_selfharm], axis=1)
# cols = df_new.columns.tolist()
# new_columns = cols[:-2] + cols[-1:] + cols[-2:-1]
# df_new = df_new[new_columns]
df_ms_gh = df_new.iloc[:-2]

# Emotion + Suicidal Tendency
df_injure = df_with_raw_selfharm.iloc[:, -1]
df_emotion = df_with_raw_selfharm.iloc[:, 4:11]
df_suicide = df_with_raw_selfharm.iloc[:, 11:13]
df_new = pd.concat([df_emotion, df_suicide, df_injure], axis=1)
df_me_m = df_new.iloc[:-2]

# Emotion + Selfharm
df_selfharm = df_with_raw_selfharm.iloc[:, -3:]
df_emotion = df_with_raw_selfharm.iloc[:, 4:11]
df_new = pd.concat([df_emotion, df_selfharm], axis=1)
# cols = df_new.columns.tolist()
# new_columns = cols[:-2] + cols[-1:] + cols[-2:-1]
# df_new = df_new[new_columns]
df_me_gh = df_new.iloc[:-2]

# Suicidal Tendency + Selfharm
df_selfharm = df_with_raw_selfharm.iloc[:, -3:]
df_suicide = df_with_raw_selfharm.iloc[:, 11:13]
df_new = pd.concat([df_suicide, df_selfharm], axis=1)
# cols = df_new.columns.tolist()
# new_columns = cols[:-2] + cols[-1:] + cols[-2:-1]
# df_new = df_new[new_columns]
df_m_gh = df_new.iloc[:-2]
##---------------------------------------------------------------------## 
# df_new = df_new.iloc[:-2]
# df_new

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pickle

datasets = {
        'all': df_all,
        'ms': df_ms,
        'me': df_me,
        'm': df_m,
        #'gh': df_gh,
        'ms_me': df_ms_me,
        'ms_m': df_ms_m,
        'ms_gh': df_ms_gh,
        'me_m': df_me_m,
        'me_gh': df_me_gh,
        'm_gh': df_m_gh
    }

datasets_drop_columns_num = {
        'all': 14,
        'ms': 4,
        'me': 7,
        'm': 2,
        #'gh': 1,
        'ms_me': 11,
        'ms_m': 6,
        'ms_gh': 5,
        'me_m': 9,
        'me_gh': 8,
        'm_gh': 3
    }

horizons = [0, 1, 2, 3]
lags = [0, 1, 3, 6, 9, 12]

for horizon in horizons:
    for lag in lags:
        for group in getList(datasets):
            values = datasets[group].values
            columns = datasets[group].columns
            data = series_to_supervised(values, columns, n_in=lag, n_out=horizon+1)
            
            if horizon > 0:
                data.drop(columns=data.columns[(datasets_drop_columns_num[group]*(lag+1))+lag:-1], inplace=True)
            #if ('GH-Death(t)' in data.columns) and (lag != 0):
            if ('GH-Death(t)' in data.columns):
                data.drop(columns='GH-Death(t)', inplace=True)
                
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            model_predictions, metrics, mape_iterations, ae_iterations, ape_iterations = walk_forward_validation(data_scaled, 12)
            
            if lag == 0:
                win = 1
            else:
                win = lag

            df_results = datasets[group][['GH-Injure']].iloc[-12:].copy()
            model_names = getList(model_predictions)

            for md in model_names:
                df_results[md] = model_predictions[md]
                
            y_true = df_results['GH-Injure'].values
            
            for md in model_names:
                corr, p_value = stats.pearsonr(y_true, df_results[md].values)
                metrics['corr'].append(corr)
                metrics['p-value (corr)'].append(p_value) 

            model_names = list(getList(model_predictions))
            metric_names = getList(metrics)
            df_eval = pd.DataFrame([], index=model_names, columns=metric_names)

            for mt in metric_names:
                df_eval[mt] = metrics[mt]
                
            error_iterations_names = list(getList(mape_iterations))
            df_mape_iterations = pd.DataFrame([], columns=error_iterations_names)
            df_ae_iterations = pd.DataFrame([], columns=error_iterations_names)
            df_ape_iterations = pd.DataFrame([], columns=error_iterations_names)

            for mn in error_iterations_names:
                df_mape_iterations[mn] = mape_iterations[mn]
                df_ae_iterations[mn] = ae_iterations[mn]
                df_ape_iterations[mn] = ape_iterations[mn]
            
            df_results.to_csv(f'../Documents/Time_series_forecasting/GH-Injure/Horizon{horizon}/Lag{lag}/traditional_ml_{group}_h{horizon}_l{lag}_predictions.csv', index=True)
            df_eval.to_csv(f'../Documents/Time_series_forecasting/GH-Injure/Horizon{horizon}/Lag{lag}/traditional_ml_{group}_h{horizon}_l{lag}_eval.csv', index=True)
            df_mape_iterations.to_csv(f'../Documents/Time_series_forecasting/GH-Injure/Horizon{horizon}/Lag{lag}/traditional_ml_{group}_h{horizon}_l{lag}_mape_iterations.csv', index=True)
            df_ae_iterations.to_csv(f'../Documents/Time_series_forecasting/GH-Injure/Horizon{horizon}/Lag{lag}/traditional_ml_{group}_h{horizon}_l{lag}_ae_iterations.csv', index=True)
            df_ape_iterations.to_csv(f'../Documents/Time_series_forecasting/GH-Injure/Horizon{horizon}/Lag{lag}/traditional_ml_{group}_h{horizon}_l{lag}_ape_iterations.csv', index=True)

            
