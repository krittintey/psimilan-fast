import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm
from scipy import stats

def absolute_error(approx_value, exact_value):
    error = np.abs(approx_value - exact_value)
    return error

def absolute_percentage_error(approx_value, exact_value):
    error = np.abs((approx_value - exact_value)/exact_value)
    return error

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

df = pd.read_csv('./ICT_SP/selfharm_time_series_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index(keys='date', inplace=True)
df_new = df.copy()
cols = df_new.columns.tolist()
new_columns = cols[:-2] + cols[-1:] + cols[-2:-1]
df_new = df_new[new_columns]
df_new = df_new.iloc[:-2]

lag_list = [0, 1, 3, 6, 9, 12]
horizon_list = [0, 1, 2, 3, 4, 5, 6]

for horizon in horizon_list:
    for lag in lag_list:
        col = 'GH-Death'
        lags = lag
        horizon = horizon
        values = df_new.values
        columns = df_new.columns
        data = series_to_supervised(values, columns, n_in=lags, n_out=1+horizon)

        if horizon > 0:
            data.drop(columns=data.columns[(14*(lags+1))+lags:-1], inplace=True)
        if ('GH-Injure(t)' in data.columns) and (lag != 0):
            data.drop(columns='GH-Injure(t)', inplace=True)
                
        data.index = df_new.index[lags+horizon:len(df_new)]

        scaler = MinMaxScaler() 
        data_scaled = data.copy()
        data_scaled[data_scaled.columns] = scaler.fit_transform(data_scaled[data_scaled.columns])

        # SARIMAX Model
        sxmodel = pm.auto_arima(y=data_scaled.iloc[:-12, -1], X=data_scaled.iloc[:-12, :-1],
                                   start_p=1, start_q=1,
                                   test='adf',
                                   max_p=3, max_q=3, m=1,
                                   seasonal=False,
                                   d=None, D=1, trace=True,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True, random_state=42)

        n_periods = 12
        preds, conf_int = sxmodel.predict(n_periods=n_periods, X=data_scaled.iloc[-12:, :-1], return_conf_int=True)

        inv_ypred = np.concatenate([data_scaled.iloc[-12:, :-1].values, np.array(preds).reshape(-1, 1)], axis=1)
        inv_ypred = scaler.inverse_transform(inv_ypred)
        inv_ypred = inv_ypred[:,-1]

        df_results = df_new[[col]].iloc[-12:].copy()
        df_results['ARIMA_Prediction'] = inv_ypred

        df_ground = df_results[col].values
        df_arima = df_results['ARIMA_Prediction'].values

        mae = mean_absolute_error(df_ground, df_arima)
        mse = mean_squared_error(df_ground, df_arima)
        rmse = np.sqrt(mean_squared_error(df_ground, df_arima))
        mape = mean_absolute_percentage_error(df_ground, df_arima)
        corr, p_value = stats.pearsonr(df_ground, df_arima) 
        
        errors = {
            'mape_iterations': list(),
            'ae_iterations': list(),
            'ape_iterations': list(),
        }
        
        for i in range(len(df_ground)):
            predictions = df_arima[:i+1]
            actual = df_ground[:i+1]
            mape_step = mean_absolute_percentage_error(actual, predictions)
            ae_step = absolute_error(df_arima[i], df_ground[i])
            ape_step = absolute_percentage_error(df_arima[i], df_ground[i])
            # Append in dict
            errors['mape_iterations'].append(mape_step)
            errors['ae_iterations'].append(ae_step)
            errors['ape_iterations'].append(ape_step)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'corr': corr,
            'p-value (corr)': p_value
        }
        
        name = 'death'
        df_eval = pd.DataFrame(metrics, index=['arima'])
        df_errors = pd.DataFrame(errors)
        df_results.to_csv(f'./ICT_SP/arima/results_arima_{name}_h{horizon}_l{lags}.csv', index=True)
        df_eval.to_csv(f'./ICT_SP/arima/eval_arima_{name}_h{horizon}_l{lags}.csv', index=False)
        df_errors.to_csv(f'./ICT_SP/arima/errors_arima_{name}_h{horizon}_l{lags}.csv', index=False)