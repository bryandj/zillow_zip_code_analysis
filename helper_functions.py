#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[ ]:


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# In[ ]:


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# In[ ]:


# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools


# In[ ]:


#function to fit a SARIMAX model
def run_SARIMAX(ts,pdq,pdqs): 
    '''
    Fits a Statsmodels SARIMAX Time Series Model.
    
    Arguments:
         ts    = The time series being modeled.
         pdq   = list containing the p,d,and q values for the SARIMAX function.
         pdqs  = list containing the seasonal p,d and q values along with seasonal 
                 shift amount for the SARIMAX function.
    
    Return Value:
         Dictionary containing {model: model, terms: pdq, seasonal terms: pdqs, AIC: model AIC} values.
         Return AIC of 99999 and model=None indicates that the SARIMAX failed to execute.
    '''
    try:
        mod = sm.tsa.statespace.SARIMAX(ts,
                                       order=pdq,
                                       seasonal_order=pdqs,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False);
        output = mod.fit();
        return {'model':output,'terms':pdq,'seasonal terms':pdqs,'AIC':output.aic}
    
    except:
        return {'model':None,'terms':pdq,'seasonal terms':pdqs,'AIC':99999}


# In[ ]:


def get_SARIMA_parameters(ts,p,d,q,sp,sd,sq,s):
    '''
    Fits a Statsmodels SARIMAX Time Series Model for every combination of
    p,d,q and seasonal p,d,q values provided.
    
    Arguments:
         ts = The time series being modeled.
         p  = list of autoregressive terms to test
         d  = list of differencing parameter terms to test
         q  = list of moving average terms to test
         sp = list of seasonal autoregressive terms to test
         sd = list of seasonal differencing terms to test
         sq = list of seasonal moving average terms to test
         s  = list of seasonal durations to test
         
    Return Value:
         List of dictionaries containing (terms: pdq, seasonal terms: pdqs, AIC: model AIC) values.
         The list is sorted by lowest AIC to highest. 
         AIC of 99999 indicates that the SARIMAX failed to execute.
    '''
    
    #Make combinations of pdq and sp,sd,sq,s for the model
    combos = list(itertools.product(p, d, q))
    combos_seasonal = list(itertools.product(sp, sd, sq, s))
    print('total combos = ',len(combos) * len(combos_seasonal))
    
    #results will be appended to this empty list
    results = []
    
    #iterate through every combination, run a SARIMAX model and store the AIC value
    for combo in combos:
        for combo_season in combos_seasonal:
            result = run_SARIMAX(ts,combo,combo_season);
            #remove model because this function is only for parameter selection
            del result['model']
            results.append(result);
    
    #return the list sorted by AIC value
    return sorted(results,key=lambda x: x['AIC'])


# In[ ]:


def plot_prediction(ts,model,date,dynamic=False):
    '''
    Plot a prediction for the time series and model provided.
    
    Arguments:
         ts    = The time series being plotted.
         model = The model being used to make predictions.
         date  = The date after which predictions will be plotted
         dynamic = If True, use dynamic predictions from date provided
         
    Return Value:
         RMSE for the predicitons for the forecast
    
    '''
    pred = model.get_prediction(start=pd.to_datetime(date),dynamic=dynamic,full_results=True)
    pred_conf = pred.conf_int()
    ax = ts.plot(label='observed', figsize=(20,5))
    if dynamic:
        label='dynamic prediction from {}'.format(date)
    else:
        label='one-step-ahead prediction'
    pred.predicted_mean.plot(ax=ax, label=label,color='darkorange',alpha=0.9)
    ax.fill_between(pred_conf.index,pred_conf.iloc[:,0],pred_conf.iloc[:,1],color='b',alpha=0.1)
    plt.legend(loc='upper left')
    forecasted = pred.predicted_mean
    truth = ts[date :]
    RMSE = np.sqrt(np.mean((forecasted-truth)**2))
    print("RMSE: ",RMSE)
    return RMSE


# In[ ]:


def get_forecast(ts,model,steps,plot=False):
    '''
    Get an ROI 1-yr and 5-yr using a forcast with the time seriex and model provided.
    
    Arguments:
         ts    = The time series being forecasted.
         model = The model being used to make a forecast.
         steps = The number of steps ahead to forecast.
         plot  = If True, plot the forecast
         
    Return Value:
         Dictionary containing 1-yr and 5-yr ROI values.  
    '''
    prediction = model.get_forecast(steps=steps)
    pred_conf = prediction.conf_int()
    
    if plot:    
        #Plot future prediction with confidence interval
        ax = ts.plot(label='observed',figsize=(20,5))
        prediction.predicted_mean.plot(ax=ax,label='forecast')
        ax.fill_between(pred_conf.index,pred_conf.iloc[:,0],pred_conf.iloc[:,1],color='b',alpha=0.1)
        plt.legend(loc='upper left')
        plt.show()
    
    #At least 60 steps are required in order to calculate the 5-yr ROI
    if steps <= 60:
        prediction = model.get_forecast(steps=60)
    
    #Calculate 1yr and 5yr ROIs
    last_val = ts[-1]
    roi_1yr = round( (prediction.predicted_mean[11] - last_val)/last_val , 3)
    roi_5yr = round( (prediction.predicted_mean[59] - last_val)/last_val , 3)
    
    return {'ROI_1yr':roi_1yr, 'ROI_5yr': roi_5yr}


# In[ ]:


def plot_forecasts(df, model_dicts, steps, title=None):
    '''
    Plot forecasts for all zipcodes in results dictionary.
    
    Arguments:
         df         = The dataframe containing time series data with zipcodes as column names.
         model_dict = List containing dictionaries with models for each zipcode.
         steps      = The number of steps ahead to forecast.
         
    Return Value:
         A plot with all zipcodes in model_dicts  
    '''
    
    for item in model_dicts:
        zipcode = item['zipcode']
        label = f"{item['city']} {zipcode}"
        model = item['model']
        ts = df[zipcode]
        prediction = model.get_forecast(steps=steps)
        all_data = pd.concat([ts,prediction.predicted_mean],axis=0)
        ax = all_data.plot(label=label,figsize=(20,5))
    
    plt.axvline(x=df.index[-1], color='black', linestyle=':')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
    return


# In[ ]:


def pacf_plot(ts,lags=100):
    fig, ax = plt.subplots(figsize=(15,5))
    sm.graphics.tsa.plot_pacf(ts, ax=ax, lags=lags)
    return


# In[ ]:


def acf_plot(ts,lags=100):
    fig, ax = plt.subplots(figsize=(15,5))
    sm.graphics.tsa.plot_acf(ts, ax=ax, lags=lags)
    return


# In[ ]:


# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

