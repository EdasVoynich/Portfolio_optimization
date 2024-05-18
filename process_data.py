import numpy as np
import pandas as pd
import pandas_ta as ta
import math

def process_financial_data(data):
    ## Simple Moving Average
    data['SMA Close (F)']= ta.sma(data["Close"], length= 21)
    data['SMA Close (M)']= ta.sma(data['Close'], length= 75)
    data['SMA Close (S)']= ta.sma(data['Close'], length= 150)


    ## Exponential Moving Average
    data['EMA Close (F)']= ta.ema(data["Close"], length= 21)
    data['EMA Close (M)']= ta.ema(data['Close'], length= 75)
    data['EMA Close (S)']= ta.ema(data['Close'], length= 150)


    ## Double Exponential Moving Average
    data['DEMA Close (F)']= ta.dema(data["Close"], length= 21)
    data['DEMA Close (M)']= ta.dema(data['Close'], length= 75)
    data['DEMA Close (S)']= ta.dema(data['Close'], length= 150)


    ## Triple Exponential Movng Average
    data['TEMA Close (F)']= ta.tema(data["Close"], length= 21)
    data['TEMA Close (M)']= ta.tema(data['Close'], length= 75)
    data['TEMA Close (S)']= ta.tema(data['Close'], length= 150)


    ## Kaufman Adaptive Moving Average
    data['KAMA Close (F)']= ta.kama(data["Close"], length= 21)
    data['KAMA Close (M)']= ta.kama(data['Close'], length= 75)
    data['KAMA Close (S)']= ta.kama(data['Close'], length= 150)

    ## MACD
    macdClose= ta.macd(data['Close'])

    ## Bollinger Bands
    bbandsClose= ta.bbands(data['Close'])

    ## Momentum
    data['Momentum Close']= ta.mom(data['Close'])

    ## Relative Strength Index
    data['RSI Close']= ta.rsi(data['Close'], period= 14)

    ## True Range
    data['TrueRange']= ta.true_range(data['High'], data['Low'], data['Close'])

    ## Avrerage True Range
    data['AvgTrueRange']= ta.atr(data['High'], data['Low'], data['Close'])

    bbandsCloseNames= {'BBL_5_2.0' : 'CloseLower', 
                   'BBM_5_2.0' : 'CloseMid', 
                   'BBU_5_2.0' : 'CloseUpper', 
                   'BBB_5_2.0' : 'CloseBandwidth', 
                   'BBP_5_2.0' : 'Close%Columns'}

    for i in bbandsClose.columns:
        data[i]= bbandsClose[i]
    data.rename(columns= bbandsCloseNames, inplace= True)

    macdCloseNames= {'MACD_12_26_9': 'MACDClose', 
                 'MACDh_12_26_9': 'HistClose', 
                 'MACDs_12_26_9': 'SignalClose'}

    for i in macdClose.columns:
        data[i]= macdClose[i]
    data.rename(columns= macdCloseNames, inplace= True)

    ## Simple Return
    data['CloseR']= data['Close']/data['Close'].shift(-1) ## target feature we want to predict
    data['OpenR']= data['Open']/data['Open'].shift(-1)
    data['HighR']= data['High']/data['High'].shift(-1)
    data['LowR']= data['Low']/data['Low'].shift(-1)

    ## Lagged Return Calculations
    data['CloseR (2)']= data['Close']/data['Close'].shift(-2)
    data['OpenR (2)']= data['Open']/data['Open'].shift(-2)
    data['HighR (2)']= data['High']/data['High'].shift(-2)
    data['LowR (2)']= data['Low']/data['Low'].shift(-2)

    data['CloseR (3)']= data['Close']/data['Close'].shift(-3)
    data['OpenR (3)']= data['Open']/data['Open'].shift(-3)
    data['HighR (3)']= data['High']/data['High'].shift(-3)
    data['LowR (3)']= data['Low']/data['Low'].shift(-3)

    data['CloseR (2/3)']= data['Close'].shift(-2)/data['Close'].shift(-3)
    data['OpenR (2/3)']= data['Open'].shift(-2)/data['Open'].shift(-3)
    data['HighR (2/3)']= data['High'].shift(-2)/data['High'].shift(-3)
    data['LowR (2/3)']= data['Low'].shift(-2)/data['Low'].shift(-3)


    data['CloseR (4)']= data['Close']/data['Close'].shift(-4)
    data['OpenR (4)']= data['Open']/data['Open'].shift(-4)
    data['HighR (4)']= data['High']/data['High'].shift(-4)
    data['LowR (4)']= data['Low']/data['Low'].shift(-4)

    data['CloseR (5)']= data['Close']/data['Close'].shift(-5)
    data['OpenR (5)']= data['Open']/data['Open'].shift(-5)
    data['HighR (5)']= data['High']/data['High'].shift(-5)
    data['LowR (5)']= data['Low']/data['Low'].shift(-5)

    data['CloseR (4/5)']= data['Close'].shift(-4)/data['Close'].shift(-5)
    data['OpenR (4/5)']= data['Open'].shift(-4)/data['Open'].shift(-5)
    data['HighR (4/5)']= data['High'].shift(-4)/data['High'].shift(-5)
    data['LowR (4/5)']= data['Low'].shift(-4)/data['Low'].shift(-5)

    data['CloseR (15)']= data['Close']/data['Close'].shift(-15)
    data['OpenR (15)']= data['Open']/data['Open'].shift(-15)
    data['HighR (15)']= data['High']/data['High'].shift(-15)
    data['LowR (15)']= data['Low']/data['Low'].shift(-15)

    data['CloseR (30)']= data['Close']/data['Close'].shift(-30)
    data['OpenR (30)']= data['Open']/data['Open'].shift(-30)
    data['HighR (30)']= data['High']/data['High'].shift(-30)
    data['LowR (30)']= data['Low']/data['Low'].shift(-30)

    data['CloseR (15/30)']= data['Close'].shift(-15)/data['Close'].shift(-30)
    data['OpenR (15/30)']= data['Open'].shift(-15)/data['Open'].shift(-30)
    data['HighR (15/30)']= data['High'].shift(-15)/data['High'].shift(-30)
    data['LowR (15/30)']= data['Low'].shift(-15)/data['Low'].shift(-30)

    data['High/Open']= data['High']/data['Open']
    data['Low/Open']= data['Low']/data['Open']
    data['High/Close']= data['High']/data['Close']
    data['Low/Close']= data['Low']/data['Close']

    data['High/Open (1)']= data['High']/data['Open'].shift(-1)
    data['Low/Open (1)']= data['Low']/data['Open'].shift(-1)
    data['High/Close (1)']= data['High']/data['Close'].shift(-1)
    data['Low/Close (1)']= data['Low']/data['Close'].shift(-1)

    data['High/Open (2)']= data['High']/data['Open'].shift(-2)
    data['Low/Open (2)']= data['Low']/data['Open'].shift(-2)
    data['High/Close (2)']= data['High']/data['Close'].shift(-2)
    data['Low/Close (2)']= data['Low']/data['Close'].shift(-2)

    data['High/Open (1/2)']= data['High'].shift(-1)/data['Open'].shift(-2)
    data['Low/Open (1/2)']= data['Low'].shift(-1)/data['Open'].shift(-2)
    data['High/Close (1/2)']= data['High'].shift(-1)/data['Close'].shift(-2)
    data['Low/Close (1/2)']= data['Low'].shift(-1)/data['Close'].shift(-2)

    data['High/Open (3)']= data['High']/data['Open'].shift(-3)
    data['Low/Open (3)']= data['Low']/data['Open'].shift(-3)
    data['High/Close (3)']= data['High']/data['Close'].shift(-3)
    data['Low/Close (3)']= data['Low']/data['Close'].shift(-3)

    data['High/Open (2/3)']= data['High'].shift(-2)/data['Open'].shift(-3)
    data['Low/Open (2/3)']= data['Low'].shift(-2)/data['Open'].shift(-3)
    data['High/Close (2/3)']= data['High'].shift(-2)/data['Close'].shift(-3)
    data['Low/Close (2/3)']= data['Low'].shift(-2)/data['Close'].shift(-3)

    data['High/Open (15)']= data['High']/data['Open'].shift(-15)
    data['Low/Open (15)']= data['Low']/data['Open'].shift(-15)
    data['High/Close (15)']= data['High']/data['Close'].shift(-15)
    data['Low/Close (15)']= data['Low']/data['Close'].shift(-15)

    data['High/Open (30)']= data['High']/data['Open'].shift(-30)
    data['Low/Open (30)']= data['Low']/data['Open'].shift(-30)
    data['High/Close (30)']= data['High']/data['Close'].shift(-30)
    data['Low/Close (30)']= data['Low']/data['Close'].shift(-30)

    data['High/Open (15/30)']= data['High'].shift(-15)/data['Open'].shift(-30)
    data['Low/Open (15/30)']= data['Low'].shift(-15)/data['Open'].shift(-30)
    data['High/Close (15/30)']= data['High'].shift(15)/data['Close'].shift(-30)
    data['Low/Close (15/30)']= data['Low'].shift(-15)/data['Close'].shift(-30)

    data['High/Open (30/30)']= data['High'].shift(-30)/data['Open'].shift(-30)
    data['Low/Open (30/30)']= data['Low'].shift(-30)/data['Open'].shift(-30)
    data['High/Close (30/30)']= data['High'].shift(-30)/data['Close'].shift(-30)
    data['Low/Close (30/30)']= data['Low'].shift(-30)/data['Close'].shift(-30)

    data['High/Open (35/35)']= data['High'].shift(-35)/data['Open'].shift(-35)
    data['Low/Open (35/35)']= data['Low'].shift(-35)/data['Open'].shift(-35)
    data['High/Close (35/35)']= data['High'].shift(-35)/data['Close'].shift(-35)
    data['Low/Close (35/35)']= data['Low'].shift(-35)/data['Close'].shift(-35)

    data['High/Open (35/35)']= data['High'].shift(-35)/data['Open'].shift(-35)
    data['Low/Open (35/35)']= data['Low'].shift(-35)/data['Open'].shift(-35)
    data['High/Close (35/35)']= data['High'].shift(-35)/data['Close'].shift(-35)
    data['Low/Close (35/35)']= data['Low'].shift(-35)/data['Close'].shift(-35)

    data['High/Open (37/40)']= data['High'].shift(-37)/data['Open'].shift(-40)
    data['Low/Open (37/40)']= data['Low'].shift(-37)/data['Open'].shift(-40)
    data['High/Close (37/40)']= data['High'].shift(-37)/data['Close'].shift(-40)
    data['Low/Close (37/40)']= data['Low'].shift(-37)/data['Close'].shift(-40)

    data['High/Open (39/40)']= data['High'].shift(-39)/data['Open'].shift(-40)
    data['Low/Open (39/40)']= data['Low'].shift(-39)/data['Open'].shift(-40)
    data['High/Close (39/40)']= data['High'].shift(-39)/data['Close'].shift(-40)
    data['Low/Close (39/40)']= data['Low'].shift(-39)/data['Close'].shift(-40)

    data['High/Open (38/40)']= data['High'].shift(-38)/data['Open'].shift(-40)
    data['Low/Open (38/40)']= data['Low'].shift(-38)/data['Open'].shift(-40)
    data['High/Close (38/40)']= data['High'].shift(-38)/data['Close'].shift(-40)
    data['Low/Close (38/40)']= data['Low'].shift(-38)/data['Close'].shift(-40)

    data['High/Open (40/40)']= data['High'].shift(-40)/data['Open'].shift(-40)
    data['Low/Open (40/40)']= data['Low'].shift(-40)/data['Open'].shift(-40)
    data['High/Close (40/40)']= data['High'].shift(-40)/data['Close'].shift(-40)
    data['Low/Close (40/40)']= data['Low'].shift(-40)/data['Close'].shift(-40)

    ## Lagging Calculated Returns
    data['RetClose']= data['CloseR'].shift(-1)
    data['RetClose (2)']= data['CloseR'].shift(-2)
    data['RetClose (3)']= data['CloseR'].shift(-3)
    data['RetClose (5)']= data['CloseR'].shift(-5)
    data['RetClose (6)']= data['CloseR'].shift(-6)
    data['RetClose (15)']= data['CloseR'].shift(-15)
    data['RetClose (20)']= data['CloseR'].shift(-20)
    data['RetClose (30)']= data['CloseR'].shift(-30)
    data['RetClose (35)']= data['CloseR'].shift(-35)
    data['RetClose (40)']= data['CloseR'].shift(-40)


    data['RetLow']= data['LowR'].shift(-1)
    data['RetLow (2)']= data['LowR'].shift(-2)
    data['RetLow (3)']= data['LowR'].shift(-3)
    data['RetLow (5)']= data['LowR'].shift(-5)
    data['RetLow (6)']= data['LowR'].shift(-6)
    data['RetLow (15)']= data['LowR'].shift(-15)
    data['RetLow (20)']= data['LowR'].shift(-20)
    data['RetLow (30)']= data['LowR'].shift(-30)
    data['RetLow (35)']= data['LowR'].shift(-35)
    data['RetLow (40)']= data['LowR'].shift(-40)


    data['RetHigh']= data['HighR'].shift(-1)
    data['RetHigh (2)']= data['HighR'].shift(-2)
    data['RetHigh (3)']= data['HighR'].shift(-3)
    data['RetHigh (5)']= data['HighR'].shift(-5)
    data['RetHigh (6)']= data['HighR'].shift(-6)
    data['RetHigh (15)']= data['HighR'].shift(-15)
    data['RetHigh (20)']= data['HighR'].shift(-20)
    data['RetHigh (30)']= data['HighR'].shift(-30)
    data['RetHigh (35)']= data['HighR'].shift(-35)
    data['RetHigh (40)']= data['HighR'].shift(-40)


    data['RetLow']= data['LowR'].shift(-1)
    data['RetLow (2)']= data['LowR'].shift(-2)
    data['RetLow (3)']= data['LowR'].shift(-3)
    data['RetLow (5)']= data['LowR'].shift(-5)
    data['RetLow (6)']= data['LowR'].shift(-6)
    data['RetLow (15)']= data['LowR'].shift(-15)
    data['RetLow (20)']= data['LowR'].shift(-20)
    data['RetLow (30)']= data['LowR'].shift(-30)
    data['RetLow (35)']= data['LowR'].shift(-35)
    data['RetLow (40)']= data['LowR'].shift(-40)

    data['RetOpen']= data['OpenR'].shift(-1)
    data['RetOpen (2)']= data['OpenR'].shift(-2)
    data['RetOpen (3)']= data['OpenR'].shift(-3)
    data['RetOpen (5)']= data['OpenR'].shift(-5)
    data['RetOpen (6)']= data['OpenR'].shift(-6)
    data['RetOpen (15)']= data['OpenR'].shift(-15)
    data['RetOpen (20)']= data['OpenR'].shift(-20)
    data['RetOpen (30)']= data['OpenR'].shift(-30)
    data['RetOpen (35)']= data['OpenR'].shift(-35)
    data['RetOpen (40)']= data['OpenR'].shift(-40)

    data.drop(['Date','Open', 'High', 'Low', 'Close'], axis= 1, inplace= True)
    data.dropna(inplace= True)
    data.reset_index(inplace= True)
    data.drop(['index'], axis= 1, inplace= True)
    data['TargetCloseR']= data['CloseR'].shift(-30).fillna(0)

    print('CloseR Column:', data.columns.get_loc('CloseR'))
    print('DataFrame Shape:', data.shape)

    return data


def merge_sentiment_with_accumulation(stock_data, sentiment_data, symbol):
    # Convert date formats if necessary
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    
    # Filter sentiment data for the specific symbol
    sentiment_data = sentiment_data[sentiment_data['Symbol'] == symbol]
    
    # Prepare stock data - ensuring it's sorted by date
    stock_data.sort_values('Date', inplace=True)
    
    # Create a DataFrame to hold the date and cumulative sentiment score
    trading_days = set(stock_data['Date'])
    sentiment_data['is_trading_day'] = sentiment_data['date'].isin(trading_days)
    
    # Initialize variables to calculate cumulative sentiment scores
    cumulative_score = 0
    cumulative_scores = []
    
    # Iterate through the sentiment data
    for idx, row in sentiment_data.iterrows():
        if row['is_trading_day']:
            # Apply the accumulated sentiment score to the current trading day
            cumulative_score += row['sentiment_index']
            cumulative_scores.append((row['date'], cumulative_score))
            cumulative_score = 0  # Reset for next accumulation
        else:
            # Accumulate sentiment scores for non-trading days
            cumulative_score += row['sentiment_index']
    
    # Creating a new DataFrame from cumulative_scores
    adjusted_sentiment_data = pd.DataFrame(cumulative_scores, columns=['Date', 'cumulative_sentiment_index'])
    
    # Merge on date with stock data
    merged_data = pd.merge(stock_data, adjusted_sentiment_data, how='left', on='Date')
    
    # Fill NaNs in cumulative_sentiment_index with 0
    merged_data['cumulative_sentiment_index'].fillna(0, inplace=True)
    
    return merged_data


