"""
Intraday Volatility Forecasting Project

This project studies how market volatility evolves
during the trading day using high frequency (1min) SPY data.

Goals:
- Measure realized intraday volatility
- Forecast volatility using simple statistical models
- Compare different forecasting approaches used in practice
   --> future goal is to select and modify the most appropriate forecasting methods
       to improve accuracy and forecasting ability
- Visualize both historical and forward looking volatility paths

The focus is on understanding market volatility and attempt 
to create accurate forecasting of volatility.

Currently trying it on SPY, the S&P 500 index

"""



#importing libraries
import yfinance as yf       #to download market data directly from yahoo finance
import pandas as pd         # for data manipulation like tables and time series
import numpy as np          # for math operations like log, sqrt, arrays, etc
import matplotlib.pyplot as plt     # for plotting time series and forecasts


# Download 1-minute SPY data for last 7 trading days
ticker = "SPY"          #SPY is the SP&P 500 etf which people tend to favor as a representative of the stock market as a whole
data = yf.download(
    ticker,
    interval="1m",      #1 minute frequency
    period="7d",        # last 7 calendar days
    #progress = false disables download progress bar
    progress=False
)


#have to flatten multi-level columns from yahoo finance so it lines up later and makes it easier to use
data.columns = data.columns.get_level_values(0)

'''
we filter the yahoo finance data to just during trading hours (9:30 to 4pm)

this is because liquidity and trading volume is significantly higher during
trading hours and 
'''
# Convert index from UTC (which is default yahoo timezone) to US/Eastern
data.index = data.index.tz_convert("US/Eastern")

#keep only data between 930 and 1600 because those are regular trading hours and market is active in those hours
data = data.between_time("09:30", "16:00")

#quick check
print(data.head())      # show first 5 rows of data to check download
print("Number of rows:", len(data))
print("Date range:", data.index.min(), "to", data.index.max())

#Step 2:

""" 
Using log returns instead of simple percentage returns because:
- log returns are time additive so returns over mutliple intervals
can be summed instead of multiplied which is important since we're
getting realized volatility by aggregating returns so additivity ensures
the total volatility is consistent
- Plus, log returns are behave better 

Log return formula:

r_t = log(P_t) - log(P_(t-1))
    ---> r_t = return at time t (price change from previous time step)
    ---> P_t = asset price at current time t
    ---> P_(t-1) = asset price at previous time t-1
    natural log base e


Volatility is the standard dev of these returns

we use this because squaring returns removes the direction of price
movement and caputres only the magnitude of variation which gives us the volatility
since volatility is the intensity of price flunctuations rather than whether prices go up or down

"""

# Compute log returns by taking natural log of price differences
data["log_return"] = np.log(data["Close"]).diff()

# Drop the first row NaN created by diff
data = data.copy()

# define rolling windows
WINDOW = 60      # past 60 minutes for rolling forecast
HORIZON = 60     # next 60 minutes for realized volatility

# Step 3: Realized volatility

"""
Realized volatility measures how volatile the market actually was

we compute it as the rolling standard deviation of log returns
shifted backwards so it represents future volatility

realized volatility is a good data driven estimate of actual market volatiltiy

this allows us to compare forecasts made at time t with what
actually happened after time t

also we use a rolling window to estimate volatility because we want
the model to adapt as new data is available and discard less relevant older data

calculated as:
sigma_t = sqrt((1/(N-1)) * sum from i = t -N +1 to t of (r_i-r_mean)^2)
    ---> sigma_t = realized volatility at time t
    ---> N = num of past observations in rolling window
    ---> r_i = log return at time i
    ---> r_mean ---> avg of log returns in the rolling window

    
we shift the window backwards so it represents the future volatility for comparison with forecasts
this allows us to check model accuracy
"""

data["realized_vol"] = (
    data["log_return"]          # take log returns
    .rolling(HORIZON)           # compute rolling stdev over next horizon minute
    .std()                      # stdev
    .shift(-HORIZON) # align realized volatility with forecast start time
)


#Step 4: Rolling volatility forecast (PAST)
'''
Simple Rolling Forecast
- uses the past window minutes of log returns
- computes standard deviation
- assumes all past returns are equally important so it doesn't emphasize recent shocks
'''


data["rolling_vol_forecast"] = (
    data["log_return"]
    .rolling(WINDOW)
    .std()
)



# Step 5: EWMA volatility forecast

'''
EWMA (exponentially weighted moving average)

Thhe idea:
- recent returns matter more than older returns
--> this keeps the forecasts more relevant to current
state rather than old data
- Volatility is persistent but mean-reverting so it doesn't grow infinetly and eventually goes back to baseline mean
- Volatility is persistent:
    --> basically periods of high volatility tend to be followed by high volatility
        --> and low volatility by low volatiltiy


Formula:
(sigma_t)^2 = lambda * (sigma_(t-1))^2 + (1 - lambda) * r_t^2
    ---> (sigma_t)^2 = forecasted variance at time t
    ---> (sigma_(t-1))^2 = variance of previous time t-1
    ---> (r_t)^2 ---> squared return at time t
    ----> lambda --> decay factor

We then take sqrt to convert variance to stdev (volatility)

'''
#(tested different lambdas but settled on .85 for now. Later want to create test to find most appropriate lambda)
lambda_ = 0.85 # decay factor (weight for past variance)

ewma_var = []       # list to store EWMA veriance at each time step

prev_var = data["log_return"].dropna().var() #initialize with overall variance

for r in data["log_return"]:        # loop through each return
    if np.isnan(r):                 # skip missing values
        ewma_var.append(np.nan)
    else:
        #update EWMA variance formula
        prev_var = lambda_ * prev_var + (1 - lambda_) * (r ** 2)
        ewma_var.append(prev_var)

# convert variance to stdev (volatility)
data["ewma_vol_forecast"] = np.sqrt(ewma_var)

# =========================
# Step 5b: HAR volatility features
# =========================

'''
HAR Model is Heterogeneous Autoregressive Volatility

we use multiple time horizons to capture market dynamics since different traders trade on different time horizons
    - short term 30 min
    - medium term 2h
    - long term 1 trading day
    - Predicts future volatility as a weighted sum of past volatility


model predicts future volatility using past volatility
measured at multiple time scales
    ---> by measuring volatility at different levels (time horizons), the model captures
        both short term market moves and long term market trends

estimate model using ordinary least squares which finds oefficients that minimize the sum of
squared forecast errors between forecated and realized volatility.

equation:
RV_t = beta_0 + beta_1 * RV_(t-1) + beta_2 * (RV_(t-1))^(w) + beta_3 * (RV_(t-1))^(m) + error
- RV_t = realized volatility at time t
- RV_(t-1) = previous period volatility (short-term)
- RV_(t-1)^w = medium-term average volatility
- RV_(t-1)^m = long-term average volatility
- beta_0 = baseline volatility
- beta_1...3 = regression weights
- error = unexplained volatility

'''


# Compute rolling stdev over multiple time horizons
data["vol_30m"] = data["log_return"].rolling(30).std()     # short term (30 min)
data["vol_2h"]  = data["log_return"].rolling(120).std()    # medium term (120 min = 2h)
data["vol_1d"]  = data["log_return"].rolling(390).std()    # long term (full trade day)

# Log-transform realized volatility for regression
data["log_realized_vol"] = np.log(data["realized_vol"])


#Step 6:

# drop rows with missing values in features or target
har_data = data.dropna(subset=[
    "log_realized_vol",
    "vol_30m",
    "vol_2h",
    "vol_1d"
])

# define X and y for regression
X = har_data[["vol_30m", "vol_2h", "vol_1d"]]   # independent variables
y = har_data["log_realized_vol"]                # dependent variable

# =========================
# Manual OLS for HAR model
# =========================

'''
OLS (ordinary least squares) regression:   
    - finds coefficients (betas) that minimize squared errors
    - formula (matrix form):


Beta_hat = inverse(X_transpose *X) * X_transpose * y
    ---> X = matrix of input variables (volatility values)
    ---> y = vector of realized volatility
    ---> X_transpose = transpose of X
    ---> inverse(X transpose* X) = inverse matrix
    ---> Beta_hat = estimated regression coefficients

Each beta tells how much each horizon contributes to predicting future volatilty
'''

# Add intercept column of ones
X_mat = np.column_stack([
    np.ones(len(X)),        # beta_0 intercept
    X["vol_30m"].values,    # beta_1
    X["vol_2h"].values,     # beta_2
    X["vol_1d"].values      # beta_3
])

# reshape y to column vector for matrix multiplication
y_mat = y.values.reshape(-1, 1)

# solve for beta using closed-form ols formula
beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_mat

# predict log-volatility
har_log_vol_pred = X_mat @ beta
har_data["har_log_vol_forecast"] = har_log_vol_pred.flatten()       # flatten converts 2D array to 1D
har_data["har_vol_forecast"] = np.exp(har_data["har_log_vol_forecast"])     # reverts log transform

# compute HAR forecast error MAE (Mean Absolute Error)
har_mae = np.mean(
    np.abs(har_data["har_vol_forecast"] - har_data["realized_vol"])
)

print("HAR MAE:", har_mae)

plt.figure(figsize=(12,6))              # set figure size
plt.plot(har_data.index, har_data["realized_vol"] * 100, label="Realized Vol", alpha=0.6)     # alpha = transparency
plt.plot(har_data.index, har_data["har_vol_forecast"] * 100, label="HAR Forecast", alpha=0.8)

plt.legend()                            # show legend
plt.title("HAR Volatility Forecast vs Realized Volatility")  #title
plt.xlabel("Time")                      # Label x-axis
plt.ylabel("Volatility (% per minute)")                # Label y-axis
plt.show()                              # display plot

har_data.tail(300)[[
    "realized_vol",
    "har_vol_forecast"
]].plot(figsize=(12,6), title="Zoomed HAR Volatility Forecast")
plt.show()


comparison = data.dropna(subset=[
    "realized_vol",
    "rolling_vol_forecast",
    "ewma_vol_forecast"
])


#Step 7:

rolling_mae = np.mean(
    np.abs(comparison["rolling_vol_forecast"] - comparison["realized_vol"])
)

ewma_mae = np.mean(
    np.abs(comparison["ewma_vol_forecast"] - comparison["realized_vol"])
)

print("Rolling MAE:", rolling_mae)
print("EWMA MAE:", ewma_mae)


#Step 8:

plt.figure(figsize=(12,6))
plt.plot(comparison.index, comparison["realized_vol"], label="Realized Vol", alpha=0.7)
plt.plot(comparison.index, comparison["rolling_vol_forecast"], label="Rolling Forecast", alpha=0.7)
plt.plot(comparison.index, comparison["ewma_vol_forecast"], label="EWMA Forecast", alpha=0.7)

plt.legend()
plt.title("Intraday Volatility: Realized vs Forecasted")
plt.show()

#(Plot 2: Zoomed in sample):

#last 300 minutes zoom in
comparison.tail(300)[[
    "realized_vol",
    "rolling_vol_forecast",
    "ewma_vol_forecast"
]].plot(figsize=(12,6), title="Zoomed-In Volatility Comparison")
plt.show()


HORIZON = 20  
zoom = comparison.iloc[-HORIZON*5:]  

plt.figure(figsize=(12,6))
plt.plot(zoom.index, zoom["realized_vol"] * 100, label="Realized Vol", alpha=0.7)
plt.plot(zoom.index, zoom["rolling_vol_forecast"] * 100, label="Rolling Forecast", alpha=0.7)
plt.plot(zoom.index, zoom["ewma_vol_forecast"] * 100, label="EWMA Forecast", alpha=0.7)

plt.legend()
plt.title(f"Zoomed Intraday Volatility Forecast (~{HORIZON*1*5} (200) minutes shown)")
plt.xlabel("Time")
plt.ylabel("Volatility (% per minute)")
plt.show()


print("Evaluation rows:", len(comparison))
print(comparison.head())

# =========================
# Step 9: Forward volatility path (EWMA persistence)
# =========================

'''
so this section is to project volatility forward from the current moment until market close

Forward path projection:

forecasts volatility for remaining minutes until market close

assumes volatility is persistent and since it forecasts ahead, the forecast
decays the further the forecast is from current time
    -->basically periods of high volatility tend to be followed by high volatility
        --> and low volatility by low volatiltiy
    --> however volatility does not grow endlessly, and eventually returns to a long run average level
        ---> so a model should gradually adjust towards equilibrium


could be used for short term risk monitoring
'''


# Assume we're at "now" so last observed EWMA volatility
current_vol = data["ewma_vol_forecast"].dropna().iloc[-1]

lambda_ = 0.85 #ewma decay factor

# compute remaining minutes until market close
last_time = data.index[-1]              # last timestamp
market_close = last_time.normalize() + pd.Timedelta(hours=16)   #1600

minutes_remaining = int((market_close - last_time).total_seconds() / 60)

if minutes_remaining > 0:

    horizon = minutes_remaining
    future_vol = []  #store forward forecast
    prev = current_vol

    #loop forward to simulate volatility decay returning to mean
    for _ in range(horizon):
        # EWMA persistence (variance form)
        prev = np.sqrt(lambda_ * prev**2 + (1 - lambda_) * prev**2)
        future_vol.append(prev)

    #create datetime index for future minutes
    future_index = pd.date_range(
        start=last_time + pd.Timedelta(minutes=1),
        periods=horizon,
        freq="1min"
    )

    # Plot forward path
    plt.figure(figsize=(12,6))
    plt.plot(
        data.index[-60:],
        data["ewma_vol_forecast"].iloc[-60:],
        label="Current EWMA Forecast"
    )
    plt.plot(
        future_index,
        future_vol,
        linestyle="--",
        label="Forward Volatility Path"
    )

    plt.axvline(last_time, color="gray", linestyle=":")
    plt.legend()
    plt.title("Forward Intraday Volatility Path (EWMA)")
    plt.xlabel("Time")
    plt.ylabel("Volatility (% per minute)")
    plt.show()


