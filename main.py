import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import cvxpy as cp
import warnings
from textblob import TextBlob
import datetime
from pygooglenews import GoogleNews

# Suppress specific warnings while maintaining important ones
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Global parameters
TRADING_DAYS = 252
MAX_FORECAST = float('inf')  # Allow unlimited upside potential
MIN_FORECAST = -0.95  # Minimum allowed forecast return (-95%)

# -------------------------------
# Interactive Input Module
# -------------------------------
def get_user_inputs():
    """
    Get user inputs for tickers and custom news search terms.
    Returns a dictionary mapping tickers to their custom search terms.
    """
    print("\n=== Interactive Trading System V6 ===")
    print("Enter your stock tickers separated by commas (e.g., AAPL,MSFT,GOOG):")
    ticker_input = input().strip()
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    if not tickers:
        raise ValueError("No valid tickers provided")
    
    ticker_news_terms = {}
    print("\nFor each ticker, you can specify custom news search terms to improve sentiment analysis.")
    print("For cryptocurrencies, consider using terms like 'bitcoin price', 'crypto market', 'blockchain'.")
    print("For stocks, consider using terms like 'company earnings', 'market analysis'.")
    print("Press Enter to use optimized default search, or enter your custom terms.")
    
    for ticker in tickers:
        # Detect if it's a crypto ticker
        is_crypto = '-USD' in ticker
        default_terms = f"{ticker.replace('-USD', '')} {('cryptocurrency' if is_crypto else 'stock')}"
        
        print(f"\nEnter search terms for {ticker} (or press Enter to use: '{default_terms}'):")
        search_terms = input().strip()
        ticker_news_terms[ticker] = search_terms if search_terms else default_terms
    
    print("\nEnter the start date (YYYY-MM-DD):")
    start_date = input().strip() or '2023-01-01'
    
    print("\nEnter the end date (YYYY-MM-DD):")
    end_date = input().strip() or datetime.datetime.now().strftime('%Y-%m-%d')
    
    print("\nEnter investment capital (default: 100000):")
    capital_input = input().strip()
    capital = float(capital_input) if capital_input else 100000
    
    print("\nEnter risk aversion factor (0.1-1.0, default: 0.5):")
    risk_input = input().strip()
    risk_aversion = float(risk_input) if risk_input else 0.5
    
    return {
        'tickers': tickers,
        'ticker_news_terms': ticker_news_terms,
        'start_date': start_date,
        'end_date': end_date,
        'capital': capital,
        'risk_aversion': risk_aversion
    }

# -------------------------------
# Enhanced Sentiment Analysis Module with Custom Search Terms
# -------------------------------
def get_enhanced_sentiment_score(ticker, search_term, days_lookback=7):
    """
    Get sentiment score using custom search terms from multiple news sources.
    Returns a score between -1 (very negative) and 1 (very positive).
    """
    try:
        # Initialize sentiment collectors
        yf_sentiments = []
        gn_sentiments = []
        
        # 1. Get yfinance news sentiment
        stock = yf.Ticker(ticker)
        yf_news = stock.news
        
        if yf_news:
            for article in yf_news:
                text = article.get('title', '')
                if article.get('summary'):
                    text += ' ' + article['summary']
                
                if text:
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity
                    pub_date = datetime.datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    days_old = (datetime.datetime.now() - pub_date).days
                    if days_old <= days_lookback:
                        weight = 1.0 - (days_old / days_lookback)
                        yf_sentiments.append(sentiment * weight)
        
        # 2. Get Google News sentiment with custom search term
        gn = GoogleNews(lang='en', country='US')
        
        # Split search terms and create multiple searches
        search_terms = [term.strip() for term in search_term.split(',')]
        for term in search_terms:
            # Add ticker to each search term for better relevance
            full_term = f"{ticker} {term}"
            search = gn.search(full_term)
            
            if search and 'entries' in search:
                for item in search['entries'][:5]:  # Process top 5 news items per term
                    text = item.get('title', '')
                    if item.get('summary'):
                        text += ' ' + item['summary']
                    
                    if text:
                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        try:
                            pub_date = datetime.datetime.strptime(item['published'], '%a, %d %b %Y %H:%M:%S %Z')
                            days_old = (datetime.datetime.now() - pub_date).days
                            if days_old <= days_lookback:
                                weight = 1.0 - (days_old / days_lookback)
                                gn_sentiments.append(sentiment * weight)
                        except (ValueError, KeyError):
                            # If date parsing fails, still include the sentiment with neutral weight
                            gn_sentiments.append(sentiment * 0.5)
        
        # Combine sentiments with dynamic source weighting
        is_crypto = '-USD' in ticker
        if is_crypto:
            yf_weight = 0.3  # Less weight to yfinance for crypto
            gn_weight = 0.7  # More weight to Google News for crypto
        else:
            yf_weight = 0.6  # More weight to yfinance for stocks
            gn_weight = 0.4  # Less weight to Google News for stocks
        
        weighted_sentiment = 0.0
        if yf_sentiments and gn_sentiments:
            weighted_sentiment = (yf_weight * np.mean(yf_sentiments) +
                                gn_weight * np.mean(gn_sentiments))
        elif yf_sentiments:
            weighted_sentiment = np.mean(yf_sentiments)
        elif gn_sentiments:
            weighted_sentiment = np.mean(gn_sentiments)
            
        # Apply stronger sentiment impact for crypto assets
        if is_crypto:
            weighted_sentiment *= 1.2  # Increase sentiment impact for crypto
        
        print(f"Found {len(yf_sentiments)} YF and {len(gn_sentiments)} GN news items for {ticker}")
        print(f"Search terms: {', '.join(search_terms)}")
        print(f"Combined sentiment: {weighted_sentiment:.3f}")
        return weighted_sentiment
    
    except Exception as e:
        print(f"Error in enhanced sentiment analysis for {ticker}: {e}")
        return 0.0

# -------------------------------
# Enhanced ARIMA Module
# -------------------------------
def arima_forecast_with_error(prices, order=(5, 1, 0)):
    """
    Enhanced ARIMA forecast with error estimation and validation.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) < max(order) + 1:
        raise ValueError("Not enough data points for the specified ARIMA order")
    
    if prices.index.freq is None:
        inferred_freq = pd.infer_freq(prices.index)
        prices = prices.asfreq(inferred_freq) if inferred_freq else prices.asfreq("B")
    
    try:
        model = ARIMA(prices, order=order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit = model.fit()
        
        # Calculate in-sample error
        fitted_values = model_fit.fittedvalues
        mse = mean_squared_error(prices[model_fit.loglikelihood_burn:], 
                               fitted_values[-len(prices[model_fit.loglikelihood_burn:]):])
        
        forecast_price = model_fit.forecast(steps=1)[0]
        last_price = prices.iloc[-1]
        
        if not np.isfinite(forecast_price) or forecast_price <= 0:
            print("Warning: Invalid ARIMA forecast price. Using last price.")
            return 0.0, np.inf
        
        daily_return = (forecast_price / last_price) - 1
        annual_return = daily_return * TRADING_DAYS
        annual_return = np.clip(annual_return, MIN_FORECAST, MAX_FORECAST)
        
        print(f"ARIMA forecast: Last price = {last_price:.2f}, Forecast = {forecast_price:.2f}, "
              f"Daily return = {daily_return:.2%}, Annual = {annual_return:.2%}, MSE = {mse:.6f}")
        return annual_return, mse
    
    except Exception as e:
        print(f"Error in ARIMA forecast: {e}")
        return 0.0, np.inf

# -------------------------------
# Enhanced ML Forecast Module
# -------------------------------
def ml_forecast_with_cv(prices, lag=10, n_splits=5):
    """
    Enhanced ML forecast with cross-validation and error estimation.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if len(prices) < lag + 2:
        raise ValueError(f"Need at least {lag + 2} data points for regression")
    
    returns = prices.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(returns.mean())
    
    X, y = [], []
    for i in range(lag, len(returns)):
        X.append(returns.iloc[i-lag:i].values)
        y.append(returns.iloc[i])
    X = np.array(X)
    y = np.array(y)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    try:
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores.append(mean_squared_error(y_test, y_pred))
        
        # Final prediction
        final_model = LinearRegression()
        final_model.fit(X, y)
        X_pred = returns.iloc[-lag:].values.reshape(1, -1)
        daily_return_pred = final_model.predict(X_pred)[0]
        
        if not np.isfinite(daily_return_pred):
            print("Warning: Invalid ML prediction. Using zero forecast.")
            return 0.0, np.inf
        
        annual_return = daily_return_pred * TRADING_DAYS
        annual_return = np.clip(annual_return, MIN_FORECAST, MAX_FORECAST)
        avg_mse = np.mean(cv_scores)
        
        print(f"ML forecast: Daily return = {daily_return_pred:.2%}, "
              f"Annual = {annual_return:.2%}, Avg CV MSE = {avg_mse:.6f}")
        return annual_return, avg_mse
    
    except Exception as e:
        print(f"Error in ML forecast: {e}")
        return 0.0, np.inf

# -------------------------------
# Portfolio Optimization Module
# -------------------------------
def portfolio_optimization(mu, covariance, risk_aversion, max_weight=0.4):
    """
    Enhanced portfolio optimization with position limits and improved error handling.
    """
    if not isinstance(mu, np.ndarray) or not isinstance(covariance, np.ndarray):
        raise TypeError("mu and covariance must be numpy arrays")
    if len(mu) != len(covariance):
        raise ValueError("Dimension mismatch between mu and covariance")
    
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu.T @ w - (risk_aversion / 2) * cp.quad_form(w, covariance))
    
    # Enhanced constraints including position limits
    constraints = [
        cp.sum(w) == 1,  # Full investment
        w >= 0,          # Long only
        w <= max_weight  # Position limit
    ]
    
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        
        if w.value is None:
            print("Warning: Optimization did not converge; using equal weights.")
            return np.ones(n) / n
        
        weights = np.array(w.value).flatten()
        weights = np.clip(weights, 0, None)
        weights = weights / np.sum(weights)
        return weights
    
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        return np.ones(n) / n

# -------------------------------
# Risk Management Module
# -------------------------------
def calculate_volatility_regime(returns, lookback=63):
    """
    Determine the current volatility regime using rolling volatility.
    """
    if len(returns) < lookback:
        return "normal"
    
    current_vol = returns.tail(lookback).std() * np.sqrt(TRADING_DAYS)
    historical_vol = returns.std() * np.sqrt(TRADING_DAYS)
    
    if current_vol > 1.5 * historical_vol:
        return "high"
    elif current_vol < 0.5 * historical_vol:
        return "low"
    return "normal"

def dynamic_risk_management(portfolio_return, risk_free_rate, portfolio_variance,
                          capital, returns, target_allocation=0.10):
    """
    Enhanced risk management with dynamic position sizing based on volatility regime.
    """
    try:
        # Basic Kelly calculation
        kelly_fraction = (portfolio_return - risk_free_rate) / portfolio_variance
        kelly_fraction = np.clip(kelly_fraction, 0.0, 1.0)
        
        # Adjust based on volatility regime
        vol_regime = calculate_volatility_regime(returns)
        regime_adjustments = {
            "low": 1.2,    # Increase allocation in low vol
            "normal": 1.0,  # Normal allocation
            "high": 0.5     # Reduce allocation in high vol
        }
        
        adjusted_kelly = kelly_fraction * regime_adjustments[vol_regime]
        allocation = min(adjusted_kelly, target_allocation)
        dollar_allocation = allocation * capital
        
        print(f"Volatility regime: {vol_regime}, Adjustment: {regime_adjustments[vol_regime]:.2f}x")
        return allocation, dollar_allocation
    
    except Exception as e:
        print(f"Risk management error: {e}")
        return 0.0, 0.0

# -------------------------------
# Integrated Trading System V6
# -------------------------------
def integrated_trading_system_v6():
    """
    Interactive trading system with custom news search terms and user inputs.
    """
    try:
        # Get user inputs
        inputs = get_user_inputs()
        tickers = inputs['tickers']
        ticker_news_terms = inputs['ticker_news_terms']
        start_date = inputs['start_date']
        end_date = inputs['end_date']
        capital = inputs['capital']
        risk_aversion = inputs['risk_aversion']
        
        print("\nInitiating analysis...\n")
        
        # Data download and preprocessing
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data downloaded")
        
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' not in data.columns.get_level_values('Price'):
                raise KeyError("No Close price data found")
            close_prices = data.xs('Close', axis=1, level='Price')
        else:
            close_prices = data['Close']
        
        close_prices.index = pd.to_datetime(close_prices.index)
        close_prices = close_prices.asfreq('B').fillna(method='ffill')
        
        # Initialize storage for results
        arima_returns = {}
        ml_returns = {}
        forecast_errors = {}
        sentiment_scores = {}
        
        # Analysis per ticker
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            if ticker not in close_prices.columns:
                print(f"Warning: {ticker} not found in data")
                continue
            
            prices = close_prices[ticker].dropna()
            
            # Get forecasts
            arima_ret, arima_error = arima_forecast_with_error(prices)
            ml_ret, ml_error = ml_forecast_with_cv(prices)
            
            # Get sentiment with custom search term
            sentiment = get_enhanced_sentiment_score(ticker, ticker_news_terms[ticker])
            sentiment_scores[ticker] = sentiment
            
            if np.isfinite(arima_ret) and np.isfinite(ml_ret):
                arima_returns[ticker] = arima_ret
                ml_returns[ticker] = ml_ret
                forecast_errors[ticker] = {
                    'arima': arima_error,
                    'ml': ml_error
                }
        
        # Combine forecasts with sentiment
        combined_returns = {}
        for ticker in arima_returns:
            if ticker in ml_returns:
                total_error = forecast_errors[ticker]['arima'] + forecast_errors[ticker]['ml']
                if total_error > 0:
                    arima_weight = forecast_errors[ticker]['ml'] / total_error
                    ml_weight = forecast_errors[ticker]['arima'] / total_error
                else:
                    arima_weight = ml_weight = 0.5
                
                base_return = (arima_weight * arima_returns[ticker] + 
                              ml_weight * ml_returns[ticker])
                sentiment_adj = 1.0 + (0.3 * sentiment_scores[ticker])
                combined_returns[ticker] = base_return * sentiment_adj
        
        # Portfolio optimization
        tickers_used = sorted(combined_returns.keys())
        mu = np.array([combined_returns[tkr] for tkr in tickers_used])
        returns_df = close_prices[tickers_used].pct_change().dropna()
        covariance_daily = returns_df.cov().values
        covariance_annual = covariance_daily * TRADING_DAYS
        
        weights = portfolio_optimization(mu, covariance_annual, risk_aversion)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_variance = np.dot(weights, np.dot(covariance_annual, weights))
        
        # Risk management
        risk_free_rate = 0.03
        allocation, dollar_allocation = dynamic_risk_management(
            portfolio_return, risk_free_rate, portfolio_variance,
            capital, returns_df.mean(axis=1)
        )
        
        # Print results
        print("\n=== Analysis Results ===")
        print("\nOptimized Portfolio Weights:")
        for tkr, wt in zip(tickers_used, weights):
            print(f"  {tkr}: {wt:.2%}")
        
        print(f"\nPortfolio Metrics:")
        print(f"Expected Return: {portfolio_return:.2%}")
        print(f"Annual Variance: {portfolio_variance:.2%}")
        print(f"Risk-Managed Allocation: {allocation:.2%} (${dollar_allocation:,.2f})")
        
        print("\nDetailed Sentiment Analysis:")
        for ticker in sentiment_scores:
            sentiment = sentiment_scores[ticker]
            sentiment_adj = 1.0 + (0.3 * sentiment)
            print(f"\n{ticker}:")
            print(f"  Search Term: {ticker_news_terms[ticker]}")
            print(f"  Sentiment Score: {sentiment:.3f}")
            print(f"  Impact: {(sentiment_adj - 1) * 100:.1f}% {'increase' if sentiment > 0 else 'decrease'}")
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        close_prices[tickers_used].plot(ax=ax1)
        ax1.set_title("Historical Closing Prices")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        
        returns = close_prices[tickers_used].pct_change()
        vol = returns.rolling(window=21).std() * np.sqrt(252)
        vol.plot(ax=ax2)
        ax2.set_title("21-Day Rolling Volatility (Annualized)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility")
        
        plt.tight_layout()
        plt.show()
        
        return {
            'tickers': tickers,
            'search_terms': ticker_news_terms,
            'sentiment_scores': sentiment_scores,
            'portfolio_weights': dict(zip(tickers_used, weights)),
            'portfolio_return': portfolio_return,
            'portfolio_variance': portfolio_variance,
            'allocation': allocation,
            'dollar_allocation': dollar_allocation
        }
        
    except Exception as e:
        print(f"\nError in analysis: {e}")
        return None

if __name__ == "__main__":
    integrated_trading_system_v6()