# Linear Trading Model

An advanced quantitative trading system that combines ARIMA forecasting, machine learning predictions, sentiment analysis, and portfolio optimization to generate investment recommendations.

## Features

- **Multi-Model Forecasting**
  - ARIMA time series forecasting with MSE evaluation
  - Machine Learning predictions with cross-validation
  - Combined forecast approach for robust predictions

- **Sentiment Analysis**
  - Integration with Google News API
  - Customizable search terms per asset
  - Sentiment impact quantification

- **Portfolio Optimization**
  - Risk-adjusted return optimization
  - Dynamic weight allocation
  - Volatility regime detection

- **Risk Management**
  - Configurable risk aversion settings
  - Position sizing based on Kelly Criterion
  - Volatility-based adjustments

## Prerequisites

- Python 3.7+
- Required packages:
  - yfinance
  - statsmodels
  - scikit-learn
  - pandas
  - numpy
  - pygooglenews

## Installation

1. Install all required dependencies:
   ```bash
   pip install numpy pandas yfinance matplotlib statsmodels scikit-learn cvxpy textblob pygooglenews --upgrade
   ```

   Note: Some packages might require additional system dependencies. If you encounter any issues, please refer to the respective package documentation.

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. Follow the interactive prompts:
   - Enter stock tickers (comma-separated)
   - Customize news search terms for each asset
   - Specify analysis timeframe
   - Set investment capital and risk parameters

## Example Output

```
=== Linear Model Trading System ===
Enter your stock tickers separated by commas (e.g., AAPL,MSFT,GOOG):
AAPL, MSFT, GOOG, NVDA      

For each ticker, you can specify custom news search terms to improve sentiment analysis.
For cryptocurrencies, consider using terms like 'bitcoin price', 'crypto market', 'blockchain'.
For stocks, consider using terms like 'company earnings', 'market analysis'.
Press Enter to use optimized default search, or enter your custom terms.

Enter search terms for AAPL (or press Enter to use: 'AAPL stock'):
FINANCIAL, STOCKS, BULLS, SP500

Enter search terms for MSFT (or press Enter to use: 'MSFT stock'):
TECH, AI, OPENAI

Enter search terms for GOOG (or press Enter to use: 'GOOG stock'):
GEMINI, GOOGLE CLOUD, 

Enter search terms for NVDA (or press Enter to use: 'NVDA stock'):
CHIP, STOCKS, GRAPHIC CARDS, JETSON, AI

Enter the start date (YYYY-MM-DD):
2022-01-01

Enter the end date (YYYY-MM-DD):
2025-02-19

Enter investment capital (default: 100000):
1000

Enter risk aversion factor (0.1-1.0, default: 0.5):
0.5

Initiating analysis...

Analyzing AAPL...
ARIMA forecast: Last price = 244.47, Forecast = 244.59, Daily return = 0.05%, Annual = 12.06%, MSE = 8.148327
ML forecast: Daily return = 0.25%, Annual = 63.44%, Avg CV MSE = 0.000259
Found 0 YF and 15 GN news items for AAPL
Search terms: FINANCIAL, STOCKS, BULLS, SP500
Combined sentiment: 0.057

Analyzing MSFT...
ARIMA forecast: Last price = 408.82, Forecast = 408.73, Daily return = -0.02%, Annual = -5.37%, MSE = 27.281479
ML forecast: Daily return = 0.10%, Annual = 26.30%, Avg CV MSE = 0.000262
Found 0 YF and 14 GN news items for MSFT
Search terms: TECH, AI, OPENAI
Combined sentiment: 0.007

Analyzing GOOG...
ARIMA forecast: Last price = 185.80, Forecast = 185.93, Daily return = 0.07%, Annual = 17.51%, MSE = 7.081569
ML forecast: Daily return = 0.40%, Annual = 101.81%, Avg CV MSE = 0.000394
Found 0 YF and 9 GN news items for GOOG
Search terms: GEMINI, GOOGLE CLOUD, 
Combined sentiment: 0.082

Analyzing NVDA...
ARIMA forecast: Last price = 139.40, Forecast = 139.39, Daily return = -0.01%, Annual = -2.60%, MSE = 5.716904
ML forecast: Daily return = 0.23%, Annual = 58.66%, Avg CV MSE = 0.001135
Found 0 YF and 19 GN news items for NVDA
Search terms: CHIP, STOCKS, GRAPHIC CARDS, JETSON, AI
Combined sentiment: 0.058
Volatility regime: normal, Adjustment: 1.00x

=== Analysis Results ===

Optimized Portfolio Weights:
  AAPL: 40.00%
  GOOG: 40.00%
  MSFT: 0.00%
  NVDA: 20.00%

Portfolio Metrics:
Expected Return: 79.64%
Annual Variance: 8.38%
Risk-Managed Allocation: 10.00% ($100.00)

Detailed Sentiment Analysis:

AAPL:
  Search Term: AAPL stock
  Sentiment Score: 0.076
  Impact: 2.3% increase

MSFT:
  Search Term: MSFT stock
  Sentiment Score: 0.076
  Impact: 2.3% increase

GOOG:
  Search Term: GOOG stock
  Sentiment Score: 0.083
  Impact: 2.5% increase

NVDA:
  Search Term: NVDA stock
  Sentiment Score: 0.064
  Impact: 1.9% increase

## Visualization
```
![Image](https://github.com/user-attachments/assets/6e207636-f2a8-4d00-a516-24051f3185d9)

```
The system generates two key visualizations to help understand market dynamics:

1. **Historical Closing Prices (Top)**:
   - Shows the price evolution of all assets in the portfolio
   - Helps identify trends, patterns, and relative performance
   - Each asset is represented by a different color line

2. **21-Day Rolling Volatility (Bottom)**:
   - Displays annualized volatility calculated on a 21-day rolling window
   - Helps identify periods of market stress or stability
   - Used by the risk management module for position sizing
   - Higher values indicate increased market uncertainty

## Output Interpretation

- **ARIMA Forecast**: Shows price predictions based on time series analysis
- **ML Forecast**: Provides machine learning-based return predictions
- **Sentiment Analysis**: Measures news sentiment impact on each asset
- **Portfolio Weights**: Suggests optimal asset allocation
- **Risk Metrics**: Includes volatility assessment and position sizing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
