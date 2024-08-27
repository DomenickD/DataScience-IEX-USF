import yfinance as yf
import streamlit as st
import pandas as pd

st.write(
    """ 
    # Simple stock price app example from freecodecamp

Shown are the stock **closing prices** and *volume of google*
- Markdown Table Example:
         
| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |
"""
)
# define the ticker symbol
tickerSymbol = "GOOGL"
# get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
# get historical data. period is 1 day. the dates need to labled too
tickerDF = tickerData.history(period="1d", start="2020-4-1", end="2023-4-1")
# dataframe will have
# open high low close volume dividends stock split
st.write(
    """
# Closing Price
"""
)
st.line_chart(tickerDF.Close)
st.write(
    """
# Volume
"""
)
st.line_chart(tickerDF.Volume)
