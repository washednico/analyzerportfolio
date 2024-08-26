import yfinance as yf

msft = yf.Ticker("MSFT").info

# get all stock info


# show analysts data
print(msft["targetMeanPrice"])
print(msft["targetLowPrice"])
print(msft["targetHighPrice"])
