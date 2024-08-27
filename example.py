import yfinance as yf

msft = yf.Ticker("E").info

# get all stock info


# show analysts data
print(msft)


import plotly.graph_objs as go

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.show()

print("This code runs immediately after the plot is shown.")
