from turtle import color
from click import style
import dash
from dash import dcc
from dash import html
from matplotlib.axis import XAxis
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import random 
# app = dash.Dash()
# server = app.server
# scaler=MinMaxScaler(feature_range=(0,1))
# df_nse = pd.read_csv("./NSE-TATA.csv")
# df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
# df_nse.index=df_nse['Date']
# data=df_nse.sort_index(ascending=True,axis=0)
# new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
# for i in range(0,len(data)):
#     new_data["Date"][i]=data['Date'][i]
#     new_data["Close"][i]=data["Close"][i]
# new_data.index=new_data.Date
# new_data.drop("Date",axis=1,inplace=True)
# dataset=new_data.values
# train=dataset[0:987,:]
# valid=dataset[987:,:]
# scaler=MinMaxScaler(feature_range=(0,1))
# scaled_data=scaler.fit_transform(dataset)
# x_train,y_train=[],[]
# for i in range(60,len(train)):
#     x_train.append(scaled_data[i-60:i,0])
#     y_train.append(scaled_data[i,0])
    
# x_train,y_train=np.array(x_train),np.array(y_train)
# x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# model=load_model("saved_model.h5")
# inputs=new_data[len(new_data)-len(valid)-60:].values
# inputs=inputs.reshape(-1,1)
# inputs=scaler.transform(inputs)
# X_test=[]
# for i in range(60,inputs.shape[0]):
#     X_test.append(inputs[i-60:i,0])
# X_test=np.array(X_test)
# X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
# closing_price=model.predict(X_test)
# closing_price=scaler.inverse_transform(closing_price)
# train=new_data[:987]
# valid=new_data[987:]
# valid['Predictions']=closing_price
# df= pd.read_csv("./stock_data.csv")
# app.layout = html.Div([
   
#     html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
#     dcc.Tabs(id="tabs", children=[
       
#         dcc.Tab(label='NSE-TATAGLOBAL Stock Data',children=[
#             html.Div([
#                 html.H2("Actual closing price",style={"textAlign": "center"}),
#                 dcc.Graph(
#                     id="Actual Data",
#                     figure={
#                         "data":[
#                             go.Scatter(
#                                 x=train.index,
#                                 y=valid["Close"],
#                                 mode='markers'
#                             )
#                         ],
#                         "layout":go.Layout(
#                             title='scatter plot',
#                             xaxis={'title':'Date'},
#                             yaxis={'title':'Closing Rate'}
#                         )
#                     }
#                 ),
#                 html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
#                 dcc.Graph(
#                     id="Predicted Data",
#                     figure={
#                         "data":[
#                             go.Scatter(
#                                 x=valid.index,
#                                 y=valid["Predictions"],
#                                 mode='markers'
#                             )
#                         ],
#                         "layout":go.Layout(
#                             title='scatter plot',
#                             xaxis={'title':'Date'},
#                             yaxis={'title':'Closing Rate'}
#                         )
#                     }
#                 )                
#             ])                
#         ]),
#         dcc.Tab(label='Facebook Stock Data', children=[
#             html.Div([
#                 html.H1("Facebook Stocks High vs Lows", 
#                         style={'textAlign': 'center'}),
              
#                 dcc.Dropdown(id='my-dropdown',
#                              options=[{'label': 'Tesla', 'value': 'TSLA'},
#                                       {'label': 'Apple','value': 'AAPL'}, 
#                                       {'label': 'Facebook', 'value': 'FB'}, 
#                                       {'label': 'Microsoft','value': 'MSFT'}], 
#                              multi=True,value=['FB'],
#                              style={"display": "block", "margin-left": "auto", 
#                                     "margin-right": "auto", "width": "60%"}),
#                 dcc.Graph(id='highlow'),
#                 html.H1("Facebook Market Volume", style={'textAlign': 'center'}),
         
#                 dcc.Dropdown(id='my-dropdown2',
#                              options=[{'label': 'Tesla', 'value': 'TSLA'},
#                                       {'label': 'Apple','value': 'AAPL'}, 
#                                       {'label': 'Facebook', 'value': 'FB'},
#                                       {'label': 'Microsoft','value': 'MSFT'}], 
#                              multi=True,value=['FB'],
#                              style={"display": "block", "margin-left": "auto", 
#                                     "margin-right": "auto", "width": "60%"}),
#                 dcc.Graph(id='volume')
#             ], className="container"),
#         ])
#     ])
# ])
# @app.callback(Output('highlow', 'figure'),
#               [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["High"],
#                      mode='lines', opacity=0.7, 
#                      name=f'High {dropdown[stock]}',textposition='bottom center'))
#         trace2.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Low"],
#                      mode='lines', opacity=0.6,
#                      name=f'Low {dropdown[stock]}',textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Price (USD)"})}
#     return figure
# @app.callback(Output('volume', 'figure'),
#               [Input('my-dropdown2', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Volume"],
#                      mode='lines', opacity=0.7,
#                      name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data, 
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M',
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Transactions Volume"})}
#     return figure
# if __name__=='__main__':
#     app.run_server(debug=True)

# app = dash.Dash()
# server = app.server
# app.layout = html.Div([
#     html.H4('Prediction model'),
#     dcc.Dropdown(['XGBoost','RNN','LSTM'], 'LSTM', id='prediction-model'),
#     html.Div(id='prediction-model-selected'),
#     html.H4('Property'),
#     dcc.Dropdown(['RSI', 'Bolling Bands', 'Moving Average', 'None'], 'None', id='prediction-property'),
#     html.Div(id='prediction-property-selected')
# ])

# @app.callback(
#     Output('prediction-model-selected', 'children'),
#     Input('prediction-model', 'value'),
#     Output('prediction-property-selected', 'children'),
#     Input('prediction-property-selected', 'value')
# )

# def update_prediction_model_select(value):
#     return f'You selected {value}'

# def update_prediction_property_select(value):
#     return f'You selected {value}'
# symbol = yf.Ticker('BTC-USD')
# df = pd.read_csv('./NSE-TATA.csv')
# df_btc = symbol.history(interval='1d', period='max')
# df_btc = df_btc[df_btc.index > datetime(2020,1,1)]
# df_btc = df_btc[df_btc.index < datetime(2021,9,1)]


# change = df['Close'].diff()
# change.dropna(inplace=True)

# change_up = change.copy()
# change_down = change.copy()

# change_up[change_up < 0] = 0
# change_down[change_down > 0] = 0

# avg_up = change_up.rolling(14).mean()
# avg_down = change_down.rolling(14).mean().abs()

# rsi = 100 * avg_up /(avg_up + avg_down)

# ax1 = plt.subplot2grid((10, 1), (0,0), rowspan=4, colspan=1)
# ax2 = plt.subplot2grid((10, 1), (5,0), rowspan=4, colspan=1)

# ax1.plot(df['Close'], linewidth = 2)
# ax1.set_title('Close Price')

# ax2.plot(rsi, color='orange', linewidth = 1)
# ax2.axhline(30, linestyle = '--', linewidth = 1.5, color='green')
# ax2.axhline(70, linestyle = '--', linewidth = 1.5, color='red')

# df = df.iloc[::-1]
# df['Date'] = pd.to_datetime(df['Date'])
# df['20wma'] = df['Close'].rolling(window=140).mean()

# fig = go.Figure(data = [go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
# fig.add_trace(go.Scatter(x=df['Date'], y = df['20wma'], line=dict(color='#e0e0e0'), name='20 Week'))
# fig.update_layout(yaxis_title = 'Price', xaxis_title = 'Date')
# fig.update_yaxes(type='log')

# fig.show()
# plt.show()

# from here
# Data
# symbol = yf.Ticker('BTC-USD')
# df = pd.read_csv('./NSE-TATA.csv')
# df = df.iloc[::-1]
df = yf.download(tickers='BTC-USD', period='1d', interval='1m')

# df['Date'] = pd.to_datetime(df['Date'])
df['20wma'] = df['Close'].rolling(window=140).mean()
df['std'] = df['Close'].rolling(window=140).mean()

change = df['Close'].diff()
change.dropna(inplace=True)

change_up = change.copy()
change_down = change.copy()

change_up[change_up < 0] = 0
change_down[change_down > 0] = 0

avg_up = change_up.rolling(14).mean()
avg_down = change_down.rolling(14).mean().abs()

rsi = 100 * avg_up /(avg_up + avg_down)

# fig - canlde stick
fig = go.Figure(data = [go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])

# fig - close price
fig.add_trace(go.Scatter(x=df.index, y =df['Close'], line=dict(color='blue'), name='Close'))

# fig - moving avg
# fig.add_trace(go.Scatter(x=df['Date'], y = df['20wma'], line=dict(color='yellow'), name='20 Week moving avg'))

# upper bolling band
fig.add_trace(go.Scatter(x=df.index, y=df['20wma'] + (df['std'] * 2), line_color = 'gray', name='upper band', opacity=0.5))
# lower bolling band
fig.add_trace(go.Scatter(x=df.index, y=df['20wma'] - (df['std'] * 2), line_color = 'gray', name='lower band', opacity=0.5))

# fig - layout
fig.update_layout(yaxis_title = 'Price', xaxis_title = 'Date')
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_yaxes(type='log')


# fig2
fig2 = go.Figure(data = [go.Scatter(x=df.index, y=rsi)])
fig2.add_hline(y=30, line_dash='dash', line_color='green')
fig2.add_hline(y=70, line_dash='dash', line_color='red')

fig3 = go.Figure(data = [go.Scatter(x=df.index, y=df['20wma'] + (df['std'] * 2))])
fig4 = go.Figure(data = [go.Scatter(x=df.index, y=df['20wma'] - (df['std'] * 2))])



# display
app = dash.Dash()
server = app.server
app.layout = html.Div([
    html.H4('Indicators'),
    dcc.Dropdown(['RSI', 'Bolling Bands', 'Moving Average', 'None'], 'None', id='select-property', multi=True),
    html.H4('Price'),
    dcc.Dropdown(['Close', 'Price of Change'], 'Price of Change', id='select-price'),
    html.Div(id='prediction-property-selected'),
    html.H4("Stock prediction"),
    dcc.Graph(id='stock_chart', 
        figure = fig,
        animate=True
    ),  
    dcc.Interval(id='update_stock_chart',interval=1000*1*60),
    html.Div(
        id='rsi_chart',
        children = [
                html.H2("RSI chart",style={"textAlign": "center"}),
                dcc.Graph(
                figure=fig2,
                animate=True
            )
        ]
    ),
    dcc.Interval(id='update_rsi_chart',interval=1000*1*60)
])

@app.callback(Output('stock_chart', 'figure'),
              [Input('update-live','n_intervals'),
              Input('select-property', 'value'), 
              Input('select-price', 'value')])

def update_graph(n_interval, selected_dropdown, selected_price):

    data = yf.download(tickers='BTC-USD', period='1d', interval='1m')
    last_row = data.iloc[-1:]
    df.append(last_row, ignore_index = True)
    print(df)
    
    fig = go.Figure(data = [go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])

    if 'Close' in selected_price:
        fig = go.Figure(data = [go.Scatter(x=df.index, y =df['Close'], line=dict(color='blue'), name='Close')])

    if 'Moving Average' in selected_dropdown:
        fig.add_trace(go.Scatter(x=df.index, y = df['20wma'], line=dict(color='yellow'), name='20 Week moving avg'))

    if 'Bolling Bands' in selected_dropdown:
        # upper bolling band
        fig.add_trace(go.Scatter(x=df.index, y=df['20wma'] + (df['std'] * 2), line_color = 'gray', name='upper band', opacity=0.5))
        # lower bolling band
        fig.add_trace(go.Scatter(x=df.index, y=df['20wma'] - (df['std'] * 2), line_color = 'gray', fill = 'tonexty', name='lower band', opacity=0.5))

    fig.update_layout(xaxis=dict(range=[min(df.index),max(df.index)]),
                                                yaxis=dict(range=[min(df['Low']),max(df['High'])]))

    return fig

if __name__=='__main__':
    app.run_server(debug=True)