from re import template
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
import yfinance as yf
import matplotlib.pyplot as plt

df = yf.download(tickers='BTC-USD', period='15d', interval='15m')

# moving avg
df['20wma'] = df['Close'].rolling(window=140).mean()
df['std'] = df['Close'].rolling(window=140).mean()

# rsi
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
fig.update_layout(xaxis_rangeslider_visible=False)

# fig - close price
fig.add_trace(go.Scatter(x=df.index, y =df['Close'], line=dict(color='#7e57c2'), name='Close'))

# fig - moving avg
fig.add_trace(go.Scatter(x=df.index, y = df['20wma'], line=dict(color='#2196f3'), name='20 Week moving avg'))

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
fig2.update_layout(hovermode='y unified')

# prediction
# Analyze data
df["Date"]=pd.to_datetime(df.index,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

# Sort and filter
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]

# Normalize

scaler=MinMaxScaler(feature_range=(0,1))
final_dataset=new_dataset.values
train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

new_dataset.index=new_dataset['Date']
new_dataset.drop('Date',axis=1,inplace=True)
scaler=MinMaxScaler(feature_range=(0,1))

scaled_data=scaler.fit_transform(new_dataset.values)

x_train_data,y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

# build model
lstm_model=load_model("saved_model_3.h5")
inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
print(inputs_data.shape[0])
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])

X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

valid_data=new_dataset[987:]
valid_data['Predictions']=predicted_closing_price

# fig for prediction
df['Predictions'] = pd.Series(valid_data['Predictions'], index=valid_data.index)
fig.add_trace(go.Scatter(x=df.index, y =df['Predictions'], line=dict(color='#d32f2f'), name='prediction'))
# end


# display
app = dash.Dash()
server = app.server
app.layout = html.Div([
    html.Div(children=[
        html.Div(children=[
            html.H3('Select Indicators', style={'font-family':'Lato'}),
            dcc.Dropdown(['RSI', 'Bolling Bands', 'Moving Average', 'None'], 'None', id='select-property', multi=True)
        ], style={
                'background-color': '#e1f5fe',
                'border': 'none',
                'border-radius': '9px',
                'color': '#039be5',
                'padding': '20px'
        }),
        html.Div(children=[
            html.H3('Select Price', style={'font-family':'Lato'}),
            dcc.Dropdown(['Close', 'Price of Change'], 'Price of Change', id='select-price')
        ], style={
            'background-color': '#ede7f6',
            'border': 'none',
            'border-radius': '9px',
            'color': '#512da8',
            'padding': '20px' 
        })
    ], style={
                'display':'flex',
                'flex-direction':'column',
                'padding': '20px' ,
                'gap': '10px' 
            }),
    html.Div(id='prediction-property-selected'),    
    html.H1("Stock Prediction", style={
                                        "textAlign": "center",
                                        'marginBottom': '0px',
                                        'font-family': 'Roboto'
                                        }),
    html.Div(children=[
        dcc.Graph(id='stock_chart', 
            figure = fig,
            animate=True
        )
    ], style={
        'padding': '20px'
    }), 
    dcc.Interval(id='update_stock_chart',interval=1000*15*60),
    html.Div(
        id='rsi_chart',
        children = [
                html.H1("RSI chart",style={
                                            "textAlign": "center",
                                            'marginBottom': '0px'
                                            }),
                dcc.Graph(id='rsi_real_chart',
                figure=fig2,
                animate=True
            )
        ]
    ),
    dcc.Interval(id='update_rsi_chart',interval=1000*15*60)
])

@app.callback(Output('stock_chart', 'figure'),
              [Input('update_stock_chart','n_intervals'),
              Input('select-property', 'value'), 
              Input('select-price', 'value')])

def update_graph(n_interval, selected_dropdown, selected_price):

    df = yf.download(tickers='BTC-USD', period='25d', interval='30m')
    
    df['20wma'] = df['Close'].rolling(window=140).mean()
    df['std'] = df['Close'].rolling(window=140).mean()

    # prediction
    df["Date"]=pd.to_datetime(df.index,format="%Y-%m-%d")
    df.index=df['Date']

    plt.figure(figsize=(16,8))
    plt.plot(df["Close"],label='Close Price history')

    # Sort and filter
    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

    for i in range(0,len(data)):
        new_dataset["Date"][i]=data['Date'][i]
        new_dataset["Close"][i]=data["Close"][i]

    # Normalize
    scaler=MinMaxScaler(feature_range=(0,1))
    final_dataset=new_dataset.values
    train_data=final_dataset[0:987,:]
    valid_data=final_dataset[987:,:]

    new_dataset.index=new_dataset['Date']
    new_dataset.drop('Date',axis=1,inplace=True)
    scaler=MinMaxScaler(feature_range=(0,1))

    scaled_data=scaler.fit_transform(new_dataset.values)

    x_train_data,y_train_data=[],[]
    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    # build model
    lstm_model=load_model("saved_model_3.h5")
    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)

    X_test=[]
    print(inputs_data.shape[0])
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])

    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price=lstm_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

    valid_data=new_dataset[987:]
    valid_data['Predictions']=predicted_closing_price

    # fig for prediction
    df['Predictions'] = pd.Series(valid_data['Predictions'], index=valid_data.index)

    # display
    
    fig = go.Figure(data = [go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])

    if 'Close' in selected_price:
        fig = go.Figure(data = [go.Scatter(x=df.index, y =df['Close'], line=dict(color='#ac65fa'), name='Close')])
    else:
        fig.update_layout(xaxis_rangeslider_visible=False)

    if 'Moving Average' in selected_dropdown:
        fig.add_trace(go.Scatter(x=df.index, y = df['20wma'], line=dict(color='#2196f3'), name='20 Week moving avg'))

    if 'Bolling Bands' in selected_dropdown:
        # upper bolling band
        fig.add_trace(go.Scatter(x=df.index, y=df['20wma'] + (df['std'] * 2), line_color = 'gray', name='upper band', opacity=0.5))
        # lower bolling band
        fig.add_trace(go.Scatter(x=df.index, y=df['20wma'] - (df['std'] * 2), line_color = 'gray', fill = 'tonexty', name='lower band', opacity=0.5))

    fig.add_trace(go.Scatter(x=df.index, y =df['Predictions'], line=dict(color='#d32f2f'), name='prediction'))
    fig.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom",  y=1.02, xanchor="right", x=1))
    fig.update_layout(xaxis=dict(range=[min(df.index),max(df.index)]),
                                                yaxis=dict(range=[min(df['Low']),max(df['High'])]))
    fig.update_xaxes(showspikes=True,spikethickness=1)
    fig.update_yaxes(showspikes=True,spikethickness=1)
                                                

    return fig

@app.callback(Output('rsi_chart', 'style'),
              [Input('select-property', 'value')])

def update_graph(selected_dropdown):
    if 'RSI' in selected_dropdown:
        return {'display':'block'}

    return {'display':'none'}

if __name__=='__main__':
    app.run_server(debug=True)