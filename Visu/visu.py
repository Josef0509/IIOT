import sqlite3
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../IIOT_DB.db')

# Connect to the database
def fetch_data(query):
    conn = sqlite3.connect(filename)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    dcc.Dropdown(
        id='table-dropdown',
        options=[
            {'label': 'Dispenser', 'value': 'Dispenser'},
            {'label': 'DispVibration', 'value': 'DispVibration'},
            {'label': 'Temperature', 'value': 'Temperature'},
            {'label': 'finalWeight', 'value': 'finalWeight'},
            {'label': 'dropVibration', 'value': 'dropVibration'},
            {'label': 'ground_truth', 'value': 'ground_truth'}
        ],
        value='Dispenser'
    ),
    dcc.Graph(id='graph')
])

# Callback to update the graph based on selected table
@app.callback(
    Output('graph', 'figure'),
    [Input('table-dropdown', 'value')]
)
def update_graph(selected_table):
    query = f"SELECT * FROM {selected_table}"
    df = fetch_data(query)

    print(df)
    
    if selected_table == 'Dispenser':
        fig = px.scatter(df, x='time', y='fill_level_grams', color='color', title='Dispenser Fill Levels', 
                         color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'}, 
                         hover_data=['fill_level_grams', 'recipe', 'bottle'])
        
    elif selected_table == 'DispVibration':
        fig = px.scatter(df, x='time', y='vibration_index', color='color', title='Dispenser Vibration Index',
                         color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green'},
                         hover_data=['vibration_index', 'bottle'])
        
    elif selected_table == 'Temperature':
        fig = px.line(df, x='time', y='temperature_C', title='Temperature Over Time')
        
    elif selected_table == 'finalWeight':
        fig = px.scatter(df, x='time', y='final_weight', title='Final Weight of Bottles', hover_data=['bottle'])

    elif selected_table == 'dropVibration':
        fig = px.line(df, x='n', y='dropVibration', color='bottle', title='Drop Vibrations', hover_data=['bottle'])
        fig.update_layout(autotypenumbers='convert types')

    elif selected_table == 'ground_truth':
        fig = px.scatter(df, x='bottle', y='is_cracked', title='Ground Truth: Cracked Bottles')

    return fig

# Run the app
app.run_server(debug=True)