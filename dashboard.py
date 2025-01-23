from dash import Dash, dcc, html, Input, Output, callback, dash_table
from generate_reports import get_duration, get_camera_list, process_files, get_dates, create_card, get_violation
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import datetime
import base64
import json
import io

from PIL import Image

# change directory accordingly
OUTPUT_PATH = "output"
CONFIG_PATH = "./configs/default.json"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Header
header = html.H4(
    "Dashboard", className="bg-info text-white p-2 mb-2 text-center"
)

# Dropdown List
indicator_dropdown = html.Div(
    [
        dbc.Label("Select Report"),
        dcc.Dropdown(
            ["Safety Score", "Safety Violations"],
            "Safety Violations",
            id="indicator",
            clearable=False,
        ),
    ],
    className="mb-4",
)

# image table
image_table = dbc.Card(id="image_table")
# image_table = html.Img(src="output/0/2023-04-27/", className="img-fluid rounded-start")

# camera selector
cameras = get_camera_list(CONFIG_PATH)
camera_selector = html.Div(
    [
        dbc.Label("Select Source"),
        dbc.Checklist(
            id="cam_number",
            options=[{"label": i, "value": i} for i in cameras],
            value=cameras,
            inline=True,
        ),
    ],
    className="mb-4",
)
# controlling the graph
controls = dbc.Card(
    [indicator_dropdown, camera_selector],
    body=True,
)
Bar_chart = dcc.Graph(id="bar-chart")

@callback(
    Output("bar-chart", "figure"),
    Input("indicator", "value"),
    Input("cam_number", "value"),
    Input('intermediate-value', 'data'))
def update_graph(indicator, cam_number, data):
    if cam_number == None:
        return {}
    else:
        score_list = {}
        violations = json.loads(data)
        df = pd.DataFrame([(key1, key2, val) 
                           for key1, inner_dict in violations.items() 
                           for key2, sublist in inner_dict.items() 
                           for val in sublist], columns=['Camera_no', 'Date', 'Timings'])
        df1 = df.loc[df['Camera_no'].isin(cam_number)]
        columns = df['Date'].unique()
        if indicator == "Safety Score":
            for cam in cam_number:
                time = get_duration(cam, CONFIG_PATH)
                score_list[cam] = {}
                df_int = df1.loc[df1['Camera_no'] == cam]
                for date in columns:
                    number_of_violations = df_int[df_int['Date'] == date].count(
                    ).Timings
                    # can actually calculate the score multiplied by 3 cameras but then the score will be too low
                    score = 1 - ((number_of_violations * time) / (86400))
                    score_list[cam][date] = score

            df1 = pd.DataFrame(score_list).T.reset_index().rename(
                columns={'index': 'Source'})
            df1 = df1.rename_axis('Date').melt(
                id_vars=['Source'], var_name='Date', value_name='Score')
            bar_chart = px.bar(
                df1, x='Date', y='Score', color='Source', title='Safety Score', barmode='group')
            return bar_chart
        else:
            df_violations = df1.groupby(['Camera_no', 'Date']).agg(
                {'Timings': 'count'}).reset_index()
            df_violations.rename(
                columns={'Timings': 'Count', 'Camera_no': 'Source'}, inplace=True)
            bar_chart = px.bar(df_violations, x="Date", y="Count",
                               color="Source", title="Violations/day", barmode='group')
            return bar_chart

@callback(
    Output("dates", "options"),
    Input("cam_number", "value"),
)
def updateDates(cam_number):
    if cam_number == None:
        return []
    else:
        dates = []
        for cam in cam_number:
            dates += get_dates(cam,path=OUTPUT_PATH)
        return [{"label": name, "value": name} for name in sorted(set(dates), reverse=True)]


# other dropdown
date_dropdown = html.Div(
    [
        dbc.Label("Select Date"),
        dcc.Dropdown(
            id="dates",
            options=[
                {"label": name, "value": name} 
                for name in sorted(updateDates(None))
            ],
            clearable=False,
        ),
    ],
    className="mb-4",
)

@callback (
    Output("image_table", "children"),
    Input("dates", "value"),
    Input("cam_number", "value"),
    Input('intermediate-value', 'data')
)
def updateImages(date, cam_number, data):
    if date == None or cam_number == None:
        return []
    else:
        images = []
        violations = json.loads(data)

        for cam in cam_number:
            if violations[cam]!= None:
                for dating in violations[cam]:
                    # converted_date = date.strftime("%Y-%m-%d")
                    if  dating == date:
                        violation_length = get_duration(cam, CONFIG_PATH)
                        # print (violation_length)
                        prev_time = 86400
                        for timing in sorted(violations[cam][dating], reverse=True):
                            # Split the string into parts and convert each part to an integer
                            hours, minutes, seconds = map(int, timing.split("-"))
                            # Calculate the total seconds
                            total_seconds = hours * 3600 + minutes * 60 + seconds
                            if prev_time - total_seconds > violation_length:
                                image_path = f"{OUTPUT_PATH}/{cam}/{dating}/{timing}.jpg"
                                with open(image_path, "rb") as image_file:
                                    # Resize image
                                    img = Image.open(image_file)
                                    img.thumbnail((320, 320))

                                    # Convert to Base64
                                    buffered = io.BytesIO()
                                    img.save(buffered, format="JPEG")
                                    img_data = base64.b64encode(buffered.getvalue())
                                    img_data = img_data.decode()
                                    img_data = "{}{}".format("data:image/jpg;base64, ", img_data)
                                # print(image_path)
                                violation_type = get_violation(CONFIG_PATH, cam)
                                images.append(create_card(img_data,timing,violation_type))
                            prev_time = total_seconds
        return images

@app.callback(Output('intermediate-value', 'data'),Input("indicator", "value"))
def load_data(value):
    violations = process_files(OUTPUT_PATH)
    return json.dumps(violations)

def serve_layout():
    return dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        controls
                    ],
                    width=4,
                ),
                dbc.Col(dbc.Card(Bar_chart, body=True), width=8),
            ]
        ),
        dbc.Row([dbc.Col([dbc.Card(date_dropdown)],width=4), dbc.Col([image_table],width=8)]),
        # dcc.Store stores the intermediate value
        dcc.Store(id='intermediate-value')
    ],
    fluid=True,
    className="dbc",
    
)

app.layout = serve_layout

app.title = "Securade.ai - HUB Dashboard"

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8888)