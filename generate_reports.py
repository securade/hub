import os
import pandas as pd
from datetime import datetime
import dash_bootstrap_components as dbc
from dash import html
from utils.general import clean_str
import json


def process_files(path):
    if os.listdir(path) == None:
        return None

    cameras = os.listdir(path)
    violations = {}
    for cam in cameras:
        if cam != ".DS_Store":
            days = os.listdir(f"{path}/{cam}")
            violations[cam] = {}
            for day in days:
                if day != ".DS_Store":
                    violations[cam][day] = [
                        i.rstrip(".jpg") for i in os.listdir(f"{path}/{cam}/{day}")]

    return violations


def get_duration(camera_no,path):
    configuration = pd.read_json(path)
    sources = configuration['sources']
    for source in sources:
        # print(camera_no)
        if clean_str(source['url']) == camera_no:
            return source['duration']


def get_camera_list(config_path):
    configuration = pd.read_json(config_path)
    sources = configuration['sources']
    result = []
    for source in sources:
        result.append(clean_str(source['url']))
    return result


def get_dates(source,path):
    if os.listdir(path) == None:
        return None

    cameras = os.listdir(path)
    # print(cameras)
    total_days = []
    for cam in cameras:
        if cam != ".DS_Store" and cam == source:
            for date in os.listdir(f"{path}/{cam}"):
                if date != ".DS_Store":
                    date = datetime.strptime(date, '%Y-%m-%d').date()
                    if date not in total_days:
                        total_days.append(date)
    # print(total_days)
    return total_days


# print(get_dates())
# print(process_files("dashboard-sk/output"))
# print(get_camera_list())

def create_card(image_path,date_time,violation_type):
    card = dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.CardImg(
                            src=image_path,
                            className="img-fluid rounded-start",
                            top = True
                        ),
                        className="col-md-4",
                    ),
                    dbc.Col(
                        dbc.CardBody(
                            [
                                html.H4(date_time, className="card-title"),
                                # html.P(
                                #     "This is a wider card with supporting text "
                                #     "below as a natural lead-in to additional "
                                #     "content. This content is a bit longer.",
                                #     className="card-text",
                                # ),
                                dbc.Badge(
                                    violation_type,
                                    color="secondary",
                                    text_color="light",
                                    className="border me-1",
                                ),
                            ]
                        ),
                        className="col-md-8",
                    ),
                ],
                className="g-0 d-flex align-items-center",
            )
        ],
        className="mb-3",
        style={"maxWidth": "540px"},
    )
    return card

def get_violation(path,camera_no):
    '''returns the type of violation a particular source is detecting'''
    configurations = pd.read_json(path)
    sources = configurations['sources']
    # print(camera_no)
    for source in sources:
        # print(source)
        if clean_str(source['url']) == camera_no:
            policy_file = source['policy_file']
            # print(policy_file)
    policy = json.load(open(policy_file))
    return policy['type']