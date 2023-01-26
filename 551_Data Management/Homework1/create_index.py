import sys
import requests
import pandas as pd
import re
import json

firebase_path = 'https://dsci551-hw-4a0a3-default-rtdb.firebaseio.com/hw1/inverted_index'
json_suffix = '.json'


def load_csv_to_df(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except:
        print('Please input the correct file path')


def create_inverted_index(df):
    dict_raw = df.to_dict(orient='records')
    dict_inverted_index = {}
    for car in dict_raw:
        car_name = car['CarName']
        keywords = re.sub(r'[^a-zA-Z0-9]+', ' ', car_name).lower().split()
        for kw in keywords:
            if kw in dict_inverted_index.keys():
                dict_inverted_index[kw].append(car['car_ID'])
            else:
                dict_inverted_index[kw] = [car['car_ID']]
    return dict_inverted_index


def upload_data_to_firebase(url, json_object):
    res = requests.patch(url, json_object)
    return res


if __name__ == '__main__':
    csv_path = sys.argv[1]
    # read data from csv to dataframe
    df_cars = load_csv_to_df(csv_path)
    # invert index as keywords
    dict_inverted_index = create_inverted_index(df_cars)
    # start uploading to firebase
    print('Uploading inverted index data to Firebase..')
    json_object = json.dumps(dict_inverted_index)
    res_ii = upload_data_to_firebase(firebase_path + json_suffix, json_object)
    if res_ii.status_code == 200:
        print('Inverted index data uploaded successfully')
    else:
        print('Failed to upload inverted index data')
