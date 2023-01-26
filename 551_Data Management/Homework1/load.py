import sys
import requests
import pandas as pd

firebase_path = 'https://dsci551-hw-4a0a3-default-rtdb.firebaseio.com/hw1/'
json_suffix = '.json'


def load_csv_to_df(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except:
        print('Please input the correct file path')


def convert_df_to_json(df, key):
    try:
        json_raw = df.to_json(orient='records')
        json_formatted = '{"' + key + '": ' + json_raw + '}'
        return json_formatted
    except:
        pass


def upload_data_to_firebase(url, json_object):
    try:
        print('Uploading raw data to Firebase..')
        res = requests.patch(url, json_object)
        if res.status_code == 200:
            print('Data uploaded successfully')
        else:
            print('Failed to upload data')
        return res
    except:
        print('Failed to upload data')


if __name__ == '__main__':
    csv_path = sys.argv[1]
    # read data from csv to dataframe
    df_cars = load_csv_to_df(csv_path)
    # convert dataframe to json
    json_cars = convert_df_to_json(df_cars, 'raw')
    # start uploading to firebase
    res_raw_cars = upload_data_to_firebase(firebase_path + json_suffix, json_cars)
