import requests
import json
import pandas as pd
import datetime
import argparse


def gen_request_url(base_url, api_key, location, start_date, end_date = None):
    ''' generate request URL for Weather API based on parameters given
    params:
        base_url : str
        api_key : str - use 6163e1f312594b0ea6761645211111
        location : str, Thailand city name
        start_date : date
        end_date : date, optional
    returns:
        request_url : str, URL to crawl from
    '''
    try:
        request_url = base_url + '?key=' + str(api_key)
        if end_date is None:
            start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            assert start_date >= (datetime.date.today() - datetime.timedelta(5)).strftime('%Y-%m-%d'), 'Can only access 5-day historical data'
            request_url = request_url + '&q=' + location + '&dt=' + start_date 
            return request_url
        else:
            start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            assert start_date >= (datetime.date.today() - datetime.timedelta(5)).strftime('%Y-%m-%d'), 'Can only access 5-day historical data'
            assert end_date <= datetime.date.today().strftime('%Y-%m-%d'), 'Can only access 5-day historical data'
            request_url = request_url + '&q=' + location + '&dt=' + start_date + '&end_dt=' + end_date
            return request_url
    except:
        return 'Incorrect input parameters for URL'
    
    
def get_raw_weather_info_from_api(request_url):
    ''' crawl Weather API from given request_url
    params:
        request_url : str, URL to crawl from
    returns:
        weather_df : Dataframe, contains weather information based on parameters given
    '''
    resp = requests.get(request_url)
    
    if resp.status_code != 200:
        return None
    
    resp_json = resp.json()
    
    df_cols = ['City', 'Date', 'Datetime', 'Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 
               'Precipitation (mm)', 'Humidity', 'Cloud', 'Chance of Rain', 'Visibility (km)']
    weather_df = pd.DataFrame(columns = df_cols)
    
    w_city = resp_json['location']['name']
    for day_node in resp_json['forecast']['forecastday']:
        w_date = day_node['date']
        for hour_node in day_node['hour']:
            w_datetime = hour_node['time']
            w_temp = hour_node['temp_c']
            w_windspeed = hour_node['wind_kph']
            w_pressure = hour_node['pressure_mb']
            w_precip = hour_node['precip_mm']
            w_humidity = hour_node['humidity']
            w_cloud = hour_node['cloud']
            w_chancerain = hour_node['chance_of_rain']
            w_vis = hour_node['vis_km']
            weather_df.loc[len(weather_df.index)] = [w_city, w_date, w_datetime, w_temp, w_windspeed,
                                                    w_pressure, w_precip, w_humidity, w_cloud, w_chancerain, w_vis]
            
    return weather_df



def format_output_df(raw_aq_df):
    ''' properly format dataframe '''
    raw_aq_df['Date'] = pd.to_datetime(raw_aq_df['Datetime'].str[:19]).dt.strftime('%Y-%m-%d')
    raw_aq_df['Hour'] = pd.to_datetime(raw_aq_df['Datetime'].str[:19]).dt.strftime('%H')
        
    df_cols = ['City', 'Date', 'Hour', 'Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 
               'Precipitation (mm)', 'Humidity', 'Cloud', 'Chance of Rain', 'Visibility (km)']
    raw_aq_df = raw_aq_df[df_cols]
        
    return raw_aq_df




def wtmain_get_weather_data_all_thailand_cities(api_key, city_list):
    ''' get weather data for all Thailand cities
    params:
        api_key : str, use - 6163e1f312594b0ea6761645211111
        city_list : list, list of Thailand cities
    returns:
        result_df : Dataframe, contains weather information of all thailand cities on dates given
    '''
    base_url = 'http://api.weatherapi.com/v1/history.json'
    start_date = (datetime.date.today() - datetime.timedelta(5)).strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    result_df = None
    
    for city in city_list:
        request_url = gen_request_url(base_url, api_key, city, start_date, end_date)
        city_output_df = get_raw_weather_info_from_api(request_url)
        if city_output_df is None:
            continue
        else:
            if result_df is None:
                result_df = city_output_df
            else:
                result_df = result_df.append(city_output_df)    
    
    result_df = format_output_df(result_df)
    return result_df










