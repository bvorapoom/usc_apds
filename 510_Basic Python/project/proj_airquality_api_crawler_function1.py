import requests
import json
import pandas as pd
import datetime
import random
import argparse


def gen_request_url(base_url, domain, params_base, **params_other):
    ''' generate request URL for OpenAQ API based on parameters given
    params:
        base_url : str
        domain : str, domains that will be used here include measurements, averages, locations
        params_base : dictionary, parameters to pass when crawling API
        params_other : other parameters to pass when crawling API apart from what's in params_base
    returns:
        request_url : str, URL to crawl from
    '''

    params_allowed_list = ['date_from', 'date_to', 'country', 'city', 'location', 'location_id', 'coordinates', 'parameter', 'limit', 'spatial', 'temporal']
    
    try:
        request_url = base_url + str(domain) + '?'
        for k, v in params_base.items():
            if k in params_allowed_list:
                if k == 'date_from' or k == 'date_to':
                    v = pd.to_datetime(v).strftime('%Y-%m-%dT00:00:00')
                request_url = request_url + str(k) + '=' + str(v) + '&' 
            else:
                return 'Invalid keyword for API request'
        for k, v in params_other.items():
            if k in params_allowed_list:
                if k == 'date_from' or k == 'date_to':
                    v = pd.to_datetime(v).strftime('%Y-%m-%dT00:00:00')
                request_url = request_url + str(k) + '=' + str(v) + '&' 
            else:
                return 'Invalid keyword for API request'
        return request_url[:-1]
    except:
        return 'Incorrect input parameters for URL'
    

    
def get_sample_location_id_list(base_url, params_base, num_sample = 3, **city_or_location):
    ''' get sample of location ids from city or location
    params:
        base_url : str
        params_base : dictionary
        num_sample : int optional, number of location ids to sample from for each city / location
        city_or_location : select between passing city or location -- eg, city = 'Bangkok' or location = 'San Francisco'
    '''
    
    '''Due to the rate limit of the API and the limitation to request data by city and limiting number of locations,
    We need the function below to sample locations for each city so that we will request info of the city only for
    these locations. Doing this helps prevent the situation that we get incomplete data due to rate limit'''
    '''There is /averages domain but it does not allow to get the average data by city.'''
    
    domain = 'locations'
    
    try:
        request_url = gen_request_url(base_url, domain, params_base, **city_or_location)
        resp = requests.get(request_url)
        resp_json = resp.json()
        all_locations = resp_json['results']
        if len(all_locations) < num_sample:
            num_sample_tuned = len(all_locations)
        else:
            num_sample_tuned = num_sample
        sample_locations = random.sample(all_locations, k = num_sample_tuned)
        location_list = [loc['id'] for loc in sample_locations]
    except:
        location_list = None
        
    return location_list
    
    
def get_raw_air_quality_info_by_location_id(base_url, location_id, params_base, **city_or_location):
    ''' get air quality data for location id
    params:
        base_url : str
        location_id : int
        params_base : dictionary
        city_or_location : select between passing city or location parameter -- is not used to process, only used for storing values in output df
    returns:
        aq_df_loc_id : Dataframe, containing air quality info for inputted location id
    '''
    domain = 'measurements'
    request_url = gen_request_url(base_url, domain, params_base, location_id = location_id)
    resp = requests.get(request_url)
    resp_json = resp.json()
    
    df_cols = ['Country Code', 'City/Location', 'Datetime', 'PM25 value']
    aq_df_loc_id = pd.DataFrame(columns = df_cols)
    
    for hour_node in resp_json['results']:
        aq_country = hour_node['country']
        aq_cityloc = list(city_or_location.values())[0]
        aq_datetime = hour_node['date']['local']
        aq_pm25 = hour_node['value']
        aq_df_loc_id.loc[len(aq_df_loc_id.index)] = [aq_country, aq_cityloc, aq_datetime, aq_pm25]
        
    return aq_df_loc_id




def get_raw_air_quality_info_by_location_or_city(base_url, params_base, **city_or_location):
    ''' get air quality data for city or location inputted
    params:
        base_url : str
        params_base : dictionary
        city_or_location : select between passing city or location parameter
    returns:
        raw_aq_df_cityloc : Dataframe, containing air quality info for inputted city or location
    '''
    domain = 'measurements'
    loc_id_list = get_sample_location_id_list(base_url, params_base, **city_or_location)

    raw_aq_df_cityloc = None

    for loc_id in loc_id_list:
        aq_df_loc_id = get_raw_air_quality_info_by_location_id(base_url, loc_id, params_base, **city_or_location)
        if aq_df_loc_id is None:
                continue
        else:
            if raw_aq_df_cityloc is None:
                raw_aq_df_cityloc = aq_df_loc_id
            else:
                raw_aq_df_cityloc = raw_aq_df_cityloc.append(aq_df_loc_id) 

    return raw_aq_df_cityloc



def get_raw_air_quality_info_by_country_code_avg_daily(base_url, country_code, params_base, **city_or_location):
    ''' get average air quality data for country code inputted
    params:
        base_url : str
        country_code : str, 2-letter code
        params_base : dictionary
        city_or_location : select between passing city or location parameter -- is not used to process, only used for storing values in output df
    returns:
        raw_aq_df_cityloc : Dataframe, containing air quality info for inputted city or location
    '''
    domain = 'averages'
    params_base['spatial'] = 'country'
    params_base['temporal'] = 'day'
 
    request_url = gen_request_url(base_url, domain, params_base, country = country_code)
    resp = requests.get(request_url)
    resp_json = resp.json()
    
    df_cols = ['Country Code', 'City/Location', 'Datetime', 'PM25 value']
    raw_aq_df_country = pd.DataFrame(columns = df_cols)
    
    for day_node in resp_json['results']:
        aq_country = day_node['name']
        aq_cityloc = list(city_or_location.values())[0]
        aq_datetime = day_node['day']
        aq_pm25 = day_node['average']
        raw_aq_df_country.loc[len(raw_aq_df_country.index)] = [aq_country, aq_cityloc, aq_datetime, aq_pm25]
        
    return raw_aq_df_country



def format_output_df(raw_aq_df, time_gran):
    ''' properly format dataframe '''
    raw_aq_df['Date'] = pd.to_datetime(raw_aq_df['Datetime'].str[:19]).dt.strftime('%Y-%m-%d')
    raw_aq_df['Hour'] = pd.to_datetime(raw_aq_df['Datetime'].str[:19]).dt.strftime('%H')
    raw_aq_df['PM25 value'] = raw_aq_df['PM25 value'].astype(int)   
    
    if time_gran == 'hour':
        agg_aq_df = raw_aq_df.groupby(['Country Code', 'City/Location', 'Date', 'Hour'])['PM25 value'].mean().reset_index()
    elif time_gran == 'day':
        agg_aq_df = raw_aq_df.groupby(['Country Code', 'City/Location', 'Date'])['PM25 value'].mean().reset_index()
        agg_aq_df['Hour'] = 0
        
    agg_aq_df = agg_aq_df[['Country Code', 'City/Location', 'Date', 'Hour', 'PM25 value']]
        
    return agg_aq_df
        
        
def aqmain1_get_air_quality_info_visited_locations(vs_loc):
    ''' get air quality data on user's visited location
    params:
        vs_loc : Dataframe, containing 4 columns - country code, city/location, start date, end date
    returns:
        agg_df_exposure : Dataframe, containing air quality data on user's visited location
    '''
    base_url = 'https://api.openaq.org/v2/'
    raw_sum_df_exposure = None
    
    for index, row in vs_loc.iterrows():
        ex_country, ex_cityloc, ex_startdate, ex_enddate = row
        params_base = {
            'date_from' : datetime.datetime.strftime(ex_startdate, '%Y-%m-%d'),
            'date_to' : datetime.datetime.strftime(ex_enddate, '%Y-%m-%d'),
            'parameter' : 'pm25',
            'limit' : 1500
        }
        
        raw_aq_df = None
        
        # check if we can get info by location
        raw_aq_df = get_raw_air_quality_info_by_location_or_city(base_url, params_base, location = ex_cityloc)   
        
        # check if we can get info by city
        if raw_aq_df is None:
            raw_aq_df = get_raw_air_quality_info_by_location_or_city(base_url, params_base, city = ex_cityloc)
        
        # if we can't get info by location and city, return the average for the country  
        if raw_aq_df is None:
            raw_aq_df = get_raw_air_quality_info_by_country_code_avg_daily(base_url, ex_country, params_base, city = ex_cityloc)

        
        # appending data for each location to the dataframe
        if raw_aq_df is None:
                continue
        else:
            if raw_sum_df_exposure is None:
                raw_sum_df_exposure = raw_aq_df
            else:
                raw_sum_df_exposure = raw_sum_df_exposure.append(raw_aq_df) 
                
    agg_df_exposure = format_output_df(raw_sum_df_exposure, time_gran = 'day')
    agg_df_exposure = agg_df_exposure.sort_values('Date').reset_index(drop = True)
    return agg_df_exposure
        

def get_data_visited_locations_from_user():
    ''' ask user to input visited locations information
    *** read the instructions carefully on how to input the data ***
    returns:
        vs_loc : Dataframe, containing 4 columns - country code, city/location, start date, end date
    '''
    vs_loc = pd.DataFrame(columns = ['Country Code', 'City', 'Start Date', 'End Date'])
    print('''\n----------------------PLEASE READ THE INSTRUCTIONS BELOW------------------------\n''')
    print('Instruction 1: For each input, you need to provide 4 information: Country, City, Start date, End date')
    print('----- Format of the input is (Country Code)|(City)|(Start Date in yyyyMMdd format)|(End Date in yyyyMMdd format)')
    print('----- The separator for each info is the vertical bar (|)')
    print('----- Example of the input is FR|Paris|20211101|20211110')
    print('Instruction 2: If finished inputting data, type --- (triple hyphens)')
    print('Note: Historical data might be available just from the past 3 months')
    while True:
        loc = input('Please input visited locations in format specified:')
        if loc == '---':
            break
        else:
            try:
                country, city, start_date, end_date = loc.split('|')
                start_date = pd.to_datetime(start_date, format = '%Y%m%d')
                end_date = pd.to_datetime(end_date, format = '%Y%m%d')
                vs_loc.loc[len(vs_loc.index)] = [country, city, start_date, end_date]
            except:
                print('Incorrect input format. Please try again')

    return vs_loc



