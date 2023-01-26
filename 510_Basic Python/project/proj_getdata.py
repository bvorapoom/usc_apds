import pandas as pd
import datetime
import proj_airquality_api_crawler_function1 as aq1
import proj_airquality_api_crawler_function2 as aq2
import proj_cnn_web_scraper as cnn
import proj_wiki_web_scraper as wiki
import proj_weather_api_crawler as wt
import sqlite3



def get_data_from_downloaded_files(file_path):
    ''' read local csv file from filepath given
    params:
        file_path : str, path of the csv file
    returns:
        Dataframe object
    '''
    df = pd.read_csv(file_path)
    return df


def save_data_to_local_storage(result_df, file_path):
    ''' save dataframe to csv to filepath given
    params:
        result_df : Dataframe object
        file_path : str, location to save the csv
    '''
    result_df.to_csv(file_path, index = False, sep = ',')



def save_data_to_database(result_df, table_name):
    con = sqlite3.connect('saved_dsci510_project.db')
    result_df.to_sql(table_name, con, if_exists = 'replace', index = False)




def getdata_from_database(query_clause):
    con = sqlite3.connect('raw_dsci510_project.db')
    result = pd.read_sql_query(query_clause, con = con)
    return result




# FOR PART 1 -- GET PM2.5 EXPOSURE DATA



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



def clean_visited_locations_df(vs_loc):
    vs_loc['Start Date'] = pd.to_datetime(vs_loc['Start Date'], format = '%Y%m%d')
    vs_loc['End Date'] = pd.to_datetime(vs_loc['End Date'], format = '%Y%m%d')
    delta = datetime.timedelta(days = 1)
    vs_loc = vs_loc.sort_values('Start Date', ascending = True).reset_index(drop = True)
    temp = None
    
    for ind, row in vs_loc.iterrows():
        if temp is None:
            temp = row['End Date']
        else:
            if row['Start Date'] <= temp and row['End Date'] <= temp:
                vs_loc.drop(index = ind, inplace = True)
                continue
            if row['Start Date'] <= temp:
                vs_loc.loc[ind, 'Start Date'] = temp + delta
            if row['End Date'] > temp:
                temp = row['End Date']
    return vs_loc


def clean_output_aq_df(aq_df):
    temp = None
    for ind, row in aq_df.iterrows():
        if temp is None:
            temp = row['Date']
        else:
            if row['Date'] == temp:
                aq_df.drop(index = ind, inplace = True)
                continue
            else:
                temp = row['Date']
    return aq_df



def getdata_pm25_exposure_rawfile():

    vs_loc = pd.read_csv('../data/raw/raw_input_visited_locations.csv',)
    vs_loc = clean_visited_locations_df(vs_loc)
    aq_df = get_data_from_downloaded_files('../data/raw/raw_airquality_1_info.csv')

    save_data_to_local_storage(aq_df, '../data/saved/saved_airquality_1_info.csv')
    save_data_to_database(aq_df, 'airquality1_info')
    return aq_df



def getdata_pm25_exposure_database():
    aq_df = getdata_from_database('select * from airquality1_info')
    save_data_to_local_storage(aq_df, '../data/saved/saved_airquality_1_info.csv')
    return aq_df



def getdata_pm25_exposure_remote():

    print('''\n---------------NEED USER INPUT--------------\n''')
    print('For Project part1 : Program PM2.5 Exposure based on user\'s visited locations')
    user_input = input('Type STATIC if you want to use visited locations from local file. Type REMOTE if you want to input new visited locations.')
    if user_input.lower() == 'static':
        vs_loc = pd.read_csv('../data/raw/raw_input_visited_locations.csv')
        vs_loc = clean_visited_locations_df(vs_loc)
    elif user_input.lower() == 'remote':
        vs_loc = get_data_visited_locations_from_user()
        vs_loc = clean_visited_locations_df(vs_loc)
    else:
        print('incorrect format to get visited locations')

    print('crawling OpenAQ API... This might take a few minutes...')
    aq_df = aq1.aqmain1_get_air_quality_info_visited_locations(vs_loc)

    save_data_to_local_storage(aq_df, '../data/saved/saved_airquality_1_info.csv')
    save_data_to_database(aq_df, 'airquality1_info')
    return aq_df


def main_getdata_pm25_exposure(mode = 'raw_file'):
    if mode.lower() == 'raw_file':
        aq_df = getdata_pm25_exposure_rawfile()
    elif mode.lower() == 'database':
        aq_df = getdata_pm25_exposure_database()
    elif mode.lower() == 'remote':
        aq_df = getdata_pm25_exposure_remote()
    else:
        return 'Please enter valid mode to get data'

    aq_df = clean_output_aq_df(aq_df)

    return aq_df




# FOR PART 2 -- GET DATA FOR CORRELATION / ML MODEL


def getdata_corr_ml_rawfile():
    th_city_df = get_data_from_downloaded_files('../data/raw/raw_thcity_info.csv')
    th_city_list = th_city_df['City'].values.tolist()
    pop_df = get_data_from_downloaded_files('../data/raw/raw_population_info.csv')
    wt_df = get_data_from_downloaded_files('../data/raw/raw_weather_info.csv')
    aq_df = get_data_from_downloaded_files('../data/raw/raw_airquality_2_info.csv')

    save_data_to_local_storage(th_city_df, '../data/saved/saved_thcity_info.csv')
    save_data_to_local_storage(pop_df, '../data/saved/saved_population_info.csv')
    save_data_to_local_storage(wt_df, '../data/saved/saved_weather_info.csv')
    save_data_to_local_storage(aq_df, '../data/saved/saved_airquality_2_info.csv')

    save_data_to_database(th_city_df, 'thcity_info')
    save_data_to_database(pop_df, 'population_info')
    save_data_to_database(wt_df, 'weather_info')
    save_data_to_database(aq_df, 'airquality2_info')

    return th_city_list, pop_df, wt_df, aq_df





def getdata_corr_ml_database():
    th_city_df = getdata_from_database('select * from thcity_info')
    th_city_list = th_city_df['City'].values.tolist()
    pop_df = getdata_from_database('select * from population_info')
    wt_df = getdata_from_database('select * from weather_info')
    aq_df = getdata_from_database('select * from airquality2_info')

    save_data_to_local_storage(th_city_df, '../data/saved/saved_thcity_info.csv')
    save_data_to_local_storage(pop_df, '../data/saved/saved_population_info.csv')
    save_data_to_local_storage(wt_df, '../data/saved/saved_weather_info.csv')
    save_data_to_local_storage(aq_df, '../data/saved/saved_airquality_2_info.csv')

    return th_city_list, pop_df, wt_df, aq_df





def getdata_corr_ml_remote():
    api_key = '6163e1f312594b0ea6761645211111'

    print('scraping CNN website to get list of Thailand cities...')
    th_city_list = cnn.get_list_of_thailand_cities_from_cnn()
    th_city_df = pd.DataFrame(th_city_list, columns = ['City'])
    print('scraping Wikipedia website... This might take a minute...')
    pop_df = wiki.popmain_get_population_data_all_thailand_cities(th_city_list)
    print('crawling Weather API... This might take a few minutes...')
    wt_df = wt.wtmain_get_weather_data_all_thailand_cities(api_key, th_city_list)
    print('crawling OpenAQ API... This might take a few minutes...')
    aq_df = aq2.aqmain2_get_air_quality_info_all_thailand_cities(th_city_list)

    
    save_data_to_local_storage(th_city_df, '../data/saved/saved_thcity_info.csv')
    save_data_to_local_storage(pop_df, '../data/saved/saved_population_info.csv')
    save_data_to_local_storage(wt_df, '../data/saved/saved_weather_info.csv')
    save_data_to_local_storage(aq_df, '../data/saved/saved_airquality_2_info.csv')

    save_data_to_database(th_city_df, 'thcity_info')
    save_data_to_database(pop_df, 'population_info')
    save_data_to_database(wt_df, 'weather_info')
    save_data_to_database(aq_df, 'airquality2_info')


    return th_city_list, pop_df, wt_df, aq_df



def main_getdata_corr_ml(mode = 'raw_file'):
    if mode.lower() == 'raw_file':
        th_city_list, pop_df, wt_df, aq_df = getdata_corr_ml_rawfile()
    elif mode.lower() == 'database':
        th_city_list, pop_df, wt_df, aq_df = getdata_corr_ml_database()
    elif mode.lower() == 'remote':
        th_city_list, pop_df, wt_df, aq_df = getdata_corr_ml_remote()
    else:
        return 'Please enter valid mode to get data'

    wt_df['Date'] = pd.to_datetime(wt_df['Date']).dt.strftime('%Y-%m-%d')
    aq_df['Date'] = pd.to_datetime(aq_df['Date']).dt.strftime('%Y-%m-%d')

    return th_city_list, pop_df, wt_df, aq_df



