import pandas as pd
from datetime import timedelta

pd.set_option('display.max_columns', 10)

path_accident_data = 'data/raw/US_Accidents_Dec21_updated.csv'
path_weather_data = 'data/raw/WeatherEvents_Jan2016-Dec2021.csv'
path_gdp_data = 'data/raw/SAGDP2N__ALL_AREAS_1997_2020.csv'
path_employment_data = 'data/raw/CAEMP25N__ALL_AREAS_2001_2019.csv'

mapping_state_path = 'data/cleaned/mapping/mapping_pop_state.csv'

path_output_accident_data = 'data/cleaned/data_accident_2016_2021.csv'
path_output_weather_data = 'data/cleaned/data_weather_2016_2021.csv'
path_output_gdp_data = 'data/cleaned/data_gdp_2015_2020.csv'
path_output_employment_data = 'data/cleaned/data_employment_2015_2019.csv'


def import_raw_data(path_accident_data, path_weather_data, path_gdp_data, path_employment_data):
    df_acc = pd.read_csv(path_accident_data)
    df_wea = pd.read_csv(path_weather_data)
    df_gdp = pd.read_csv(path_gdp_data)
    df_emp = pd.read_csv(path_employment_data, encoding='ISO-8859-1')
    return df_acc, df_wea, df_gdp, df_emp


def aggregate_accident_data(df_acc):
    """
    Get year, month of each accident from Start_Time attribute
    Aggregate by year, month, state, zipcode, severity of accidents
    map 'state' column'
    """
    df_acc['year'] = pd.to_datetime(df_acc['Start_Time']).dt.year
    df_acc['month'] = pd.to_datetime(df_acc['Start_Time']).dt.month
    df_acc['Zipcode'] = df_acc['Zipcode'].str[:5]

    df_acc_agg = df_acc.groupby(['year', 'month', 'State', 'Zipcode', 'Severity'], group_keys=False)[
        'ID'].count().reset_index()
    df_acc_agg.columns = ['year', 'month', 'postal_abbr', 'zipcode', 'severity', 'count_accidents']

    df_mapping_state = pd.read_csv(mapping_state_path)
    df_acc_agg = df_acc_agg.merge(df_mapping_state, how='left', on='postal_abbr')

    return df_acc_agg[['year', 'month', 'state_code', 'state', 'postal_abbr', 'zipcode', 'severity', 'count_accidents']]


def aggregate_weather_data(df_wea):
    """
    Get day, month, year of each record from StartTime attribute
    Aggregate by year, month, zipcode, state, event type, and severity of events
    Count the duration (in days) for the specific events by year/month/zipcode/state using the unique count of days
    map 'state' column
    """
    df_wea['start_time'] = pd.to_datetime(df_wea['StartTime(UTC)'])
    df_wea['end_time'] = pd.to_datetime(df_wea['EndTime(UTC)'])
    df_wea['year'] = df_wea['start_time'].dt.year
    df_wea['month'] = df_wea['start_time'].dt.month
    df_wea['day'] = df_wea['start_time'].dt.day

    df_wea_agg = df_wea.groupby(['year', 'month', 'State', 'ZipCode', 'Type', 'Severity'])[
        'day'].nunique().reset_index()
    df_wea_agg.columns = ['year', 'month', 'postal_abbr', 'zipcode', 'event_type', 'severity', 'duration_days']

    df_mapping_state = pd.read_csv(mapping_state_path)
    df_wea_agg = df_wea_agg.merge(df_mapping_state, how='left', on='postal_abbr')

    return df_wea_agg[['year', 'month', 'state_code', 'state', 'postal_abbr', 'zipcode', 'event_type', 'severity', 'duration_days']]


def aggregate_gdp_data(df_gdp):
    """
    flatten number of GDP for each year (columns) to rows & map 'state' column
    Aggregate by year, state, and industry of GDP
    """
    start_year, end_year = 2015, 2019

    # filtering industries
    col_include_linecode = [3, 6, 10, 11, 12, 34, 35, 36, 45, 50, 59, 68, 75, 82, 83]
    df_gdp = df_gdp[df_gdp['LineCode'].isin(col_include_linecode)]

    # mapping new industry names
    mapping_industry = {
        3: 'agriculture',
        6: 'mining',
        10: 'utilities',
        11: 'construction',
        12: 'manufacturing',
        34: 'wholesale_trade',
        35: 'retail_trade',
        36: 'transportation',
        45: 'information',
        50: 'finance',
        59: 'services',
        68: 'education',
        75: 'entertainment',
        82: 'others',
        83: 'government'
    }
    df_gdp['industry'] = df_gdp['LineCode'].map(mapping_industry)

    df_gdp_melt = pd.melt(df_gdp, id_vars=['GeoName', 'industry'],
                          value_vars=[str(yr) for yr in range(start_year, end_year + 1)], \
                          var_name='year', value_name='gdp_millions')
    # df_gdp_melt['Description'] = df_gdp_melt['Description'].str.strip()

    df_gdp_melt.columns = ['state', 'industry', 'year', 'gdp_millions']

    df_mapping_state = pd.read_csv(mapping_state_path)
    df_gdp_melt = df_gdp_melt.merge(df_mapping_state, how='inner', on='state')

    return df_gdp_melt[['year', 'state_code', 'state', 'postal_abbr', 'industry', 'gdp_millions']]


def aggregate_employment_data(df_emp):
    """
    flatten number of employment for each year (columns) to rows & map 'state' column
    Aggregate by year, state, and industry of employment
    """
    start_year, end_year = 2015, 2019

    # filtering industries
    col_include_linecode = [70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2001, 2002, 2010]
    df_emp = df_emp[df_emp['LineCode'].isin(col_include_linecode)]

    # mapping new industry names
    mapping_industry = {
        70: 'farm',
        100: 'forestry',
        200: 'mining',
        300: 'utilities',
        400: 'construction',
        500: 'manufacturing',
        600: 'wholesale_trade',
        700: 'retail_trade',
        800: 'transportation',
        900: 'information',
        1000: 'finance',
        1100: 'real_estate',
        1200: 'professional',
        1300: 'management',
        1400: 'administrative_support',
        1500: 'education',
        1600: 'healthcare',
        1700: 'entertainment',
        1800: 'accommodation_food',
        1900: 'others',
        2001: 'federal_civilian',
        2002: 'military',
        2010: 'state_local'
    }
    df_emp['industry'] = df_emp['LineCode'].map(mapping_industry)

    df_emp_melt = pd.melt(df_emp, id_vars=['GeoName', 'industry'],
                          value_vars=[str(yr) for yr in range(start_year, end_year + 1)], \
                          var_name='year', value_name='num_employment')

    # df_emp_melt['Description'] = df_emp_melt['Description'].str.strip()
    df_emp_melt.columns = ['state', 'industry', 'year', 'num_employment']

    df_mapping_state = pd.read_csv(mapping_state_path)
    df_emp_melt = df_emp_melt.merge(df_mapping_state, how='inner', on='state')

    return df_emp_melt[['year', 'state_code', 'state', 'postal_abbr', 'industry', 'num_employment']]


def write_to_csv(df_list, path_list):
    assert len(df_list) == len(path_list), 'different dataframe and output path lengths'
    for i in range(len(df_list)):
        df_list[i].to_csv(path_list[i], index=False)


if __name__ == '__main__':
    df_acc, df_wea, df_gdp, df_emp = import_raw_data(path_accident_data, path_weather_data, path_gdp_data, path_employment_data)


    df_acc_agg = aggregate_accident_data(df_acc)
    df_wea_agg = aggregate_weather_data(df_wea)
    df_gdp_agg = aggregate_gdp_data(df_gdp)
    df_emp_agg = aggregate_employment_data(df_emp)

    df_list = [df_acc_agg, df_wea_agg, df_gdp_agg, df_emp_agg]
    path_list = [path_output_accident_data, path_output_weather_data, path_output_gdp_data, path_output_employment_data]

    write_to_csv(df_list, path_list)









