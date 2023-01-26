import requests
import pandas as pd
import json

# input start and end years to get the data from API
start_year = 2015
end_year = 2019

# define path for output file of population data
output_path = 'data/cleaned/data_pop_' + str(start_year) + '_' + str(end_year) + '.csv'

# define path to get the mapping CSV files
mapping_agegroup_path = 'data/cleaned/mapping/mapping_pop_agegroup.csv'
mapping_race_path = 'data/cleaned/mapping/mapping_pop_race.csv'
mapping_state_path = 'data/cleaned/mapping/mapping_pop_state.csv'


def get_data(start_year, end_year):
    df_output = pd.DataFrame(columns=['pop', 'agegroup_code', 'race_code', 'sex_code', 'state_code', 'year'])
    for year in range(start_year, end_year + 1):
        # GET request
        url = 'https://api.census.gov/data/' + str(year) + '/pep/charagegroups?get=POP,AGEGROUP,RACE,SEX&for=state:*'
        res = requests.get(url)

        # read response to pandas dataframe
        df = pd.DataFrame(json.loads(res.text)[1:], columns=['pop', 'agegroup_code', 'race_code', 'sex_code', 'state_code'])
        df = df.astype(float).round().astype(int) # change datatype to int, found some anomalies where num population is float
        df['year'] = year # add year column

        df_output = df_output.append(df, ignore_index=True)

    return df_output


def filter_data(df_output):
    df_filtered = df_output[(df_output.agegroup_code >= 1) & (df_output.agegroup_code <= 18) & \
                            (df_output.race_code >= 1) & (df_output.race_code <= 6) & \
                            (df_output.sex_code != 0)]
    return df_filtered


def map_data(df_filtered):
    df_mapping_agegroup = pd.read_csv(mapping_agegroup_path)
    df_mapping_race = pd.read_csv(mapping_race_path)
    df_mapping_state = pd.read_csv(mapping_state_path)

    df_mapped = df_filtered.merge(df_mapping_agegroup, how='left', left_on='agegroup_code', right_on='age_group_code')
    df_mapped = df_mapped.merge(df_mapping_race, how='left', on='race_code')
    df_mapped = df_mapped.merge(df_mapping_state, how='left', on='state_code')
    df_mapped['sex'] = df_mapped['sex_code'].map({1:'M', 2:'F'})

    df_mapped = df_mapped[['year', 'agegroup_code', 'min_age', 'max_age', 'average_age', 'race_code',\
                           'race', 'state_code', 'state', 'postal_abbr', 'sex_code', 'sex', 'pop']]

    return df_mapped


def write_to_csv(df, path):
    df.to_csv(path, index=False)


if __name__ == '__main__':
    # get population data for the specified years from API and output as pandas df
    df_output = get_data(start_year, end_year)
    # filter data so that it doesn't contain any duplicates > age group, race, sex
    df_filtered = filter_data(df_output)
    # map columns (age group, race, state) with the mapping read from CSV
    df_mapped = map_data(df_filtered)
    # write output files
    write_to_csv(df_mapped, output_path)




