import sys
import requests
import pandas as pd
import numpy as np
import firebaseUpload as f
from pyspark.sql import SparkSession
import re
import mysqlupload as s
import mongodbUpload as mg

firebase_url = 'https://dsci551-project-40a58-default-rtdb.firebaseio.com/'


def pmr_get_data(method, file_name, source='firebase'):
    assert method in ['custom', 'pyspark'], 'please input valid method'
    assert source in ['firebase', 'mysql', 'mongodb'], 'please input valid source to read data from'

    if source == 'firebase':
        num_partition = len(f.get_partition_locations(firebase_url, file_name))
        df_partitioned = list()
        for num_partition in range(1, num_partition + 1):
            temp = f.read_partition(firebase_url, file_name, num_partition)
            df_partitioned.append(temp)
    elif source == 'mysql':
        num_partition = len(s.getPartitionLocations(file_name))
        df_partitioned = list()
        for num_partition in range(1, num_partition + 1):
            temp = s.readPartition(file_name, num_partition)
            df_partitioned.append(temp)
    elif source == 'mongodb':
        num_partition = len(mg.get_partition_locations(file_name))
        df_partitioned = list()
        for num_partition in range(1, num_partition + 1):
            temp = mg.read_partition(file_name, num_partition)
            df_partitioned.append(temp)


    # output data
    if method == 'custom':
        return df_partitioned
    elif method == 'pyspark':
        df = pd.concat(df_partitioned, ignore_index=True)

        # Create PySpark SparkSession
        spark = SparkSession.builder \
            .appName('dsci551_project') \
            .getOrCreate()
        # Create PySpark DataFrame from Pandas
        sparkDF = spark.createDataFrame(df)
        return sparkDF



def deconstruct_query(query):

    kws = ['select', 'from', 'where', 'group by', 'order by', 'limit']

    query_split = re.split(r'\s*(?:select|from|where|group by|order by|limit)\s*', query, flags=re.IGNORECASE)[1:]

    i = 0
    dict_decon = dict()
    for kw in kws:
        if kw in query.lower():
            dict_decon[kw] = query_split[i]
            i += 1

    # format select
    list_final_name, list_aggfunc, list_initial_name = list(), list(), list()
    for item in dict_decon['select'].split(','):
        split_comp = item.strip().split()
        if len(split_comp) == 1:
            list_final_name.append(item.strip())
            list_initial_name.append(item.strip())
        else:
            list_final_name.append(split_comp[-1])
            split_comp = split_comp[0].strip(')').split('(')
            if split_comp[0] == 'avg':
                list_aggfunc.append(np.mean)
            else:
                list_aggfunc.append(split_comp[0])
            list_initial_name.append(split_comp[1])


    dict_decon['select'] = (list_final_name, list_aggfunc, list_initial_name)

    # format group by
    if 'group by' in dict_decon.keys():
        list_groupby = list()
        for item in dict_decon['group by'].split(','):
            item = item.strip()
            try:
                attr = dict_decon['select'][0][int(item.split()[0].strip()) - 1]
                list_groupby.append(attr)
            except:
                list_groupby.append(item.split()[0].strip())

        dict_decon['group by'] = list_groupby


    # format order by
    if 'order by' in dict_decon.keys():
        list_attr, list_order_asc = list(), list()
        for item in dict_decon['order by'].split(','):
            item = item.strip()
            try:
                attr = dict_decon['select'][0][int(item.split()[0].strip()) - 1]
                list_attr.append(attr)
            except:
                list_attr.append(item.split()[0].strip())

            if 'desc' in item.lower():
                list_order_asc.append(False)
            else:
                list_order_asc.append(True)

        dict_decon['order by'] = (list_attr, list_order_asc)

    # format limit
    if 'limit' in dict_decon.keys():
        dict_decon['limit'] = int(dict_decon['limit'])

    return dict_decon


def pmr_filter_custom(df_partitioned, filter_query_clause):
    output_filtered = list()

    filter_query_clause = filter_query_clause.replace('=', '==')

    for df in df_partitioned:
        temp = df.query(filter_query_clause)
        output_filtered.append(temp)

    return output_filtered


def pmr_selection_custom(df_partitioned, list_selection):
    output_filtered = list()

    for df in df_partitioned:
        temp = df.loc[:, list_selection]
        output_filtered.append(temp)

    return output_filtered


def pmr_groupby_custom(df_reduce, col_val, col_groupby, aggfunc_list, col_initial):
    temp = pd.pivot_table(df_reduce, values=col_val, index=col_groupby, \
                          aggfunc={col_val[i]: aggfunc_list[i] for i in range(len(col_val))})

    temp = temp.reset_index()
    temp = temp[col_initial]

    return temp


def pmr_query_main(method, query, source='firebase'):
    list_tables = re.findall(r'\btable\w+', query)
    if source == 'firebase':
        file_name_mapping = {
            'table_accident': 'data_accident_2016_2021.csv',
            'table_employment': 'data_employment_2015_2019.csv',
            'table_gdp': 'data_gdp_2015_2020.csv',
            'table_population': 'data_pop_2015_2019.csv',
            'table_weather': 'data_weather_2016_2021.csv'
        }
    elif source == 'mysql':
        file_name_mapping = {
            'table_accident': 'data/car_accidents/accident',
            'table_employment': 'data/employment/employment',
            'table_gdp': 'data/gdp/gdp',
            'table_population': 'data/pop/pop',
            'table_weather': 'data/weather/weather'
        }
    elif source == 'mongodb':
        file_name_mapping = {
            'table_accident': 'data/car_accidents/data_accident_2016_2021.csv',
            'table_employment': 'data/employment/data_employment_2015_2019.csv',
            'table_gdp': 'data/gdp/data_gdp_2015_2020.csv',
            'table_population': 'data/pop/data_pop_2015_2019.csv',
            'table_weather': 'data/weather/data_weather_2016_2021.csv'
        }

    assert all([s in file_name_mapping.keys() for s in list_tables]), 'please input valid table names'

    if method == 'custom':
        try:
            query_decon = deconstruct_query(query)

            # get data from edfs
            file_name = file_name_mapping[query_decon['from']]
            df_partitioned = pmr_get_data(method, file_name, source)

            # filter partitioned data
            if 'where' in query_decon.keys():
                df_partitioned = pmr_filter_custom(df_partitioned, query_decon['where'])

            # select
            df_partitioned = pmr_selection_custom(df_partitioned, query_decon['select'][2])

            # reduce: concat partitioned dataframe to one
            if 'group by' in query_decon.keys():
                df_reduce = pd.concat(df_partitioned, ignore_index=True)
                col_val = [col for col in query_decon['select'][2] if col not in query_decon['group by']]
                col_groupby = query_decon['group by']
                aggfunc_list = query_decon['select'][1]
                col_initial = query_decon['select'][2]

                df_reduce = pmr_groupby_custom(df_reduce, col_val, col_groupby, aggfunc_list, col_initial)
                df_reduce.columns = query_decon['select'][0]
            else:
                df_reduce = pd.concat(df_partitioned, ignore_index=True)

            # order by
            if 'order by' in query_decon.keys():
                attr_sort = query_decon['order by'][0]
                attr_asc = query_decon['order by'][1]
                df_reduce.sort_values(by=attr_sort, ascending=attr_asc, inplace=True)
                df_reduce.reset_index(inplace=True, drop=True)

            # limit
            if 'limit' in query_decon.keys():
                df_reduce = df_reduce.head(query_decon['limit'])

            return df_reduce

        except:
            return pmr_query_main('pyspark', query, source)
    elif method == 'pyspark':
        try:
            spark = SparkSession.builder \
                .appName('dsci551_project') \
                .getOrCreate()
            for table in list_tables:
                file_name = file_name_mapping[table]
                temp_sparkDF = pmr_get_data(method, file_name, source)
                temp_sparkDF.createOrReplaceGlobalTempView(table)
            query = query.replace('table', 'global_temp.table')
            result = spark.sql(query)
            return result.toPandas()
        except:
            return 'please input valid query'


if __name__ == '__main__':
    method = 'pyspark'
    source = 'mongodb'

    # query = 'select year, month, duration_days from table_weather where zipcode = 90007 and event_type = "Rain" and severity = "Moderate" order by year, month limit 10'

    # query = 'select state, gdp_millions from table_gdp where year = 2018 and industry = "education" order by gdp_millions desc'

    # query = 'select state, sum(count_accidents) as total_accidents from table_accident group by state order by 2 desc limit 5'

    # query = 'select state, sum(gdp_millions) from table_gdp where year = 2016 group by state'

    query = """
        select a.state, sum(accidents)/sum(population) as accidents_per_capita
        from (select postal_abbr, year, state, sum(count_accidents) as accidents from table_accident where year = 2019 group by 1, 2, 3) a
        left join (select postal_abbr, year, state, sum(pop) as population from table_population where year = 2019 group by 1, 2, 3) p
            on a.postal_abbr = p.postal_abbr and a.year = p.year
        group by 1
        order by 2 desc
        """

    query = 'select * from table_gdp limit 5'


    # temp = pmr_query_main(method, query, source)
    # print(temp)

    temp = pmr_get_data(method, 'data/gdp/data_gdp_2015_2020.csv', source)
    print(temp)

