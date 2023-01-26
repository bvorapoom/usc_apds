import sys
import pandas as pd
import numpy as np
import pmr as p
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.max_columns', 200)


def prep_data_for_training(method='pyspark', source='firebase'):
    year_start = 2016
    year_end = 2019

    #####
    # get data used for training using map-reduce functions
    #####

    ### get accident data
    query = """
        select year, month, postal_abbr, sum(count_accidents) as count_accidents
        from table_accident
        where year between """ + str(year_start) + ' and ' + str(year_end) + """
        group by 1, 2, 3
        """
    df_accident = p.pmr_query_main(method, query, source)


    ### get employment data
    query = """
            select year, postal_abbr, industry, num_employment
            from table_employment
            where year between """ + str(year_start) + ' and ' + str(year_end)
    df_employment = p.pmr_query_main(method, query, source)
    df_employment['num_employment'] = df_employment['num_employment'].apply(pd.to_numeric, errors='coerce')
    df_employment.dropna(inplace=True)

    # pivot num_employment by industry to columns
    df_employment = df_employment.pivot(values='num_employment',\
                           index=['year', 'postal_abbr'],\
                           columns=['industry']).reset_index()

    # use %total for num_employment
    df_employment.loc[:, ~df_employment.columns.isin(['year', 'postal_abbr'])] = \
            df_employment.loc[:, ~df_employment.columns.isin(['year', 'postal_abbr'])].\
            div(df_employment.loc[:, ~df_employment.columns.isin(['year', 'postal_abbr'])].sum(axis=1), axis=0)

    # rename columns
    df_employment.columns = [col if col in ['year', 'postal_abbr'] else 'numEmp_' + col for col in df_employment.columns]


    ### get GDP data
    query = """
            select year, postal_abbr, industry, gdp_millions
            from table_gdp
            where year between """ + str(year_start) + ' and ' + str(year_end)
    df_gdp = p.pmr_query_main(method, query, source)
    df_gdp['gdp_millions'] = df_gdp['gdp_millions'].apply(pd.to_numeric, errors='coerce')
    df_gdp.dropna(inplace=True)

    # pivot num_employment by industry to columns
    df_gdp = df_gdp.pivot(values='gdp_millions',\
                           index=['year', 'postal_abbr'],\
                           columns=['industry']).reset_index()

    # use %total for num_employment
    df_gdp.loc[:, ~df_gdp.columns.isin(['year', 'postal_abbr'])] = \
            df_gdp.loc[:, ~df_gdp.columns.isin(['year', 'postal_abbr'])].\
            div(df_gdp.loc[:, ~df_gdp.columns.isin(['year', 'postal_abbr'])].sum(axis=1), axis=0)

    # rename columns
    df_gdp.columns = [col if col in ['year', 'postal_abbr'] else 'gdp_' + col for col in df_gdp.columns]


    ### get Population data
    query = """
                select distinct year, postal_abbr, sex, race_code
                , sum(average_age * pop) over (partition by year, postal_abbr) / sum(pop) over (partition by year, postal_abbr) as avg_age
                , sum(pop) over (partition by year, postal_abbr) as pop_norm
                , sum(pop) over (partition by year, postal_abbr, sex, race_code) as population
                from table_population
                where year between """ + str(year_start) + ' and ' + str(year_end)
    df_population = p.pmr_query_main(method, query, source)
    df_population['population'] = df_population['population'].apply(pd.to_numeric, errors='coerce')
    df_population.dropna(inplace=True)

    # pivot population by sex to columns
    df_pop_sex = pd.pivot_table(df_population, values='population', \
                          index=['year', 'postal_abbr', 'avg_age', 'pop_norm'], \
                          columns=['sex'], aggfunc=sum).reset_index()
    df_pop_sex.loc[:, ['F', 'M']] = df_pop_sex.loc[:, ['F', 'M']]. \
            div(df_pop_sex.loc[:, ['F', 'M']].sum(axis=1), axis=0)
    df_pop_sex.columns = [col if col in ['year', 'postal_abbr', 'avg_age', 'pop_norm'] else 'pop_sex_' + col for col in df_pop_sex.columns]

    # pivot population by race_code to columns
    df_pop_race = pd.pivot_table(df_population, values='population', \
                                index=['year', 'postal_abbr', 'avg_age', 'pop_norm'], \
                                columns=['race_code'], aggfunc=sum).reset_index()
    df_pop_race.loc[:, ~df_pop_race.columns.isin(['year', 'postal_abbr', 'avg_age', 'pop_norm'])] = \
        df_pop_race.loc[:, ~df_pop_race.columns.isin(['year', 'postal_abbr', 'avg_age', 'pop_norm'])]. \
        div(df_pop_race.loc[:, ~df_pop_race.columns.isin(['year', 'postal_abbr', 'avg_age', 'pop_norm'])].sum(axis=1), axis=0)
    df_pop_race.columns = [col if col in ['year', 'postal_abbr', 'avg_age', 'pop_norm'] else 'pop_race_' + str(col) for col in
                          df_pop_race.columns]

    # merge df_sex and df_race
    df_population = df_pop_sex.merge(df_pop_race, how='inner', on=['year', 'postal_abbr', 'avg_age', 'pop_norm'])

    ### get Weather data
    query = """
            select year, month, postal_abbr, event_type, max(duration_days) as duration_days
            from table_weather
            where year between """ + str(year_start) + ' and ' + str(year_end) + """
            group by 1, 2, 3, 4
            """
    df_weather =  p.pmr_query_main(method, query, source)

    # pivot num_employment by industry to columns
    df_weather = df_weather.pivot(values='duration_days',\
                           index=['year', 'month', 'postal_abbr'],\
                           columns=['event_type']).reset_index()

    df_weather.fillna(0, inplace=True)

    # rename columns
    df_weather.columns = [col if col in ['year', 'month', 'postal_abbr'] else 'weather_' + col for col in df_weather.columns]


    #####
    # Join 5 datasets to create training data
    #####
    df_train = df_accident. \
        merge(df_employment, how='left', on=['year', 'postal_abbr']). \
        merge(df_gdp, how='left', on=['year', 'postal_abbr']). \
        merge(df_population, how='left', on=['year', 'postal_abbr']). \
        merge(df_weather, how='left', on=['year', 'month', 'postal_abbr'])

    df_train.fillna(0, inplace=True)

    return df_train


def prepare_training_data(df):
    df['count_accidents_norm'] = df['count_accidents'] / df['pop_norm'] * 10e4
    df = df[df['count_accidents_norm'] < df['count_accidents_norm'].quantile(0.95)]
    col_to_drop = ['year', 'month', 'postal_abbr', 'count_accidents', 'pop_norm',\
                   'pop_race_6', 'pop_sex_M', 'gdp_others', 'numEmp_others']
    df.drop(columns=col_to_drop, inplace=True)

    # split dependent & independent vars
    df_X = df.loc[:, df.columns != 'count_accidents_norm']
    df_y = df.loc[:, 'count_accidents_norm']

    # split training and testing sets
    df_train_X, df_test_X, df_train_y, df_test_y = train_test_split(df_X, df_y,\
                            test_size=0.25, random_state=0)

    # standardize independent vars
    scaler = StandardScaler()
    df_train_X = pd.DataFrame(scaler.fit_transform(df_train_X), columns=df_train_X.columns)
    df_test_X = pd.DataFrame(scaler.transform(df_test_X), columns=df_test_X.columns)

    return df_train_X, df_test_X, df_train_y, df_test_y


def train_regression(df_train_X, df_test_X, df_train_y, df_test_y, algo):
    assert algo in ['linreg', 'ridge', 'lasso', 'poly', 'tree', 'linsvm', 'svm', 'knn', 'rf']

    if algo == 'linreg':
        reg = LinearRegression()
        reg.fit(df_train_X, df_train_y)
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'ridge':
        reg_temp = Ridge(random_state=0)
        params = {'alpha': [10e-2, 10e-1, 1, 10, 100]}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'lasso':
        reg_temp = Lasso(random_state=0)
        params = {'alpha': [10e-2, 10e-1, 1, 10, 100]}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'poly':
        poly = PolynomialFeatures(degree=2)
        df_train_X = poly.fit_transform(df_train_X)
        df_test_X = poly.fit_transform(df_test_X)
        reg = LinearRegression()
        reg.fit(df_train_X, df_train_y)
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'tree':
        reg_temp = DecisionTreeRegressor(random_state=0)
        params = {'max_depth': [1, 2, 3, 5, 10, 20, 50],\
                  'ccp_alpha': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'linsvm':
        reg_temp = LinearSVR(random_state=0)
        params = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 10e2, 10e3, 10e4, 10e5]}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'svm':
        reg_temp = SVR(kernel='rbf')
        params = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 10e2, 10e3, 10e4, 10e5]}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'knn':
        reg_temp = KNeighborsRegressor()
        params = {'n_neighbors': np.arange(50, 200, 10),\
                       'weights': ['uniform', 'distance']}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse
    elif algo == 'rf':
        reg_temp = RandomForestRegressor(random_state=0)
        params = {'max_depth': [1, 2, 3, 5, 10, 20], \
                       'n_estimators': [10, 20, 50, 100, 200, 500]}
        reg_cv = GridSearchCV(reg_temp, params, cv=5, scoring='neg_root_mean_squared_error')
        reg_cv.fit(df_train_X, df_train_y)
        # get the best model
        reg = reg_cv.best_estimator_
        y_pred = reg.predict(df_test_X)
        rmse = mean_squared_error(df_test_y, y_pred, squared=False)
        return reg, rmse

def predict(input_df, algo='linreg'):
    # get training data
    train_data_path = 'data/cleaned/data_training.csv'
    df = pd.read_csv(train_data_path)

    # prep data
    df['count_accidents_norm'] = df['count_accidents'] / df['pop_norm'] * 10e4
    df = df[df['count_accidents_norm'] < df['count_accidents_norm'].quantile(0.95)]
    col_to_drop = ['year', 'month', 'postal_abbr', 'count_accidents', 'pop_norm', \
                   'pop_race_6', 'pop_sex_M', 'gdp_others', 'numEmp_others']
    df.drop(columns=col_to_drop, inplace=True)

    # split dependent & independent vars
    df_X = df.loc[:, df.columns != 'count_accidents_norm']
    df_y = df.loc[:, 'count_accidents_norm']

    # split training and testing sets
    df_train_X, df_test_X, df_train_y, df_test_y = train_test_split(df_X, df_y, \
                                                                    test_size=0.25, random_state=0)

    # select columns
    columns = list(input_df.columns)
    df_train_X = df_train_X.loc[:, columns]
    df_test_X = df_test_X.loc[:, columns]

    # Standardize
    scaler = StandardScaler()
    df_train_X = pd.DataFrame(scaler.fit_transform(df_train_X), columns=df_train_X.columns)
    df_test_X = pd.DataFrame(scaler.transform(df_test_X), columns=df_test_X.columns)

    # train
    reg, rmse = train_regression(df_train_X, df_test_X, df_train_y, df_test_y, algo)

    # process test data
    input_df = scaler.transform(input_df)

    # predict
    pred = reg.predict(input_df)

    return pred



if __name__ == '__main__':
    # get data using map-reduce and generate training data from joining 5 data sources
    df = prep_data_for_training(method='pyspark', source='mysql')
    # train_data_path = 'data/cleaned/data_training.csv'
    # df.to_csv(train_data_path, index=False)
    # print(df)

    # read training data from csv
    # train_data_path = 'data/cleaned/data_training.csv'
    # df = pd.read_csv(train_data_path)

    # df['count_accidents_norm'] = df['count_accidents'] / df['pop_norm'] * 10e4
    # df = df[df['count_accidents_norm'] < df['count_accidents_norm'].quantile(0.95)]
    # df.drop(columns=['year', 'month', 'postal_abbr', 'count_accidents', 'pop_norm'], inplace=True)
    # print(df.describe())

    # split training/testing, normalize data
    df_train_X, df_test_X, df_train_y, df_test_y = prepare_training_data(df)
    # print(df_train_X.describe())

    # train
    reg, rmse = train_regression(df_train_X, df_test_X, df_train_y, df_test_y, 'linreg')

    temp = {df_train_X.columns[i]: reg.coef_[i] for i in range(len(reg.coef_))}
    temp = sorted(temp.items(), key=lambda r: (-abs(r[1]), r[0]))

    print(temp)
    print(rmse)
    # print(df[df.num_employment == '(D)'])

