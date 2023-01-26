import proj_getdata as dat # files that contain all functions for getting and storing data
import proj_utils as utils # files that contains functions to help with analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action = 'store_true', help = 'get data from database (pre-scraped and pre-crawled) for analysis')
    args = parser.parse_args()    


    if args.static:
        raw_expos_df = dat.main_getdata_pm25_exposure(mode = 'database')
        th_city_list, pop_df, wt_df, aq_df = dat.main_getdata_corr_ml(mode = 'database')
    else:
        raw_expos_df = dat.main_getdata_pm25_exposure(mode = 'remote')
        th_city_list, pop_df, wt_df, aq_df = dat.main_getdata_corr_ml(mode = 'remote')




    # ------------------------------------------------------------------------
    # # PART 1 - Program to calculate PM2.5 exposure based on visited locations
    # ------------------------------------------------------------------------



    print('PART 1 - Program to calculate PM2.5 exposure based on visited locations')
    time.sleep(3)


    # getting PM2.5 exposure by visited locations
    print('here is the first 5 entries of PM2.5 data based on your visited location')
    print(raw_expos_df.head())
    time.sleep(1)




    # showing average PM2.5 exposure overall
    print('(OVERALL) Your average PM2.5 exposure based on visited locations is')
    print(round(raw_expos_df['PM25 value'].mean(), 1), 'ug/m3\n')

    # process data for viz
    clean_expos_df = utils.clean_grouping(raw_expos_df)   
    agg_result = utils.aggregate_aq_df_show_average_by_locations(clean_expos_df)
    time.sleep(1)

    # set plot style
    sns.set_style('whitegrid')

    # showing average PM2.5 exposure by location (table and bar plot)
    print('Your average PM2.5 exposure by locations is ', agg_result, sep = '\n\n')
    utils.bar_plot_avg_exposure_by_location(agg_result)
    time.sleep(1)


    # showing daily PM2.5 exposure 
    print('Here is your PM2.5 exposure daily:')
    utils.visualize_pm25_exposure_daily(clean_expos_df)
    time.sleep(1)



    # ------------------------------------------------------------------------
    # # PART 2 - Correlation / ML Model between Population / Weather / Air Quality
    # ------------------------------------------------------------------------


    # **Getting data of list of Thailand cities, population and density, weather, and air quality**


    print('Here are the first 5 entries for each dataaset:')
    print(th_city_list[:5], pop_df.head(5), wt_df.head(5), aq_df.head(5), sep = '\n\n')
    time.sleep(1)

    # **Visualizing basic stats on Thailand's population**
    # 
    # - Only Bangkok has significantly high population. Other provinces are more or less on the same level
    # - On population density, Bangkok is also the most densely populated province. Nonthaburi and Samut Prakan ranked 2nd and 3rd.



    print('Visualizing Thailand population data...')
    utils.visualize_thailand_population(pop_df)


    # **Visualizing basic stats on weather data**
    # - Precipitation, Chance of rain, and Visibility attributes are having values skewed towards one side. Therefore, these information might not be useful when seeing correlation or training the model



    print('Visualizing Thailand weather data...')
    utils.visualize_weather_data(wt_df)


    # **Visualizing basic stats on air quality data**
    # - Average PM2.5 level for each province in Thailand ranges from as low as 5 ug/m3 to 30 ug/m3



    print('Visualizing Thailand air quality data...')
    utils.visualize_airquality(aq_df)


    # **Finding correlation between population / density versus air quality**

    print('Finding correlation between population / density versus air quality...')
    # aggregating air quality data to be by city
    agg_aq_df = aq_df.groupby('City/Location')['PM25 value'].mean().reset_index()

    # joining air quality & population/density dataframes together
    df_join_pop_aq = pop_df.merge(agg_aq_df, left_on = 'City', right_on = 'City/Location', how = 'inner')
    df_join_pop_aq = df_join_pop_aq[['City', 'Area (km2)', 'Population', 'Population Density (/km2)', 'PM25 value']]

    print(df_join_pop_aq.head())


    # *Correlation between population / population density versus air quality*
    # - There is a slightly positive correlation between population and PM2.5 as the R^2 value is 0.25
    # - The R^2 of correlation between population density and PM2.5 is 0.37. This is sensible as PM2.5 should be more of a function of population density than population itself.


    

    fig, ax = plt.subplots(1, 2, figsize = (20, 5))
    df_join_pop_aq['log10 Population'] = np.log10(df_join_pop_aq.Population)
    utils.corr_plot_with_r2(ax[0], df_join_pop_aq, 'log10 Population', 'PM25 value')

    df_join_pop_aq['log10 Density'] = np.log10(df_join_pop_aq['Population Density (/km2)'])
    utils.corr_plot_with_r2(ax[1], df_join_pop_aq, 'log10 Density', 'PM25 value')

    plt.show()



    print('Finding correlation between weather versus air quality...')

    # prepping data for finding correlation between weather versus air quality
    df_join_wt_aq = wt_df.merge(aq_df, left_on = ['City', 'Date', 'Hour'], right_on = ['City/Location', 'Date', 'Hour'], how = 'inner')
    df_join_wt_aq = df_join_wt_aq.drop(['City/Location', 'Country Code'], axis=1)
    df_join_wt_aq[['Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 'Precipitation (mm)', 
                   'Humidity', 'Cloud','Chance of Rain', 'Visibility (km)', 'PM25 value']] = df_join_wt_aq[['Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 'Precipitation (mm)', 
                   'Humidity', 'Cloud','Chance of Rain', 'Visibility (km)', 'PM25 value']].astype(float)

    print(df_join_wt_aq.head())


    # *Correlation between weather and air quality data*
    # - Each attribute for weather data doesn't show a strong correlation with air quality. R2 value is also relatively low.




    utils.corr_plot_weather_airquality(df_join_wt_aq)


    # **Training Regression Models to predict PM2.5 values based on population and weather data**
    # - This part comes up since I wondered if we can use multiple variables to predict how PM2.5 value will be, when looking at a single variable at a time doesn't give a strong correlation
    # - Algorithms picked for this analysis is multiple linear regression, polynomial regression, and decision tree regressor
    # - Please note that this analysis might not be 100% correct. I just did my own research and try to test with a few algorithms. So with time limited, each algo uses default parameter without the best preprocessing or hyperparameter tuning.




    # import ML packages
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor 





    # join all 3 data sources
    combined_df = df_join_wt_aq.merge(pop_df, on = 'City', how = 'inner')
    combined_df = combined_df.drop(['City', 'Date', 'Hour'], axis = 1)

    # defining X and y
    x_cols = ['Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 'Precipitation (mm)', 'Humidity', 'Cloud',
                     'Chance of Rain', 'Visibility (km)', 'Area (km2)', 'Population', 'Population Density (/km2)']
    X = combined_df[x_cols]
    y = combined_df[['PM25 value']]

    # split to training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    # creating empty result dataframe for storing each model's results
    reg_result = pd.DataFrame(columns = ['rmse', 'r2'])





    # modelling multivariate linear regression with StandardScaling

    print('Trying to model PM2.5 based on population and weather data')
    time.sleep(2)
    print('Below are RMSE and R2 for different 5 regression algorithms:')


    # scaling / normalization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

    # fit model
    linreg = LinearRegression()
    linreg.fit(X_train_scaled, y_train)

    # make prediction
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_train.columns)
    y_test_pred = linreg.predict(X_test_scaled)

    linreg_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    linreg_r2 = r2_score(y_test, y_test_pred)

    reg_result.loc['linear regression', ['rmse', 'r2']] = [linreg_rmse, linreg_r2]

    # print('Multivariate Linear Regression: RMSE of testing dataset is {:.2f}'.format(linreg_rmse))
    # print('Multivariate Linear Regression: R2 of testing dataset is {:.2f}'.format(linreg_r2))






    # try polynomial features linear regression with StandardScaling (degree 2-4)
    num_degrees = range(2, 5)

    for deg in num_degrees:
        # generate polynomial features for X
        polynomial_features = PolynomialFeatures(degree = deg)
        X_train_poly = polynomial_features.fit_transform(X_train)
        
        # scaling
        scaler = StandardScaler()
        scaler.fit(X_train_poly)
        X_train_poly_scaled = scaler.transform(X_train_poly)

        # fit model
        linreg = LinearRegression()
        linreg.fit(X_train_poly_scaled, y_train)

        # make prediction
        X_test_poly_scaled = scaler.transform(polynomial_features.transform(X_test))
        y_test_pred = linreg.predict(X_test_poly_scaled)

        linreg_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
        linreg_r2 = r2_score(y_test, y_test_pred)

        reg_result.loc['polynomial regression deg ' + str(deg), ['rmse', 'r2']] = [linreg_rmse, linreg_r2]

        # print('Polynomial Regression Degree {}: RMSE of testing dataset is {:.2f}'.format(deg, linreg_rmse))
        # print('Polynomial Regression Degree {}: R2 of testing dataset is {:.2f}'.format(deg, linreg_r2))





    # try decision tree regressor
      
    # create a regressor object
    dtreg = DecisionTreeRegressor(random_state = 0) 
      
    # fit model
    dtreg.fit(X_train, y_train)
    y_train_pred = dtreg.predict(X_train)

    # make prediction
    y_test_pred = dtreg.predict(X_test)

    dtreg_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    dtreg_r2 = r2_score(y_test, y_test_pred)

    reg_result.loc['decision tree regressor', ['rmse', 'r2']] = [dtreg_rmse, dtreg_r2]

    # print('Decision Tree Regressor: RMSE of testing dataset is {:.2f}'.format(dtreg_rmse))
    # print('Decision Tree Regressor: R2 of testing dataset is {:.2f}'.format(dtreg_r2))


    # **Results from different regression models**
    # - Metrics considered here include RMSE and R2 on the testing datasets
    # - Linear Regression model gives the worst RMSE and R2 compared to the best polynomial regression and decision tree
    # - With polynomial regression, degree of 2 gives a better result than linear while having the degree of 3 gives the best results with RMSE of 4.9 and R2 of 0.63. However, degree of 4 might overfit the model too much and gives bad results
    # - Decision tree regressor works best among the other two algorithms as its RMSE is the lowest at 3.19 and its R2 is the highest at 0.85




    print(reg_result)

