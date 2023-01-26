import pandas as pd
import datetime
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math


def clean_grouping(raw_expos_df):
    i = 1
    df_copy = raw_expos_df.copy()
    temp = df_copy.loc[0, 'City/Location']
    df_copy.loc[0, 'grouping'] = i
    for ind, row in df_copy.iloc[1:].iterrows():
        if row['City/Location'] == temp:
            df_copy.loc[ind, 'grouping'] = i
        else:
            i += 1
            temp = df_copy.loc[ind, 'City/Location']
            df_copy.loc[ind, 'grouping'] = i
    df_copy['grouping'] = df_copy.grouping.astype(int).astype(str)
    df_copy['new_Date'] = df_copy['Date'].apply(lambda x: pd.to_datetime(x).strftime('%d-%b'))
    df_copy['new_group'] = df_copy['City/Location'] + ': ' +  df_copy['new_Date'].groupby(df_copy['grouping']).transform('min') \
        + ' to ' + df_copy['new_Date'].groupby(df_copy['grouping']).transform('max')
    return df_copy


def aggregate_aq_df_show_average_by_locations(clean_expos_df):
    agg_result = clean_expos_df.groupby(['City/Location', 'grouping']).agg(date_from = ('Date', 'min'), date_to = ('Date', 'max'), avg_PM25 = ('PM25 value', 'mean'))
    agg_result = agg_result.sort_values('date_from').reset_index()[['date_from', 'date_to', 'City/Location', 'avg_PM25']]
    agg_result['avg_PM25'] = agg_result['avg_PM25'].round(2)
    agg_result[['date_from', 'date_to']] = agg_result[['date_from', 'date_to']].apply(lambda x: pd.to_datetime(x).dt.strftime('%d-%b'))
    return agg_result


def bar_plot_avg_exposure_by_location(agg_result):
    temp = agg_result.copy()
    temp['Location'] = temp['City/Location'] + ' :\n' + temp['date_from'] + ' to ' + temp['date_to']
    
    color_labels = temp['City/Location'].unique()
    color_pal = sns.color_palette('pastel', len(color_labels))
    color_map = dict(zip(color_labels, color_pal))
    
    plt.figure(figsize = (13, 6)) 
    plt.bar(temp['Location'], temp['avg_PM25'], color = temp['City/Location'].map(color_map))
    
    plt.xlabel('Location / Date')
    plt.ylabel('PM2.5 Exposure (ug/m3)')
    plt.title('Average PM2.5 Exposure by Locations')
    plt.show()

    
def visualize_pm25_exposure_daily(clean_expos_df):
    color_labels = clean_expos_df['City/Location'].unique()
    color_pal = sns.color_palette('pastel', len(color_labels))
    color_map = dict(zip(color_labels, color_pal))
    
    plt.figure(figsize = (17, 8)) 
    
    for i in clean_expos_df['new_group'].unique():
        temp_df = clean_expos_df[clean_expos_df['new_group'] == i]
        plt.bar(temp_df['new_Date'], temp_df['PM25 value'], 
                 color = temp_df['City/Location'].map(color_map), label = temp_df['City/Location'].unique()[0])
    
    plt.axhline(y = 35.5, color = 'r', linestyle = '-')
    x_pos = clean_expos_df['new_Date'][int(math.floor(len(clean_expos_df)/2))]
    plt.text(x_pos , 36, 'Unhealthy PM2.5 Level at 35.5 ug/m3', horizontalalignment = 'center', color = 'red')
    
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Exposure (ug/m3)')
    plt.title('Daily PM2.5 Exposure based on Visited Locations')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.show()



def visualize_thailand_population(pop_df):
    rank_pop = pop_df.sort_values('Population', ascending = False).reset_index(drop = True)
    rank_den = pop_df.sort_values('Population Density (/km2)', ascending = False).reset_index(drop = True)
    
    sns.set_style('whitegrid')
    
    plt.figure(figsize = (15, 5)) 
    plt.bar(rank_pop['City'], rank_pop['Population'], color = 'b')
    
    plt.xlabel('Thailand\'s Cities ')
    plt.ylabel('Population')
    plt.title('Thailand Population by Cities')
    plt.xticks(rotation = 90)
    plt.show()
    
    plt.figure(figsize = (15, 5)) 
    plt.bar(rank_den['City'], rank_den['Population Density (/km2)'], color = 'g')
    
    plt.xlabel('Thailand\'s Cities ')
    plt.ylabel('Population Density (/km2)')
    plt.title('Thailand Population Density by Cities')
    plt.xticks(rotation = 90)
    plt.show()


def visualize_weather_data(wt_df):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(4, 2, figsize=(20,20))
    wt_cols = ['Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 'Precipitation (mm)', 
               'Humidity', 'Cloud','Chance of Rain', 'Visibility (km)']
    for i in range(len(wt_cols)):
        ind_x = math.floor(i / 2)
        ind_y = i % 2
        temp_df = wt_df[[wt_cols[i]]].astype(float)
        temp_df.hist(ax = ax[ind_x, ind_y], bins = 50)
        ax[ind_x, ind_y].set_title(wt_cols[i])
        ax[ind_x, ind_y].set_xlabel(wt_cols[i])
        ax[ind_x, ind_y].set_ylabel('count')
        
    fig.suptitle('Thailand Cities Weather Data')
    plt.show()


def visualize_airquality(aq_df):
    sns.set_style('whitegrid')
    
    plt.figure(figsize = (15, 5)) 
    sort_order = list(aq_df.groupby('City/Location')['PM25 value'].mean().sort_values(ascending = False).keys())
    sns.barplot(data = aq_df, x = 'City/Location', y = 'PM25 value', order = sort_order, color = 'orange')
    plt.xticks(rotation = 90)
    plt.title('Average PM2.5 Values by Thailand Cities')
    plt.xlabel('Thailand Cities')
    plt.ylabel('PM2.5 Value (ug/m3)')
    plt.show()


def corr_plot_with_r2(ax, df, x_col, y_col):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
    sns.regplot(ax = ax, x = x_col, y = y_col, data = df, line_kws = {'label':"R^2 = {0:.4f}".format(r_value)})
    ax.legend()


def corr_plot_weather_airquality(df):
    fig, ax = plt.subplots(4, 2, figsize=(20,20))
    wt_cols = ['Temperature (C)', 'Wind Speed (kph)', 'Pressure (mb)', 'Precipitation (mm)', 
               'Humidity', 'Cloud','Chance of Rain', 'Visibility (km)']
    aq_col = 'PM25 value'
    for i in range(len(wt_cols)):
        ind_x = math.floor(i / 2)
        ind_y = i % 2
        corr_plot_with_r2(ax[ind_x, ind_y], df, wt_cols[i], aq_col)
        
    plt.show()





