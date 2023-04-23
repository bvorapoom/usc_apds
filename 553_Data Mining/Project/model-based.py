import sys
from pyspark import SparkContext, SparkConf
import time
import json
import xgboost as xgb
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 50)


def getDataBusiness(sc, input_folder_path):
    # read business.json to get attributes: latitude, longitude, stars, review_count
    path_business = input_folder_path + 'business.json'
    rdd_business = sc.textFile(path_business).\
        map(lambda r: json.loads(r)).\
        map(lambda r: (r['business_id'], r['latitude'], r['longitude'], r['stars'], r['review_count']))

    # read photo.json to get attributes: num_photo
    path_photo = input_folder_path + 'photo.json'
    rdd_photo = sc.textFile(path_photo). \
        map(lambda r: json.loads(r)). \
        map(lambda r: (r['business_id'], r['photo_id'])).\
        groupByKey().\
        map(lambda r: (r[0], len(r[1])))


    rdd_business_merge = rdd_business.keyBy(lambda r: r[0]).leftOuterJoin(rdd_photo.keyBy(lambda r: r[0])).\
        map(lambda r: (r[0], r[1][0][1], r[1][0][2], r[1][0][3], r[1][0][4], r[1][1])).\
        map(lambda r: (r[0], r[1], r[2], r[3], r[4], r[5][1]) if r[5] is not None else (r[0], r[1], r[2], r[3], r[4], 0))

    return rdd_business_merge


def getDataUser(sc, input_folder_path):
    # read user.json to get attributes: review_count, average_stars, useful, funny, cool, fans
    path_user = input_folder_path + 'user.json'
    rdd_user = sc.textFile(path_user). \
        map(lambda r: json.loads(r)). \
        map(lambda r: (r['user_id'], r['review_count'], r['average_stars'], r['useful'], r['funny'], r['cool'], r['fans']))

    return rdd_user


def mapFeaturesTrainingData(sc, input_folder_path, rdd_business, rdd_user):
    # read training data
    path_train =  input_folder_path + 'yelp_train.csv'
    rdd_train_raw = sc.textFile(path_train)
    rdd_header = rdd_train_raw.first()
    rdd_train = rdd_train_raw.filter(lambda item: item != rdd_header). \
        map(lambda r: (r.split(',')[0], r.split(',')[1], float(r.split(',')[2])))

    # join with user-side data
    rdd_train = rdd_train.keyBy(lambda r: r[0]).leftOuterJoin(rdd_user.keyBy(lambda r: r[0])).\
        map(lambda r: (r[0], r[1][0][1], r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5], r[1][1][6], r[1][0][2]))

    # join with business-side data
    rdd_train = rdd_train.keyBy(lambda r: r[1]).leftOuterJoin(rdd_business.keyBy(lambda r: r[0])).\
        map(lambda r: (r[1][0][0], r[0], r[1][0][2], r[1][0][3], r[1][0][4], r[1][0][5], r[1][0][6], r[1][0][7],\
                       r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5], r[1][0][8]))

    return rdd_train


def trainXGBoost(rdd_train):
    # convert to pandas DF
    col_names = ['uid', 'bid', 'user_num_rev', 'user_avg_star', 'user_useful', 'user_funny', 'user_cool', \
                 'user_fans', 'bus_latitude', 'bus_longitude', 'bus_avg_star', 'bus_num_rev', 'bus_num_photo', 'y']
    df_train = pd.DataFrame(rdd_train.collect(), columns=col_names)
    df_train.fillna(0, inplace=True)

    # separate X and y
    X_train = df_train.loc[:, ~df_train.columns.isin(['uid', 'bid', 'y'])]
    y_train = df_train.loc[:, 'y']

    # train using XGBoost
    reg_xgb = xgb.XGBRegressor(random_state=0)
    reg_xgb.fit(X_train, y_train)

    return reg_xgb


def mapFeaturesTestingData(sc, test_data_path, rdd_business, rdd_user):
    # read testing data
    rdd_test_raw = sc.textFile(test_data_path)
    rdd_header = rdd_test_raw.first()
    rdd_test = rdd_test_raw.filter(lambda item: item != rdd_header). \
        map(lambda r: (r.split(',')[0], r.split(',')[1]))

    # join with user-side data
    rdd_test = rdd_test.keyBy(lambda r: r[0]).leftOuterJoin(rdd_user.keyBy(lambda r: r[0])). \
        map(lambda r: (r[0], r[1][0][1], r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5], r[1][1][6]))

    # join with business-side data
    rdd_test = rdd_test.keyBy(lambda r: r[1]).leftOuterJoin(rdd_business.keyBy(lambda r: r[0])). \
        map(lambda r: (r[1][0][0], r[0], r[1][0][2], r[1][0][3], r[1][0][4], r[1][0][5], r[1][0][6], r[1][0][7], \
                       r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5]))

    return rdd_test


def cleanPredictionValue(predicted_rating):
    if predicted_rating > 5:
        return 5
    elif predicted_rating < 1:
        return 1
    else:
        return predicted_rating


def predictXGBoost(reg_xgb, rdd_test):
    # convert to pandas DF
    col_names = ['uid', 'bid', 'user_num_rev', 'user_avg_star', 'user_useful', 'user_funny', 'user_cool', \
                 'user_fans', 'bus_latitude', 'bus_longitude', 'bus_avg_star', 'bus_num_rev', 'bus_num_photo']
    df_test = pd.DataFrame(rdd_test.collect(), columns=col_names)
    df_test.fillna(0, inplace=True)

    # define X_test
    X_test = df_test.loc[:, ~df_test.columns.isin(['uid', 'bid'])]

    # predict using XGBoost
    prediction = reg_xgb.predict(X_test)
    prediction = list(map(cleanPredictionValue, prediction))

    return list(zip(list(df_test.loc[:, 'uid']), list(df_test.loc[:, 'bid']), prediction))


def format_output(list_prediction, header, output_path):
    with open(output_path, 'w+') as o:
        o.write(','.join(str(hd) for hd in header) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in list_prediction))
        o.close()


def calculateRMSE(output_path, actual_path):
    pred = pd.read_csv(output_path)
    actual = pd.read_csv(actual_path)

    merge = actual.merge(pred, how='left', on=['user_id', 'business_id'])
    merge['diff_square'] = (merge.prediction - merge.stars) ** 2

    rmse = (sum(merge.diff_square) / len(merge)) ** 0.5

    return rmse



def convertToDict(x):
    temp = dict((a.strip(), b.strip()) for a, b in (element.split(':') for element in re.sub(r'[{}\'\"]', '', x).split(',')))
    return temp


def flatten_data(y):
    out = {}

    def flatten(x, name=''):
        for k, v in x.items():
            if type(v) is dict:
                flatten(v, name + k + '_')
            elif '{' in str(v) and '}' in str(v):
                try:
                    v = convertToDict(v)
                    flatten(v, name + k + '_')
                except:
                    out[name + k] = v
            else:
                out[name + k] = v

    flatten(y)
    return out


def generateBusinessInfoDataframe(business_info_file_path):
    list_bus_info = list()

    for bus in open(business_info_file_path):
        temp_dict = json.loads(bus)
        temp_dict = flatten_data(temp_dict)
        list_bus_info.append(temp_dict)

    df_business_info = pd.DataFrame(list_bus_info)
    # df_business_info.set_index('business_id', inplace=True)

    # list_col_include = ['business_id', 'stars', 'review_count', 'is_open', 'attributes_BikeParking', 'attributes_BusinessAcceptsCreditCards', 'attributes_BusinessParking_garage', 'attributes_BusinessParking_street', 'attributes_BusinessParking_validated', 'attributes_BusinessParking_lot', 'attributes_BusinessParking_valet', 'attributes_GoodForKids', 'attributes_HasTV', 'attributes_NoiseLevel', 'attributes_OutdoorSeating', 'attributes_RestaurantsDelivery', 'attributes_RestaurantsGoodForGroups', 'attributes_RestaurantsPriceRange2', 'attributes_RestaurantsReservations', 'attributes_RestaurantsTakeOut', 'attributes_Alcohol', 'attributes_Caters', 'attributes_DogsAllowed', 'attributes_DriveThru', 'attributes_GoodForMeal_dessert', 'attributes_GoodForMeal_latenight', 'attributes_GoodForMeal_lunch', 'attributes_GoodForMeal_dinner', 'attributes_GoodForMeal_breakfast', 'attributes_GoodForMeal_brunch', 'attributes_RestaurantsTableService', 'attributes_WheelchairAccessible', 'attributes_WiFi', 'attributes_Ambience_romantic', 'attributes_Ambience_intimate', 'attributes_Ambience_classy', 'attributes_Ambience_hipster', 'attributes_Ambience_touristy', 'attributes_Ambience_trendy', 'attributes_Ambience_upscale', 'attributes_Ambience_casual', 'attributes_Ambience_divey', 'attributes_CoatCheck', 'attributes_Corkage', 'attributes_GoodForDancing', 'attributes_HappyHour', 'attributes_Music_dj', 'attributes_Music_background_music', 'attributes_Music_no_music', 'attributes_Music_karaoke', 'attributes_Music_live', 'attributes_Music_video', 'attributes_Music_jukebox', 'attributes_Smoking', 'attributes_ByAppointmentOnly', 'attributes_AcceptsInsurance', 'attributes_BusinessAcceptsBitcoin', 'attributes_AgesAllowed', 'attributes_RestaurantsCounterService', 'attributes_Open24Hours']
    list_col_include = ['business_id', 'stars', 'latitude', 'longitude', 'review_count', 'is_open']

    df_business_info = df_business_info[[col for col in df_business_info.columns if col in list_col_include]]

    # create dummy variables for business info
    df_business_info_dummy = pd.get_dummies(df_business_info, columns=[col for col in df_business_info.columns if col not in ['business_id', 'is_open', 'stars', 'review_count']])
    df_business_info_dummy.set_index('business_id', inplace=True)

    # normalize col
    df_business_info_norm = (df_business_info_dummy - df_business_info_dummy.mean()) / df_business_info_dummy.std()


    return df_business_info_dummy










if __name__ == '__main__':
    time_start = time.time()

    # set input params

    # input_folder_path = sys.argv[1]
    # test_data_path = sys.argv[2]
    # output_path = sys.argv[3]

    input_folder_path = '../dsci553_hw3/data_input/'
    test_data_path = input_folder_path + 'yelp_val.csv'
    output_path = 'output_model_rf.txt'


    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task2_2').set('spark.executor.memory', '4g').set('spark.driver.memory', '4g')
    sc = SparkContext(conf=spark_config)
    sc.setLogLevel('OFF')


    # read business-side and user-side data into RDD
    # rdd_business = getDataBusiness(sc, input_folder_path)

    df_business = generateBusinessInfoDataframe(input_folder_path + 'business.json')
    df_business.columns = ['bus_' + col for col in df_business.columns]


    path_photo = input_folder_path + 'photo.json'
    rdd_photo = sc.textFile(path_photo). \
        map(lambda r: json.loads(r)). \
        map(lambda r: (r['business_id'], r['photo_id'])).\
        groupByKey().\
        map(lambda r: (r[0], len(r[1])))
    df_photo = pd.DataFrame(rdd_photo.collect(), columns=['business_id', 'num_photo'])
    # print(df_photo)

    rdd_user = getDataUser(sc, input_folder_path)
    df_user = pd.DataFrame(rdd_user.collect(), columns=['user_id', 'review_count', 'average_stars', 'useful', 'funny', 'cool', 'fans'])
    df_user.set_index('user_id', inplace=True)
    df_user.columns = ['user_' + col for col in df_user.columns]

    # print(df_user)
    # df_user = pd.read_json(input_folder_path + 'user.json', lines=True)
    #
    # print(df_user)
    #
    #
    #
    # # get training data + map features on business and user sides
    # rdd_train = mapFeaturesTrainingData(sc, input_folder_path, rdd_business, rdd_user)
    df_train = pd.read_csv(input_folder_path + 'yelp_train.csv')
    # print(df_train)
    df_train = df_train.merge(df_business, on='business_id', how='left'). \
        merge(df_photo, on='business_id', how='left'). \
        merge(df_user, on='user_id', how='left')

    df_train.fillna(0, inplace=True)

    # # separate X and y
    X_train = df_train.loc[:, ~df_train.columns.isin(['user_id', 'business_id', 'user_user_id', 'bus_business_id', 'stars'])]
    y_train = df_train.loc[:, 'stars']

    # test data
    df_test = pd.read_csv(input_folder_path + 'yelp_val.csv')
    # print(df_train)
    df_test = df_test.merge(df_business, on='business_id', how='left'). \
        merge(df_photo, on='business_id', how='left'). \
        merge(df_user, on='user_id', how='left')

    df_test.fillna(0, inplace=True)

    # # separate X and y
    X_test = df_test.loc[:, ~df_test.columns.isin(['user_id', 'business_id', 'user_user_id', 'bus_business_id', 'stars'])]
    y_test = df_test.loc[:, 'stars']

    # print('start RF')
    #
    # list_depth = [2, 3, 5, 7]
    # list_est = [100, 200, 300, 500]
    #
    # for d in list_depth:
    #     for n in list_est:
    #
    #         reg_rf = RandomForestRegressor(random_state=0, max_depth=d, n_estimators=n)
    #         reg_rf.fit(X_train, y_train)
    #
    #         prediction = reg_rf.predict(X_test)
    #         prediction = list(map(cleanPredictionValue, prediction))
    #
    #         temp = df_test.loc[:, ['user_id', 'business_id']]
    #         temp['prediction'] = prediction
    #
    #         temp.to_csv(output_path, index=False)


    # train using XGBoost
    reg_xgb = xgb.XGBRegressor(random_state=0, max_depth=4, n_estimators=300)
    reg_xgb.fit(X_train, y_train)



    # # predict
    prediction = reg_xgb.predict(X_test)
    prediction = list(map(cleanPredictionValue, prediction))

    temp = df_test.loc[:, ~df_test.columns.isin(['user_user_id', 'bus_business_id', 'stars'])]
    temp['prediction'] = prediction

    # temp.to_csv(output_path, index=False)


    print('Duration: ', time.time() - time_start)
    #
            #
            # # calculate RMSE
    actual_path = input_folder_path + 'yelp_val.csv'
    rmse = calculateRMSE(output_path, actual_path)
    print(rmse)
            # print(time.time() - time_start, d, n, rmse)
            # # print('max_depth ', md, ' n_estimators ', n, ' rmse ', rmse)


