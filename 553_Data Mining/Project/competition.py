"""
Method Description:


Error Distribution:
>=0 and <1:
>=1 and <2:
>=2 and <3:
>=3 and <4:
>=4:

RMSE:


Execution Time:
s
"""

import sys
from pyspark import SparkContext, SparkConf
import time
import xgboost as xgb
import pandas as pd
import json
import re
import statistics as stat
from datetime import datetime as dt


#####################################################################################
################## OTHER FUNCTIONS ##################################################
#####################################################################################


def calculateRMSE(output_path, actual_path):
    pred = pd.read_csv(output_path)
    actual = pd.read_csv(actual_path)

    merge = actual.merge(pred, how='left', on=['user_id', 'business_id'])
    merge['diff_square'] = (merge.prediction - merge.stars) ** 2

    rmse = (sum(merge.diff_square) / len(merge)) ** 0.5
    return rmse


def format_output(list_prediction, header, output_path):
    with open(output_path, 'w+') as o:
        o.write(','.join(str(hd) for hd in header) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in list_prediction))
        o.close()


#####################################################################################
################## CF: ITEM-BASED / USER-BASED ######################################
#####################################################################################


def calculatePearsonCorr(dict_a, dict_b):
    user_a = set(dict_a.keys())
    user_b = set(dict_b.keys())
    co_user = (user_a & user_b)
    len_co_user = len(co_user)

    if len_co_user <= 1:
        corr = 0
    else:
        dict_co_a = {k: v for k, v in dict_a.items() if k in co_user}
        dict_co_b = {k: v for k, v in dict_b.items() if k in co_user}
        rating_co_a = list(dict_co_a.values())
        rating_co_b = list(dict_co_b.values())
        avg_co_a = sum(rating_co_a) / len_co_user
        avg_co_b = sum(rating_co_b) / len_co_user
        numer = 0
        denom_a = 0
        denom_b = 0

        for i in range(len_co_user):
            numer += ((rating_co_a[i] - avg_co_a) * (rating_co_b[i] - avg_co_b))
            denom_a += ((rating_co_a[i] - avg_co_a) ** 2)
            denom_b += ((rating_co_b[i] - avg_co_b) ** 2)

        if denom_a == 0 or denom_b == 0:
            corr = 0
        else:
            corr = numer / ((denom_a ** 0.5) * (denom_b ** 0.5))

    return corr


def predictCFIndividual(how, target_user, target_business, dict_user_group, dict_business_group, nlargest):
    dict_business_this_user = dict_user_group.get(target_user, {})
    dict_user_this_business = dict_business_group.get(target_business, {})

    avg_rating_this_user = sum(dict_business_this_user.values()) / len(dict_business_this_user.values()) if len(
        dict_business_this_user.values()) != 0 else 0
    avg_rating_this_business = sum(dict_user_this_business.values()) / len(dict_user_this_business.values()) if len(
        dict_user_this_business.values()) != 0 else 0

    len_dict_corr = 0

    # if new business & new user -> predict 3
    if len(dict_user_this_business) == 0 and len(dict_business_this_user) == 0:
        predicted_rating = 3.7
    # if new business -> predict average user star
    elif len(dict_user_this_business) == 0:
        predicted_rating = avg_rating_this_user
    # if new user (haven't reviewed anything before) -> predict average business star
    elif len(dict_business_this_user) == 0:
        predicted_rating = avg_rating_this_business
    else:
        if how == 'item-based':
            # get correlation
            dict_corr = dict()
            for co_rated_business in dict_business_this_user.keys():
                dict_user_co_business = dict_business_group.get(co_rated_business, {})
                dict_corr[co_rated_business] = calculatePearsonCorr(dict_user_this_business, dict_user_co_business)
            # filter dict_corr to include only corr > 0
            dict_corr_filtered = {k: v for k, v in dict_corr.items() if v > 0}
            len_dict_corr = len(dict_corr_filtered)
            # get prediction
            if sum(dict_corr_filtered.values()) == 0 or len(dict_corr_filtered) <= 5:
                predicted_rating = (avg_rating_this_user + avg_rating_this_business) / 2
            else:
                dict_corr_nlargest = dict(sorted(dict_corr_filtered.items(), key=lambda item: -item[1])[:nlargest])
                predicted_rating = sum(
                    [corr * dict_business_this_user[bus] for bus, corr in dict_corr_nlargest.items()]) / sum(
                    dict_corr_nlargest.values())
        elif how == 'user-based':
            # get correlation
            dict_corr = dict()
            dict_co_user_avg = dict()
            for co_rated_user in dict_user_this_business.keys():
                dict_business_co_user = dict_user_group.get(co_rated_user, {})
                dict_co_user_avg[co_rated_user] = sum(dict_business_co_user.values()) / len(
                    dict_business_co_user.values()) if len(dict_business_co_user.values()) != 0 else 0
                dict_corr[co_rated_user] = calculatePearsonCorr(dict_business_this_user, dict_business_co_user)
            # filter dict_corr to include only corr > 0
            dict_corr_filtered = {k: v for k, v in dict_corr.items() if v > 0}
            len_dict_corr = len(dict_corr_filtered)
            # get prediction
            if sum(dict_corr_filtered.values()) == 0 or len(dict_corr_filtered) <= 5:
                predicted_rating = (avg_rating_this_user + avg_rating_this_business) / 2
            else:
                dict_corr_nlargest = dict(sorted(dict_corr_filtered.items(), key=lambda item: -item[1])[:nlargest])
                predicted_rating = avg_rating_this_user + sum(
                    [corr * (dict_user_this_business[user] - dict_co_user_avg[user]) for user, corr in
                     dict_corr_nlargest.items()]) / sum(dict_corr_nlargest.values())

    return (predicted_rating, len_dict_corr)


def predictCF(sc, train_data_path, test_data_path, n_largest, how):
    # read training data into RDD
    rdd_train = sc.textFile(train_data_path)
    rdd_header = rdd_train.first()
    rdd_train = rdd_train.filter(lambda item: item != rdd_header)

    # group by business -> will be used to find Pearson correlation
    rdd_business_group = rdd_train. \
        map(lambda item: (item.split(',')[1], (item.split(',')[0], float(item.split(',')[2])))). \
        groupByKey(). \
        map(lambda item: (item[0], dict(item[1])))
    dict_business_group = dict(list(rdd_business_group.collect()))

    # group by user -> will be used for prediction to select business to find pearson corr
    rdd_user_group = rdd_train. \
        map(lambda item: (item.split(',')[0], (item.split(',')[1], float(item.split(',')[2])))). \
        groupByKey(). \
        map(lambda item: (item[0], dict(item[1])))
    dict_user_group = dict(list(rdd_user_group.collect()))

    # read test data
    rdd_test = sc.textFile(test_data_path)
    rdd_header = rdd_test.first()
    rdd_test = rdd_test.filter(lambda item: item != rdd_header)
    rdd_test = rdd_test.map(lambda item: (item.split(',')[0], item.split(',')[1]))

    # predict
    rdd_test_prediction = rdd_test. \
        map(lambda item: (
    (item[0], item[1]), *predictCFIndividual(how, item[0], item[1], dict_user_group, dict_business_group, n_largest)))

    return rdd_test_prediction


#####################################################################################
################## MODEL-BASED ######################################################
#####################################################################################


# def getDataBusiness(sc, input_folder_path):
#     # read business.json to get attributes: latitude, longitude, stars, review_count
#     path_business = input_folder_path + 'business.json'
#     rdd_business = sc.textFile(path_business).\
#         map(lambda r: json.loads(r)).\
#         map(lambda r: (r['business_id'], r['latitude'], r['longitude'], r['stars'], r['review_count']))
#
#     # read photo.json to get attributes: num_photo
#     path_photo = input_folder_path + 'photo.json'
#     rdd_photo = sc.textFile(path_photo). \
#         map(lambda r: json.loads(r)). \
#         map(lambda r: (r['business_id'], r['photo_id'])).\
#         groupByKey().\
#         map(lambda r: (r[0], len(r[1])))
#
#
#     rdd_business_merge = rdd_business.keyBy(lambda r: r[0]).leftOuterJoin(rdd_photo.keyBy(lambda r: r[0])).\
#         map(lambda r: (r[0], r[1][0][1], r[1][0][2], r[1][0][3], r[1][0][4], r[1][1])).\
#         map(lambda r: (r[0], r[1], r[2], r[3], r[4], r[5][1]) if r[5] is not None else (r[0], r[1], r[2], r[3], r[4], 0))
#
#     return rdd_business_merge


def getDataUser(sc, input_folder_path):
    # read user.json to get attributes: review_count, average_stars, useful, funny, cool, fans
    path_user = input_folder_path + 'user.json'
    rdd_user = sc.textFile(path_user). \
        map(lambda r: json.loads(r)). \
        map(
        lambda r: (r['user_id'], r['review_count'], r['average_stars'], r['useful'], r['funny'], r['cool'], r['fans']))

    return rdd_user


# def mapFeaturesTrainingData(sc, input_folder_path, rdd_business, rdd_user):
#     # read training data
#     path_train =  input_folder_path + 'yelp_train.csv'
#     rdd_train_raw = sc.textFile(path_train)
#     rdd_header = rdd_train_raw.first()
#     rdd_train = rdd_train_raw.filter(lambda item: item != rdd_header). \
#         map(lambda r: (r.split(',')[0], r.split(',')[1], float(r.split(',')[2])))
#
#     # join with user-side data
#     rdd_train = rdd_train.keyBy(lambda r: r[0]).leftOuterJoin(rdd_user.keyBy(lambda r: r[0])).\
#         map(lambda r: (r[0], r[1][0][1], r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5], r[1][1][6], r[1][0][2]))
#
#     # join with business-side data
#     rdd_train = rdd_train.keyBy(lambda r: r[1]).leftOuterJoin(rdd_business.keyBy(lambda r: r[0])).\
#         map(lambda r: (r[1][0][0], r[0], r[1][0][2], r[1][0][3], r[1][0][4], r[1][0][5], r[1][0][6], r[1][0][7],\
#                        r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5], r[1][0][8]))
#
#     return rdd_train


# def trainXGBoost(rdd_train):
#     # convert to pandas DF
#     col_names = ['uid', 'bid', 'user_num_rev', 'user_avg_star', 'user_useful', 'user_funny', 'user_cool', \
#                  'user_fans', 'bus_latitude', 'bus_longitude', 'bus_avg_star', 'bus_num_rev', 'bus_num_photo', 'y']
#     df_train = pd.DataFrame(rdd_train.collect(), columns=col_names)
#     df_train.fillna(0, inplace=True)
#
#     # separate X and y
#     X_train = df_train.loc[:, ~df_train.columns.isin(['uid', 'bid', 'y'])]
#     y_train = df_train.loc[:, 'y']
#
#     # train using XGBoost
#     reg_xgb = xgb.XGBRegressor(random_state=0)
#     reg_xgb.fit(X_train, y_train)
#
#     return reg_xgb


# def mapFeaturesTestingData(sc, test_data_path, rdd_business, rdd_user):
#     # read testing data
#     rdd_test_raw = sc.textFile(test_data_path)
#     rdd_header = rdd_test_raw.first()
#     rdd_test = rdd_test_raw.filter(lambda item: item != rdd_header). \
#         map(lambda r: (r.split(',')[0], r.split(',')[1]))
#
#     # join with user-side data
#     rdd_test = rdd_test.keyBy(lambda r: r[0]).leftOuterJoin(rdd_user.keyBy(lambda r: r[0])). \
#         map(lambda r: (r[0], r[1][0][1], r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5], r[1][1][6]))
#
#     # join with business-side data
#     rdd_test = rdd_test.keyBy(lambda r: r[1]).leftOuterJoin(rdd_business.keyBy(lambda r: r[0])). \
#         map(lambda r: (r[1][0][0], r[0], r[1][0][2], r[1][0][3], r[1][0][4], r[1][0][5], r[1][0][6], r[1][0][7], \
#                        r[1][1][1], r[1][1][2], r[1][1][3], r[1][1][4], r[1][1][5]))
#
#     return rdd_test


def cleanPredictionValue(predicted_rating):
    if predicted_rating > 5:
        return 5
    elif predicted_rating < 1:
        return 1
    else:
        return predicted_rating


# def predictXGBoost(reg_xgb, rdd_test):
#     # convert to pandas DF
#     col_names = ['uid', 'bid', 'user_num_rev', 'user_avg_star', 'user_useful', 'user_funny', 'user_cool', \
#                  'user_fans', 'bus_latitude', 'bus_longitude', 'bus_avg_star', 'bus_num_rev', 'bus_num_photo']
#     df_test = pd.DataFrame(rdd_test.collect(), columns=col_names)
#     df_test.fillna(0, inplace=True)
#
#     # define X_test
#     X_test = df_test.loc[:, ~df_test.columns.isin(['uid', 'bid'])]
#
#     # predict using XGBoost
#     prediction = reg_xgb.predict(X_test)
#     prediction = list(map(cleanPredictionValue, prediction))
#
#     return list(zip(list(df_test.loc[:, 'uid']), list(df_test.loc[:, 'bid']), prediction))


def convertToDict(x):
    temp = dict(
        (a.strip(), b.strip()) for a, b in (element.split(':') for element in re.sub(r'[{}\'\"]', '', x).split(',')))
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

    list_col_include = ['business_id', 'stars', 'review_count', 'is_open', 'attributes_BikeParking',
                        'attributes_BusinessAcceptsCreditCards', 'attributes_BusinessParking_garage',
                        'attributes_BusinessParking_street', 'attributes_BusinessParking_validated',
                        'attributes_BusinessParking_lot', 'attributes_BusinessParking_valet', 'attributes_GoodForKids',
                        'attributes_HasTV', 'attributes_NoiseLevel', 'attributes_OutdoorSeating',
                        'attributes_RestaurantsDelivery', 'attributes_RestaurantsGoodForGroups',
                        'attributes_RestaurantsPriceRange2', 'attributes_RestaurantsReservations',
                        'attributes_RestaurantsTakeOut', 'attributes_Alcohol', 'attributes_Caters',
                        'attributes_DogsAllowed', 'attributes_DriveThru', 'attributes_GoodForMeal_dessert',
                        'attributes_GoodForMeal_latenight', 'attributes_GoodForMeal_lunch',
                        'attributes_GoodForMeal_dinner', 'attributes_GoodForMeal_breakfast',
                        'attributes_GoodForMeal_brunch', 'attributes_RestaurantsTableService',
                        'attributes_WheelchairAccessible', 'attributes_WiFi', 'attributes_Ambience_romantic',
                        'attributes_Ambience_intimate', 'attributes_Ambience_classy', 'attributes_Ambience_hipster',
                        'attributes_Ambience_touristy', 'attributes_Ambience_trendy', 'attributes_Ambience_upscale',
                        'attributes_Ambience_casual', 'attributes_Ambience_divey', 'attributes_CoatCheck',
                        'attributes_Corkage', 'attributes_GoodForDancing', 'attributes_HappyHour',
                        'attributes_Music_dj', 'attributes_Music_background_music', 'attributes_Music_no_music',
                        'attributes_Music_karaoke', 'attributes_Music_live', 'attributes_Music_video',
                        'attributes_Music_jukebox', 'attributes_Smoking', 'attributes_ByAppointmentOnly',
                        'attributes_AcceptsInsurance', 'attributes_BusinessAcceptsBitcoin', 'attributes_AgesAllowed',
                        'attributes_RestaurantsCounterService', 'attributes_Open24Hours', 'latitude', 'longitude']

    df_business_info = df_business_info[[col for col in df_business_info.columns if col in list_col_include]]

    # create dummy variables for business info
    df_business_info_dummy = pd.get_dummies(df_business_info, columns=[col for col in df_business_info.columns if
                                                                       col not in ['business_id', 'is_open', 'stars',
                                                                                   'review_count', 'latitude', 'longitude']])
    df_business_info_dummy.set_index('business_id', inplace=True)

    # normalize col
    # df_business_info_norm = (df_business_info_dummy - df_business_info_dummy.mean()) / df_business_info_dummy.std()

    return df_business_info_dummy


def predictModel(sc, train_data_path, test_data_path, input_folder_path):
    # get business data
    df_business = generateBusinessInfoDataframe(input_folder_path + 'business.json')
    df_business.columns = ['bus_' + col for col in df_business.columns]

    # get business's photo data
    path_photo = input_folder_path + 'photo.json'
    rdd_photo = sc.textFile(path_photo). \
        map(lambda r: json.loads(r)). \
        map(lambda r: (r['business_id'], r['photo_id'])). \
        groupByKey(). \
        map(lambda r: (r[0], len(r[1])))
    df_photo = pd.DataFrame(rdd_photo.collect(), columns=['business_id', 'num_photo'])

    # get user data
    rdd_user = getDataUser(sc, input_folder_path)
    df_user = pd.DataFrame(rdd_user.collect(),
                           columns=['user_id', 'review_count', 'average_stars', 'useful', 'funny', 'cool', 'fans'])
    df_user.set_index('user_id', inplace=True)
    df_user.columns = ['user_' + col for col in df_user.columns]

    # prep training data
    df_train = pd.read_csv(train_data_path)
    df_train = df_train.merge(df_business, on='business_id', how='left'). \
        merge(df_photo, on='business_id', how='left'). \
        merge(df_user, on='user_id', how='left')
    df_train.fillna(0, inplace=True)

    X_train = df_train.loc[:,
              ~df_train.columns.isin(['user_id', 'business_id', 'user_user_id', 'bus_business_id', 'stars'])]
    y_train = df_train.loc[:, 'stars']

    # prep testing data
    df_test = pd.read_csv(test_data_path)
    df_test = df_test.merge(df_business, on='business_id', how='left'). \
        merge(df_photo, on='business_id', how='left'). \
        merge(df_user, on='user_id', how='left')
    df_test.fillna(0, inplace=True)

    X_test = df_test.loc[:,
             ~df_test.columns.isin(['user_id', 'business_id', 'user_user_id', 'bus_business_id', 'stars'])]
    y_test = df_test.loc[:, 'stars']

    # train using XGBoost
    reg_xgb = xgb.XGBRegressor(random_state=0, max_depth=4, n_estimators=300)
    reg_xgb.fit(X_train, y_train)

    # predict
    prediction = reg_xgb.predict(X_test)
    prediction = list(map(cleanPredictionValue, prediction))

    result = df_test.loc[:, ~df_train.columns.isin(['user_user_id', 'bus_business_id', 'stars'])]
    result['prediction'] = prediction

    return result

    # result = list(zip(list(df_test.loc[:, ['user_id', 'business_id']].itertuples(index=False, name=None)), prediction, list(X_test.itertuples(index=False, name=None))))
    #
    # return sc.parallelize(result)


#####################################################################################
################## TRAIN HYBRID USING XGBOOST #######################################
#####################################################################################


#####################################################################################
################## MAIN EXECUTION ###################################################
#####################################################################################


if __name__ == '__main__':
    time_start = time.time()

    # input params
    input_folder_path = '../dsci553_hw3/data_input/'
    test_file_path = input_folder_path + 'yelp_val.csv'
    output_file_path = 'output.txt'

    # input_folder_path = sys.argv[1]
    # test_file_path = sys.argv[2]
    # output_file_path = sys.argv[3]

    # print(input_folder_path)
    # print(test_file_path)

    # input_folder_path = input_folder_path.strip('/')

    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task2_2').set('spark.executor.memory', '4g').set(
        'spark.driver.memory', '4g')
    sc = SparkContext('local[*]')
    sc.setLogLevel('OFF')

    # get training and testing data in RDD
    rdd_train_raw = sc.textFile(input_folder_path + 'yelp_train.csv')
    rdd_header = rdd_train_raw.first()
    rdd_train = rdd_train_raw.filter(lambda r: r != rdd_header). \
        map(lambda r: ((r.split(',')[0], r.split(',')[1]), float(r.split(',')[2])))

    rdd_test_raw = sc.textFile(test_file_path)
    rdd_header = rdd_test_raw.first()
    rdd_test = rdd_test_raw.filter(lambda r: r != rdd_header). \
        map(lambda r: ((r.split(',')[0], r.split(',')[1]), float(r.split(',')[2])))

    # get dicts for business & user group
    rdd_business_group = rdd_train. \
        map(lambda r: (r[0][1], (r[0][0], r[1]))). \
        groupByKey(). \
        map(lambda r: (r[0], dict(r[1])))
    dict_business_group = dict(list(rdd_business_group.collect()))

    rdd_user_group = rdd_train. \
        map(lambda r: (r[0][0], (r[0][1], r[1]))). \
        groupByKey(). \
        map(lambda r: (r[0], dict(r[1])))
    dict_user_group = dict(list(rdd_user_group.collect()))

    # get prediction for training data
    # rdd_pred_training = rdd_train.\
    #     map(lambda r: (*r[0], *predictCFIndividual('user-based', r[0][0], r[0][1], dict_user_group, dict_business_group, 30)))
    #
    # print(rdd_pred_training.collect())

    # get prediction for testing data
    train_data_path = input_folder_path + 'yelp_train.csv'
    # test_data_path = input_folder_path + '/yelp_val.csv'
    prediction_test_userbased = rdd_test. \
        map(lambda r: (
    *r[0], *predictCFIndividual('user-based', r[0][0], r[0][1], dict_user_group, dict_business_group, 30)))
    prediction_test_userbased = pd.DataFrame(prediction_test_userbased.collect(),
                                             columns=['user_id', 'business_id', 'prediction_user', 'len_corr_user'])
    prediction_test_itembased = rdd_test. \
        map(lambda r: (
    *r[0], *predictCFIndividual('item-based', r[0][0], r[0][1], dict_user_group, dict_business_group, 30)))
    prediction_test_itembased = pd.DataFrame(prediction_test_itembased.collect(),
                                             columns=['user_id', 'business_id', 'prediction_item', 'len_corr_item'])
    prediction_test_modelbased = predictModel(sc, train_data_path, test_file_path, input_folder_path)
    # print(prediction_test_userbased.take(5))

    merge_test = prediction_test_itembased.merge(prediction_test_userbased, on=['user_id', 'business_id']). \
        merge(prediction_test_modelbased, on=['user_id', 'business_id'])

    merge_test['pred_hybrid'] = (merge_test.prediction_item * merge_test.len_corr_item * 0.0028 + \
                                 merge_test.prediction_user * merge_test.len_corr_user * 0.0059 + \
                                 merge_test.prediction * 0.99129) / (merge_test.len_corr_item * 0.0028 + \
                                                                     merge_test.len_corr_user * 0.0059 + \
                                                                     0.99129)

    result = merge_test[['user_id', 'business_id', 'pred_hybrid']].itertuples(index=False, name=None)

    format_output(result, ['user_id', 'business_id', 'prediction'], output_file_path)

    # calculate RMSE
    actual_path = test_file_path
    rmse = calculateRMSE(output_file_path, actual_path)
    print(rmse)

    # merge_test_X = merge_test.loc[:,
    #           ~merge_test.columns.isin(['prediction_item', 'prediction_user', 'prediction', 'user_id', 'business_id'])]
    # res_rmse_test = merge_test.loc[:, ['user_id', 'business_id', 'prediction_item', 'prediction_user', 'prediction']]
    # actual_test = pd.read_csv('../dsci553_hw3/data_input/yelp_val.csv')
    # res_rmse_test = res_rmse_test.merge(actual_test, on=['user_id', 'business_id'])
    #
    #
    # ################################################################################################################################
    #
    #
    # x = pd.read_csv('output_model_train.txt')
    # # print(x)
    #
    # y = pd.read_csv('output_cf_item_train.txt')
    # # print(y)
    #
    # z = pd.read_csv('output_cf_user_train.txt')
    # # print(z)
    #
    # merge = y.merge(z, on=['user_id', 'business_id'], suffixes=('_item', '_user')).merge(x,
    #                                                                                      on=['user_id', 'business_id'])
    #
    # merge_X = merge.loc[:,
    #           ~merge.columns.isin(['prediction_item', 'prediction_user', 'prediction', 'user_id', 'business_id'])]
    # res_rmse = merge.loc[:, ['user_id', 'business_id', 'prediction_item', 'prediction_user', 'prediction']]
    # actual = pd.read_csv('../dsci553_hw3/data_input/yelp_train.csv')
    # res_rmse = res_rmse.merge(actual, on=['user_id', 'business_id'])
    #
    # res_rmse['diff_square_item'] = (res_rmse.prediction_item - res_rmse.stars) ** 2
    # res_rmse['diff_square_user'] = (res_rmse.prediction_user - res_rmse.stars) ** 2
    # res_rmse['diff_square_model'] = (res_rmse.prediction - res_rmse.stars) ** 2
    #
    # res_rmse['col_min'] = res_rmse[['diff_square_item', 'diff_square_user', 'diff_square_model']].idxmin(axis=1)
    # res_rmse['col_min'] = res_rmse['col_min'].replace(
    #     {'diff_square_item': 0, 'diff_square_user': 1, 'diff_square_model': 2})
    #
    # merge_y = res_rmse.loc[:, 'col_min']
    #
    # # rmse = (sum(merge.diff_square) / len(merge)) ** 0.5
    #
    # ################################################################################################################################
    #
    # print(time.time() - time_start)
    #
    # # list_depth = [2, 3, 5, 10]
    # # list_alp = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #
    # # for d in list_depth:
    # #     for alp in list_alp:
    #
    # tree = DecisionTreeClassifier(random_state=0)
    # tree.fit(merge_X, merge_y)
    #
    # pred_model_selected = tree.predict(merge_test_X)
    # res_rmse_test['model_selected'] = pred_model_selected
    #
    # res_rmse_test['pred_hybrid'] = res_rmse_test.apply(lambda r: r.prediction_item if r.model_selected == 0 else (r.prediction_user if r.model_selected == 1 else r.prediction), axis=1)
    #
    #
    #
    # rmse = (sum((res_rmse_test.pred_hybrid - res_rmse_test.stars) ** 2) / len(res_rmse_test)) ** 0.5
    #
    # # print(d, alp, rmse)
    # print(rmse)

    print('Duration: ', time.time() - time_start)

