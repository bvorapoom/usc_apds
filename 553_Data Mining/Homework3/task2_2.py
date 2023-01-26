import sys
from pyspark import SparkContext, SparkConf
import time
import json
import xgboost as xgb
import pandas as pd

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


if __name__ == '__main__':
    time_start = time.time()

    # set input params

    # input_folder_path = sys.argv[1]
    # test_data_path = sys.argv[2]
    # output_path = sys.argv[3]

    input_folder_path = 'data_input/'
    test_data_path = 'data_input/yelp_val.csv'
    output_path = 'data_output/output2b_new2.txt'


    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task2_2').set('spark.executor.memory', '4g').set('spark.driver.memory', '4g')
    sc = SparkContext(conf=spark_config)
    sc.setLogLevel('OFF')


    # read business-side and user-side data into RDD
    rdd_business = getDataBusiness(sc, input_folder_path)
    rdd_user = getDataUser(sc, input_folder_path)


    # get training data + map features on business and user sides
    rdd_train = mapFeaturesTrainingData(sc, input_folder_path, rdd_business, rdd_user)

    # train XGBoost
    reg_xgb = trainXGBoost(rdd_train)
    sys.exit(0)

    # get testing data + map features
    rdd_test = mapFeaturesTestingData(sc, test_data_path, rdd_business, rdd_user)

    # predict
    pred = predictXGBoost(reg_xgb, rdd_test)
    print(pred)

    # write output
    header = ['user_id', 'business_id', 'prediction']
    format_output(pred, header, output_path)


    print('Duration: ', time.time() - time_start)


    # calculate RMSE
    actual_path = input_folder_path + 'yelp_val.csv'
    rmse = calculateRMSE(output_path, actual_path)
    print(rmse)


