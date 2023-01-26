import sys
from pyspark import SparkContext
import time


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


def predictRating(target_user, target_business, dict_user_group, dict_business_group, nlargest):
    # if new business -> predict average user star
    if target_business not in dict_business_group.keys():
        dict_rated_business = dict_user_group[target_user]
        predicted_rating = sum(dict_rated_business.values()) / len(dict_rated_business.values())
        return_val = (predicted_rating, 0, 0, 0)
    # if new user (haven't reviewed anything before) -> predict average business star
    elif target_user not in dict_user_group.keys():
        dict_rated_user = dict_business_group.get(target_business, {})
        predicted_rating = sum(dict_rated_user.values()) / len(dict_rated_user.values())
        return_val = (predicted_rating, 0, 0, 0)
    else:
        dict_rated_business = dict_user_group[target_user]
        dict_rated_user = dict_business_group.get(target_business, {})

        # calculate pearson corr for every pair of the target business and other businesses the user has rated
        dict_corr = dict()
        dict_rating_target_bus = dict_business_group.get(target_business, {})
        for bus in dict_rated_business.keys():
            dict_rating_bus = dict_business_group.get(bus, {})
            dict_corr[bus] = calculatePearsonCorr(dict_rating_target_bus, dict_rating_bus)

        dict_corr_filtered = {k: v for k, v in dict_corr.items() if v > 0}
        # dict_corr_filtered = {k: v for k, v in dict_corr.items() if v != 0}

        # if num of co-rated businesses less than 5, predict using average between business rating and user rating
        if sum(dict_corr_filtered.values()) == 0 or len(dict_corr_filtered) <= 5:
            avg_user = sum(dict_rated_business.values()) / len(dict_rated_business.values())
            avg_business = sum(dict_rated_user.values()) / len(dict_rated_user.values())
            # predicted_rating = sum(dict_rated_business.values()) / len(dict_rated_business.values())
            # predicted_rating = sum(dict_rated_user.values()) / len(dict_rated_user.values())
            predicted_rating = (avg_user + avg_business) / 2
        # otherwise, use all co-rated businesses to predict
        # else:
        #     numer = sum([corr * dict_rated_business[bus] if corr > 0 else abs(corr * (dict_rated_business[bus] - 6)) for bus, corr in dict_corr_filtered.items()])
        #     denom = sum(map(abs, dict_corr_filtered.values()))
        #     predicted_rating = numer / denom
        elif len(dict_corr_filtered) <= nlargest:
            predicted_rating = sum([corr * dict_rated_business[bus] for bus, corr in dict_corr_filtered.items()]) / sum(dict_corr_filtered.values())
        else:
            dict_corr_nlargest = dict(sorted(dict_corr_filtered.items(), key=lambda item: -item[1])[:nlargest])
            predicted_rating = sum([corr * dict_rated_business[bus] for bus, corr in dict_corr_nlargest.items()]) / sum(dict_corr_nlargest.values())

        return_val = (predicted_rating, len(dict_corr_filtered), sum(dict_rated_business.values()) / len(dict_rated_business.values()), sum(dict_rated_user.values()) / len(dict_rated_user.values()))

    return return_val


def format_output(list_prediction, header, output_path):
    with open(output_path, 'w+') as o:
        o.write(','.join(str(hd) for hd in header) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in list_prediction))
        o.close()


if __name__ == '__main__':

    time_start = time.time()

    # input_path = sys.argv[1]
    # test_data_path = sys.argv[2]
    # output_path = sys.argv[3]

    input_path = 'data_input/yelp_train.csv'
    test_data_path = 'data_input/yelp_val.csv'
    output_path = 'data_output/output2a.txt'

    n_largest = 30

    # read training data into RDD
    sc = SparkContext('local[*]', 'task2a')
    sc.setLogLevel('OFF')
    rdd_raw = sc.textFile(input_path)

    # remove header
    rdd_header = rdd_raw.first()
    rdd_data = rdd_raw.filter(lambda item: item != rdd_header)

    # group by business -> will be used to find Pearson correlation
    rdd_business_group = rdd_data.\
        map(lambda item: (item.split(',')[1], (item.split(',')[0], float(item.split(',')[2])))). \
        groupByKey(). \
        map(lambda item: (item[0], dict(item[1])))
    dict_business_group = dict(list(rdd_business_group.collect()))

    # group by user -> will be used for prediction to select business to find pearson corr
    rdd_user_group = rdd_data.\
        map(lambda item: (item.split(',')[0], (item.split(',')[1], float(item.split(',')[2])))). \
        groupByKey(). \
        map(lambda item: (item[0], dict(item[1])))
    dict_user_group = dict(list(rdd_user_group.collect()))

    ####
    # ??? might need to average in case user reviewed businesses more than once ???
    ####

    # rdd_distinct_bid = rdd_data_train.map(lambda item: item[0]).distinct().collect()
    # dict_business_mapping = {bid: new_ind for new_ind, bid in enumerate(sorted(rdd_distinct_bid))}



    # read test data
    rdd_val = sc.textFile(test_data_path)

    # remove header
    rdd_header = rdd_val.first()
    rdd_data = rdd_val.filter(lambda item: item != rdd_header)
    rdd_data_val = rdd_data.map(lambda item: (item.split(',')[0], item.split(',')[1]))

    # predict
    rdd_val_prediction = rdd_data_val.\
        map(lambda item: (item[0], item[1], *predictRating(item[0], item[1], dict_user_group, dict_business_group, n_largest)))

    # print(rdd_val_prediction.collect())

    # write output
    header = ['user_id', 'business_id', 'prediction', 'len_co_bus', 'avg_bus', 'avg_user']
    format_output(rdd_val_prediction.collect(), header, output_path)


    # rdd_business_cartesian = rdd_business_group.\
    #     cartesian(rdd_business_group).\
    #     filter(lambda item: item[1][0] > item[0][0])
    #
    # rdd_business_corr = rdd_business_cartesian.\
    #     map(lambda item: ((item[0][0], item[1][0]), calculatePearsonCorr(item[0][1], item[1][1]))).\
    #     filter(lambda item: item[1] > 0)


    # print(rdd_val_prediction.take(50))
    # print(dict_user_group, dict_business_group)

    print('Duration: ', time.time() - time_start)

