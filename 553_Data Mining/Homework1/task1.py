import sys
from pyspark import SparkContext
import json
import datetime


def count_total_reviews(rdd_raw):
    return rdd_raw.count()


def count_num_reviews_year(rdd_year, target_year):
    return rdd_year.filter(lambda item: item[0] == target_year).count()


def count_distinct(rdd):
    return rdd.map(lambda item: item[0]).distinct().count()


def get_top10_by_review(rdd):
    list_top10 = rdd.\
        reduceByKey(lambda x, y: x + y).\
        takeOrdered(10, lambda item: (-item[1], item[0]))
    return [list(item) for item in list_top10]


if __name__ == '__main__':
    # read command arguments
    review_path = sys.argv[1]
    output_path = sys.argv[2]

    # read review json file into spark RDD
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')
    rdd_raw = sc.textFile(review_path).map(lambda item: json.loads(item))
    rdd_year = rdd_raw.map(lambda item: (datetime.datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S').year, 1))
    rdd_userid = rdd_raw.map(lambda item: (item['user_id'], 1))
    rdd_businessid = rdd_raw.map(lambda item: (item['business_id'], 1))

    # create output dictionary
    output_dict = dict()

    # task 1A: total number of reviews
    num_reviews_total = count_total_reviews(rdd_raw)
    output_dict['n_review'] = num_reviews_total

    # task 1B: number of reviews in 2018
    num_reviews_2018 = count_num_reviews_year(rdd_year, 2018)
    output_dict['n_review_2018'] = num_reviews_2018

    # task 1C: number of distinct users who wrote reviews
    num_distinct_users = count_distinct(rdd_userid)
    output_dict['n_user'] = num_distinct_users

    # task 1D: top 10 users writing highest numbers of reviews & num reviews they wrote
    top_10_users = get_top10_by_review(rdd_userid)
    output_dict['top10_user'] = top_10_users

    # task 1E: number of distinct businesses that have been reviewed
    num_distinct_business = count_distinct(rdd_businessid)
    output_dict['n_business'] = num_distinct_business

    # task 1F: top 10 businesses with highest numbers of reviews & num reviews gotten
    top_10_business = get_top10_by_review(rdd_businessid)
    output_dict['top10_business'] = top_10_business

    # write results to output json file
    with open(output_path, 'w+') as o:
        json.dump(output_dict, o)
        o.close()