import sys
from pyspark import SparkContext
import json
import time


def join_review_business(review_path, business_path):
    # read data from json file
    rdd_review = sc.textFile(review_path).map(lambda item: json.loads(item))
    rdd_business = sc.textFile(business_path).map(lambda item: json.loads(item))
    # select related fields
    rdd_review_stars = rdd_review.map(lambda item: (item['business_id'], item['stars']))
    rdd_business_city = rdd_business.map(lambda item: (item['business_id'], item['city']))
    # use left outer join
    rdd_join = rdd_review_stars.leftOuterJoin(rdd_business_city)
    rdd_join = rdd_join.map(lambda item: (item[0], (item[1][0], 'Undefined' if item[1][1] is None else item[1][1])))
    return rdd_join


if __name__ == '__main__':
    review_path = sys.argv[1]
    business_path = sys.argv[2]
    output_path_1 = sys.argv[3]
    output_path_2 = sys.argv[4]

    sc = SparkContext('local[*]', 'task3')
    sc.setLogLevel('OFF')
    mode_sorting = ['m1', 'm2']

    # task 3A: find average starts from each city
    rdd_join = join_review_business(review_path, business_path)
    avg_star_by_city_sorted = rdd_join. \
        map(lambda item: (item[1][1], (item[1][0], 1))). \
        reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])). \
        map(lambda item: (item[0], item[1][0] / item[1][1])). \
        sortBy(lambda item: (-item[1], item[0])).collect()

    # write results to output json file
    with open(output_path_1, 'w+') as o:
        header_list = ['city', 'stars']
        o.write(','.join(str(hd) for hd in header_list) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in avg_star_by_city_sorted))
        o.close()

    # task 3B: compare execution time of python sort and spark sort
    output_3b = dict()
    for mode in mode_sorting:
        time_start = time.time()

        rdd_join = join_review_business(review_path, business_path)

        if mode == 'm1':
            avg_star_by_city = rdd_join.\
                map(lambda item: (item[1][1], (item[1][0], 1))).\
                reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).\
                map(lambda item: (item[0], item[1][0] / item[1][1])).collect()
            avg_star_by_city_sorted = sorted(avg_star_by_city, key = lambda item: (-item[1], item[0]))
            top_10_cities = [item[0] for item in avg_star_by_city_sorted[:10]]
        else:
            avg_star_by_city_top10 = rdd_join. \
                map(lambda item: (item[1][1], (item[1][0], 1))). \
                reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])). \
                map(lambda item: (item[0], item[1][0] / item[1][1])).\
                takeOrdered(10, lambda item: (-item[1], item[0]))
            top_10_cities = [item[0] for item in avg_star_by_city_top10]

        print(top_10_cities)

        time_finish = time.time()

        output_3b[mode] = time_finish - time_start

    # output_3b['reason'] = "After comparing the two approaches several times, the execution time is not different between sorting in Python and in PySpark. The reason for this observation could be that for PySpark, there are shuffles of data between partitions in order to sort the data since it was not partitioned by the average stars from the beginning, which does not make a difference from sorting all the collected data in Python"
    output_3b['reason'] = 'test'

    # write results to output json file
    with open(output_path_2, 'w+') as o:
        json.dump(output_3b, o)
        o.close()
