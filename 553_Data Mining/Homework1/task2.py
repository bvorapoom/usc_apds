import sys
from pyspark import SparkContext
import json
import time


def get_top10_by_review(rdd):
    list_top10 = rdd.\
        reduceByKey(lambda x, y: x + y).\
        takeOrdered(10, lambda item: (-item[1], item[0]))
    return [list(item) for item in list_top10]


if __name__ == '__main__':
    # read command arguments
    review_path = sys.argv[1]
    output_path = sys.argv[2]
    num_customized_partition = int(sys.argv[3])

    # read data from json file
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('OFF')
    rdd_review = sc.textFile(review_path).map(lambda item: json.loads(item))

    mode_partition = ['default', 'customized']
    output_dict = dict()

    # loop through 2 modes to compare execution time of task 1F
    for mode in mode_partition:
        if mode == 'default':
            rdd_businessid = rdd_review.map(lambda item: (item['business_id'], 1))
        else:
            rdd_businessid = rdd_review.map(lambda item: (item['business_id'], 1)).\
                partitionBy(num_customized_partition, lambda item: ord(item[0]))

        result_dict = dict()

        # get number of partitions
        result_dict['n_partition'] = rdd_businessid.getNumPartitions()

        # get number of items per partition
        result_dict['n_items'] = rdd_businessid.glom().map(len).collect()

        # get time comsumed for executing task 1F
        time_start = time.time()
        top_10_business = get_top10_by_review(rdd_businessid)
        time_finish = time.time()
        result_dict['exe_time'] = time_finish - time_start

        # append result_dict for this mode to output_dict
        output_dict[mode] = result_dict

    # write results to output json file
    with open(output_path, 'w+') as o:
        json.dump(output_dict, o)
        o.close()






