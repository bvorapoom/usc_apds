import os
import sys
import time
from pyspark import SparkContext, SparkConf
from blackbox import BlackBox
import random
import binascii
import math
import statistics


# def myhashs(user_id):
#     if 'num_hashfuncs' in globals():
#         global num_hashfuncs
#     else:
#         num_hashfuncs = 15
#
#     if 'num_bits' in globals():
#         global num_bits
#     else:
#         num_bits = 10
#
#     if 'num_buckets' in globals():
#         global num_buckets
#     else:
#         num_buckets = 1000
#
#
#     list_hashfuncs = list()
#
#     list_param_a = random.sample(range(0, 1000000000), num_hashfuncs)
#     list_param_b = random.sample(range(0, 1000000000), num_hashfuncs)
#
#     def generateHashFunc(param_a, param_b, param_m):
#         def apply_funcs(user_id_bin):
#             return format((param_a * user_id_bin + param_b) % param_m, 'b').zfill(num_bits)
#
#         return apply_funcs
#
#
#     for param_a, param_b in zip(list_param_a, list_param_b):
#         list_hashfuncs.append(generateHashFunc(param_a, param_b, num_buckets))
#
#     user_id_bin = int(binascii.hexlify(user_id.encode('utf8')), 16)
#     list_hashvals = [hfunc(user_id_bin) for hfunc in list_hashfuncs]
#
#     return list_hashvals


def myhashs(user_id):
    result = list()

    if 'hash_function_list' in globals():
        global hash_function_list
    else:
        if 'num_hashfuncs' in globals():
            global num_hashfuncs
        else:
            num_hashfuncs = 17

        if 'num_bits' in globals():
            global num_bits
        else:
            num_bits = 69997

        hash_function_list = list()

        list_param_a = random.sample(range(0, 1000000000), num_hashfuncs)
        list_param_b = random.sample(range(0, 1000000000), num_hashfuncs)

        def generateHashFunc(param_a, param_b, param_m):
            def apply_funcs(user_id_bin):
                return (param_a * user_id_bin + param_b) % param_m

            return apply_funcs

        for param_a, param_b in zip(list_param_a, list_param_b):
            hash_function_list.append(generateHashFunc(param_a, param_b, num_bits))

    for f in hash_function_list:
        user_id_bin = int(binascii.hexlify(user_id.encode('utf8')), 16)
        result.append(f(user_id_bin))

    return result


def getLongestTrailingZeros(bit):
    len_bit_strip = len(str(bit).rstrip('0'))
    len_bit = len(bit)
    len_trailing_zero = len_bit - len_bit_strip
    if len_trailing_zero == len_bit:
        return 0
    else:
        return len_trailing_zero


def writeOutput(header, result_list, output_file_path):
    with open(output_file_path, 'w+') as o:
        o.write(','.join(str(hd) for hd in header) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in result_list))
        o.close()


if __name__ == '__main__':
    time_start = time.time()

    input_file_path = 'input/users.txt'
    stream_size = 300
    num_of_asks = 30
    output_file_path = 'output/output_task2.txt'

    # input_file_path = sys.argv[1]
    # stream_size = int(sys.argv[2])
    # num_of_asks = int(sys.argv[3])
    # output_file_path = sys.argv[4]

    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task1').set('spark.executor.memory', '4g').set(
        'spark.driver.memory', '4g')
    sc = SparkContext(conf=spark_config)
    sc.setLogLevel('OFF')

    # set up params and variables
    result_list = list()
    random.seed(553)
    num_hashfuncs = 15
    num_buckets = 1000
    num_bits = len(format(num_buckets, 'b'))
    group_size = 3

    # read data stream + fjm
    bb = BlackBox()
    for tm in range(num_of_asks):
        stream_users = bb.ask(input_file_path, stream_size)
        actual_distinct = len(set(stream_users))

        rdd_user = sc.parallelize(stream_users)

        # hash user_id to binary
        rdd_user = rdd_user.\
            map(lambda u: myhashs(u)).\
            map(lambda list_bit: list(map(getLongestTrailingZeros, list_bit)))

        result = rdd_user.collect()
        transposed_result = list(zip(*result))
        max_trailing_zero_list = [2 ** max(vals) for vals in transposed_result]

        # get average 2**r for each group
        list_avg_2r = list()
        for i in range(math.ceil(num_hashfuncs / group_size)):
            temp_list = max_trailing_zero_list[i*group_size:(i+1)*group_size]
            avg_2r = sum(temp_list) / len(temp_list)
            list_avg_2r.append(avg_2r)

        estimate_distinct = round(statistics.median(list_avg_2r))

        result = (tm, actual_distinct, estimate_distinct)
        result_list.append(result)

    sum_act, sum_est = (0, 0)
    for _, act, est in result_list:
        sum_act += act
        sum_est += est

    print('Ratio: ', sum_est / sum_act)

    header = ['Time', 'Ground Truth', 'Estimation']
    writeOutput(header, result_list, output_file_path)

    print('Duration: ', time.time() - time_start)
