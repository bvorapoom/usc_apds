import os
import sys
import time
from pyspark import SparkContext, SparkConf
from blackbox import BlackBox
import binascii
import math
import random


def myhashs(user_id):
    if 'num_hashfuncs' in globals():
        global num_hashfuncs
    else:
        num_hashfuncs = 17

    if 'num_bits' in globals():
        global num_bits
    else:
        num_bits = 69997

    list_hashfuncs = list()

    list_param_a = random.sample(range(0, 1000000000), num_hashfuncs)
    list_param_b = random.sample(range(0, 1000000000), num_hashfuncs)

    def generateHashFunc(param_a, param_b, param_m):
        def apply_funcs(user_id_bin):
            return (param_a * user_id_bin + param_b) % param_m

        return apply_funcs


    for param_a, param_b in zip(list_param_a, list_param_b):
        list_hashfuncs.append(generateHashFunc(param_a, param_b, num_bits))

    user_id_bin = int(binascii.hexlify(user_id.encode('utf8')), 16)
    list_hashvals = [hfunc(user_id_bin) for hfunc in list_hashfuncs]

    return list_hashvals


def checkExistActual(user_id, set_seen_users):
    if user_id in set_seen_users:
        return 1
    else:
        return 0


def checkExistBloomFilter(bit_array, list_hashvals):
    seen = 1
    for hval in list_hashvals:
        if bit_array[hval] == 0:
            seen = 0
            break
    return seen


def writeOutput(header, result_list, output_file_path):
    with open(output_file_path, 'w+') as o:
        o.write(','.join(str(hd) for hd in header) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in result_list))
        o.close()


if __name__ == '__main__':
    time_start = time.time()

    input_file_path = 'input/users.txt'
    stream_size = 100
    num_of_asks = 30
    output_file_path = 'output/output_task1.txt'

    # input_file_path = sys.argv[1]
    # stream_size = int(sys.argv[2])
    # num_of_asks = int(sys.argv[3])
    # output_file_path = sys.argv[4]

    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task1').set('spark.executor.memory', '4g').set(
        'spark.driver.memory', '4g')
    sc = SparkContext(conf=spark_config)
    sc.setLogLevel('OFF')

    # set up params
    num_bits = 69997
    num_hashfuncs = math.ceil(num_bits / (stream_size * num_of_asks) * math.log(2))
    random.seed(553)

    # set up variables
    bit_array = [0] * num_bits
    result_list = list()
    set_seen_users = set()

    # read data stream + bloom filter
    bb = BlackBox()

    # test = [[], [], []]
    for tm in range(num_of_asks):
        stream_users = bb.ask(input_file_path, stream_size)
        rdd_user = sc.parallelize(stream_users)

        # pass set of hash functions to user_id
        rdd_user = rdd_user.\
            map(lambda u: (u, myhashs(u)))

        # check actual existence / existence using bloom filter
        rdd_user = rdd_user.\
            map(lambda u: (u[0], u[1], checkExistActual(u[0], set_seen_users), checkExistBloomFilter(bit_array, u[1])))

        # calculate FPR
        list_existence = rdd_user.map(lambda u: (u[2], u[3])).collect()
        cnt_negatives = 0
        cnt_fp = 0
        for actual, bloomf in list_existence:
            if actual == 0:
                cnt_negatives += 1
                if bloomf == 1:
                    cnt_fp += 1
        fpr = cnt_fp / cnt_negatives

        result = (tm, fpr)
        result_list.append(result)

        # update bit arrays
        list_current_bit_array = list(set(rdd_user.flatMap(lambda u: u[1]).collect()))
        for bit in list_current_bit_array:
            bit_array[bit] = 1

        # add this batch of user to the seen_user set
        set_seen_users = set_seen_users.union(set(stream_users))

    header = ['Time', 'FPR']
    writeOutput(header, result_list, output_file_path)

    print('Duration: ', time.time() - time_start)

