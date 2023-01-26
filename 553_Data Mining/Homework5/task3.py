import os
import sys
import time
from blackbox import BlackBox
import random


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
    output_file_path = 'output/output_task3.txt'

    # input_file_path = sys.argv[1]
    # stream_size = int(sys.argv[2])
    # num_of_asks = int(sys.argv[3])
    # output_file_path = sys.argv[4]

    # set spark
    # spark_config = SparkConf().setMaster('local').setAppName('task3').set('spark.executor.memory', '4g').set(
    #     'spark.driver.memory', '4g')
    # sc = SparkContext(conf=spark_config)
    # sc.setLogLevel('OFF')

    # set params and variables
    random.seed(553)
    sample_list = list()
    result_list = list()

    bb = BlackBox()
    # get first batch of stream and save to sample list
    sample_list = bb.ask(input_file_path, stream_size)
    result = (stream_size, sample_list[0], sample_list[20], sample_list[40], sample_list[60], sample_list[80])
    result_list.append(result)
    num_pos = stream_size

    for tm in range(stream_size, stream_size*num_of_asks, stream_size):
        stream_users = bb.ask(input_file_path, stream_size)

        for user in stream_users:
            num_pos += 1
            prob_keep = 100 / num_pos
            keep = random.random() < prob_keep
            if keep:
                pos_replace = random.randint(0, stream_size-1)
                sample_list[pos_replace] = user

        result = (tm, sample_list[0], sample_list[20], sample_list[40], sample_list[60], sample_list[80])
        result_list.append(result)

    print(result_list)

    header = ['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id']
    writeOutput(header, result_list, output_file_path)

    print('Duration: ', time.time() - time_start)

