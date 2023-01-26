import sys
from pyspark import SparkContext
import random
import time
from itertools import combinations


def genPermutation(num_funcs, list_all_uids):
    size = len(list_all_uids)
    list_param_a = random.sample(range(0, 1000000000), num_funcs)
    list_param_b = random.sample(range(0, 1000000000), num_funcs)

    def hashFunc(list_all_uids, param_a, param_b, size):
        return {uid: (param_a * int(uid) + param_b) % size for uid in list_all_uids}

    list_all_permutation = list()
    for i in range(num_funcs):
        dict_permutation = hashFunc(list_all_uids, list_param_a[i], list_param_b[i], size)
        list_all_permutation.append(dict_permutation)

    return list_all_permutation


def minHashUserBusiness(list_uids, list_all_permutation):
    list_output_minhash = list()

    for dict_permutation in list_all_permutation:
        list_uids_hash = [dict_permutation[uid] for uid in list_uids]
        minhash = min(list_uids_hash)
        list_output_minhash.append(minhash)

    return list_output_minhash


def splitMinHashList(minhash_list, num_bands, num_rows):
    return [tuple(minhash_list[i*num_rows:(i+1)*num_rows]) for i in range(num_bands)]



def checkCandidates(pair_candidate, dict_business_id_group):
    item_a, item_b = pair_candidate[0], pair_candidate[1]
    list_a, list_b = dict_business_id_group[item_a], dict_business_id_group[item_b]
    jaccard_sim = calculateJaccardSimilarity(list_a, list_b)
    return jaccard_sim


def calculateJaccardSimilarity(list_a, list_b):
    return len(set(list_a) & set(list_b)) / len(set(list_a) | set(list_b))


def format_output(list_similar_items, header, output_path):
    with open(output_path, 'w+') as o:
        o.write(','.join(str(hd) for hd in header) + '\n')
        o.write('\n'.join(','.join(str(item) for item in tup) for tup in list_similar_items))
        o.close()


if __name__ == '__main__':

    time_start = time.time()

    # set parameters for similar items algorithm
    sim_thres = 0.5
    num_minhash_func = 150
    num_bands = 50
    num_rows = 3

    # input_path = sys.argv[1]
    # output_path = sys.argv[2]

    input_path = 'data_input/yelp_train.csv'
    output_path = 'data_output/output1.txt'

    # read data into RDD
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')
    rdd_raw = sc.textFile(input_path)

    # remove header
    rdd_header = rdd_raw.first()
    rdd_data = rdd_raw.filter(lambda item: item != rdd_header)
    rdd_data_split = rdd_data.map(lambda item: (item.split(',')[0], item.split(',')[1]))

    # create mapping of user_id to numbers, for ease of hashing
    rdd_distinct_user_id = rdd_data_split.map(lambda item: item[0]).distinct().collect()
    dict_uids_mapping = {uid: new_ind for new_ind, uid in enumerate(sorted(rdd_distinct_user_id))}
    list_all_uids = [ind for uid, ind in dict_uids_mapping.items()]

    # group user_ids that reviewed each business into lists
    rdd_business_id_group = rdd_data_split.map(lambda item: (item[1], dict_uids_mapping[item[0]])).\
        groupByKey().map(lambda item: (item[0], list(set(list(item[1])))))

    # generate permutation to be used in minhash
    list_all_permutation = genPermutation(num_minhash_func, list_all_uids)

    # minhashing step
    rdd_business_minhash = rdd_business_id_group.map(lambda item: (item[0], minHashUserBusiness(item[1], list_all_permutation)))

    # LSH step
    rdd_candidates = rdd_business_minhash.\
        flatMap(lambda item: [(band, item[0]) for band in splitMinHashList(item[1], num_bands, num_rows)]).\
        groupByKey().\
        map(lambda item: list(item[1])).\
        filter(lambda item: len(item) > 1).\
        flatMap(lambda item: [tuple(sorted(pair_candidate)) for pair_candidate in combinations(item, 2)]).\
        distinct()

    # recheck if the candidates gotten from LSH are actually similar
    dict_business_id_group = {bid: list_uids for bid, list_uids in rdd_business_id_group.collect()}
    rdd_similar_items = rdd_candidates.\
        map(lambda item: (item[0], item[1], checkCandidates(item, dict_business_id_group))).\
        filter(lambda item: item[2] >= sim_thres).\
        sortBy(lambda item: (item[0], item[1], item[2]))

    # write output file
    header = ['business_id_1', 'business_id_2', 'similarity']
    format_output(rdd_similar_items.collect(), header, output_path)

    # print(len(rdd_similar_items.collect()))
    print('Duration: ', time.time() - time_start)

