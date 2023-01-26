import sys
from pyspark import SparkContext
from collections import defaultdict, Counter
from itertools import combinations
import time
import math


def hash_function(itemset, bucket_size = 49, param_a = 47, param_b = 19):
    temp_val = sum(map(int, itemset))
    return (param_a * temp_val + param_b) % bucket_size


def filter_frequent_itemsets_from_count_dict(dict_count, thres):
    return sorted([itemset for itemset, count in dict_count.items() if count >= thres])


def generate_bitmap_for_hash_table(hash_table, thres):
    dict_bitmap = defaultdict(int)
    for k, v in hash_table.items():
        if v >= thres:
            dict_bitmap[k] = 1
    return dict_bitmap


def generate_next_combinations(list_candidates_freq):
    if len(list_candidates_freq) > 0:
        size = len(list_candidates_freq[0])
        list_next_combinations = list()
        list_candidates_freq = sorted(list_candidates_freq)  # make sure everything is sorted

        # 2 nested for-loops to check if the first n-1 items are the same --> potential to be next combination
        for i, candidate1 in enumerate(list_candidates_freq):
            for candidate2 in list_candidates_freq[i + 1:]:
                if candidate1[:-1] == candidate2[:-1]:
                    # if same first n-1 items, combinations will be first itemset plus the last item of the second itemset
                    combination = candidate1 + (candidate2[-1],)
                    # reconfirm combination found if its subsets are subset of list candidates
                    if set(combinations(combination, size)).issubset(list_candidates_freq):
                        list_next_combinations.append(combination)
                else:
                    break
        return list_next_combinations
    else:
        return list()


def find_candidate_itemsets_pcy_phase1(partition, count_all, support_thres):
    partition = list(partition)

    # find support threshold for each subset
    support_thres_subset = math.ceil(int(support_thres) * len(partition) / count_all)

    # initialize values
    k_items = 1
    dict_final_candidate_itemsets = dict()

    ##########
    # pass1: count each item and create hash table for pairs
    ##########
    dict_count_singleton = defaultdict(int)
    dict_hash_table_pairs = defaultdict(int)

    for basket in partition:
        for singleton in basket:
            dict_count_singleton[tuple([singleton])] += 1
        for pair in combinations(basket, k_items + 1):
            hash_val = hash_function(pair)
            dict_hash_table_pairs[hash_val] += 1

    dict_bitmap = generate_bitmap_for_hash_table(dict_hash_table_pairs, support_thres_subset)
    list_candidates_freq = filter_frequent_itemsets_from_count_dict(dict_count_singleton, support_thres_subset)
    dict_final_candidate_itemsets[k_items] = list_candidates_freq

    ##########
    # pass2 onwards
    ##########
    while len(list_candidates_freq) > k_items:
        k_items += 1
        dict_count = defaultdict(int)

        # for 2-itemset, also check with bitmap generated on pass1
        if k_items == 2:
            for basket in partition:
                for itemset in combinations(basket, k_items):
                    if dict_bitmap[hash_function(itemset)]:
                        dict_count[itemset] += 1
        # for itemset size of 3+, just check from list of combinations generated from permutation
        elif k_items > 2:
            for basket in partition:
                for candidate in list_next_candidates:
                    if set(candidate).issubset(set(basket)):
                        dict_count[candidate] += 1

        list_candidates_freq = filter_frequent_itemsets_from_count_dict(dict_count, support_thres_subset)
        list_next_candidates = generate_next_combinations(list_candidates_freq)
        if len(list_candidates_freq) > 0:
            dict_final_candidate_itemsets[k_items] = list_candidates_freq
        else:
            break

    return dict_final_candidate_itemsets.items()


def count_candidate_phase2(partition, list_candidates_flat):
    partition = list(partition)
    count_candidate = defaultdict(int)

    for basket in partition:
        # check each candidate if they are subset of the basket
        for candidate in list_candidates_flat:
            if set(candidate).issubset(basket):
                # if yes, add value count by 1
                count_candidate[candidate] += 1

    return count_candidate.items()


def format_output(list_candidates, header):
    output = str(header) + '\n'
    for cdd in sorted(list_candidates):
        if cdd[0] == 1:
            output += ','.join([str(item).replace(',', '') for item in cdd[1]]) + '\n\n'
        else:
            output += ','.join(map(str, cdd[1])) + '\n\n'
    return output.strip()


def format_output_v2(list_candidates, header):
    temp_dict = defaultdict(list)
    for item in list_candidates:
        temp_dict[len(item[0])].append(item[0])
    sorted_list = [(k, sorted(v)) for k, v in temp_dict.items()]
    return format_output(sorted_list, header)


if __name__ == '__main__':
    time_start = time.time()

    # case_number = int(sys.argv[1])
    # support_thres = float(sys.argv[2])
    # input_file_path = sys.argv[3]
    # output_file_path = sys.argv[4]

    case_number = 2
    support_thres = 9
    input_file_path = 'data/test_task1.csv'
    output_file_path = 'data/output_test_task1_case2.txt'

    # read csv file to RDD
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')
    rdd_raw = sc.textFile(input_file_path)
    rdd_header = rdd_raw.first()
    rdd_data = rdd_raw.filter(lambda item: item != rdd_header)

    # group to make basket
    if case_number == 1:
        rdd_basket = rdd_data.map(lambda item: (item.split(',')[0], item.split(',')[1])).\
            groupByKey().map(lambda item: sorted(list(set(list(item[1])))))
        count_all_baskets = rdd_basket.count()
    elif case_number == 2:
        rdd_basket = rdd_data.map(lambda item: (item.split(',')[1], item.split(',')[0])). \
            groupByKey().map(lambda item: sorted(list(set(list(item[1])))))
        count_all_baskets = rdd_basket.count()

    # SON phase1: get subset's candidates using PCY
    rdd_candidates = rdd_basket.mapPartitions(lambda partition: find_candidate_itemsets_pcy_phase1(partition, count_all_baskets, support_thres)).\
        reduceByKey(lambda x, y: x + y).\
        map(lambda item: (item[0], sorted(list(set(item[1]))))).\
        sortBy(lambda item: item[0]).collect()

    output_candidate = format_output(rdd_candidates, header='Candidates:')
    print(output_candidate)

    k_largest = max([k for k, v in rdd_candidates])
    list_candidates_flat = [candidate for k_items, list_candidate in rdd_candidates for candidate in list_candidate]


    # SON phase2: validate candidates if truly frequent with entire data
    rdd_frequent = rdd_basket.mapPartitions(lambda partition: count_candidate_phase2(partition, list_candidates_flat)).\
        reduceByKey(lambda x, y: x + y).\
        filter(lambda item: item[1] >= support_thres).collect()

    output_frequent_itemset = format_output_v2(rdd_frequent, header='Frequent Itemsets:')
    print(output_frequent_itemset)

    # write results to output file
    with open(output_file_path, 'w+') as o:
        o.write(output_candidate + '\n\n')
        o.write(output_frequent_itemset)
        o.close()

    time_end = time.time()

    print("Duration:", time_end - time_start)








#
# def generate_candidates_from_freq_item_and_bitmap(list_candidates_freq, dict_bitmap, k_items):
#     list_flatten_candidate = sorted(list(set([item for itemset in list_candidates_freq for item in itemset])))
#     list_candidates_temp = [candidate for candidate in combinations(list_flatten_candidate, k_items) if dict_bitmap[hash_function(candidate)]]
#     return list_candidates_temp
#
#
# def generate_next_candidates_from_current_candidate(list_candidates_temp, k_items):
#     list_flatten_candidate = sorted(list(set([item for itemset in list_candidates_temp for item in itemset])))
#     list_candidates_next = [candidate for candidate in combinations(list_flatten_candidate, k_items)]
#     return list_candidates_next


# def generate_next_combinations_slow(list_candidates_freq):
#     size = len(list_candidates_freq[0])
#     unpack = [item for itemset in list_candidates_freq for item in itemset]
#     count_element = Counter(unpack)
#     freq_element = sorted([item for item, cnt in count_element.items() if cnt >= size])
#     if len(freq_element) < size + 1:
#         return list()
#     else:
#         list_combinations = list()
#         for combination in combinations(freq_element, size + 1):
#             if set(combinations(combination, size)).issubset(list_candidates_freq):
#                 list_combinations.append(combination)
#     return list_combinations