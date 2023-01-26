import numpy as np
from sklearn.cluster import KMeans
import sys
import time
import random
from collections import Counter
import math
from itertools import combinations


def custom_kmeans(data, num_cluster, method):
    assert method in ['init_getRS', 'init_group', 'getCS']

    if method == 'init_getRS':
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(data)
        labels = list(kmeans.labels_)
        counter = Counter(labels)
        list_cid_RS = [cid for cid, cnt in counter.items() if cnt == 1] # get cluster number that's RS
        list_ind_RS = [ind for ind, cid in enumerate(labels) if cid in list_cid_RS] # get index of data that's RS
        return list_ind_RS
    elif method == 'init_group':
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(data)
        labels = list(kmeans.labels_)
        dict_cluster = dict()
        for ind, cid in enumerate(labels):
            if cid in dict_cluster.keys():
                dict_cluster[cid] = np.append(dict_cluster[cid], [data[ind]], axis=0)
            else:
                dict_cluster[cid] = np.array([data[ind]])
        dict_DS = {cid: {'n': len(data), 'sum': np.sum(data, axis=0), 'sumsq': np.sum(data**2, axis=0)} \
                   for cid, data in dict_cluster.items()}
        return dict_DS, labels
    elif method == 'getCS':
        if len(data) < num_cluster:
            return {}, [-1]*len(data)
        else:
            kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(data)
            labels = list(kmeans.labels_)
            counter = Counter(labels)
            list_cid_RS = [cid for cid, cnt in counter.items() if cnt == 1]  # get cluster number that's RS
            # list_ind_RS = [ind for ind, cid in enumerate(labels) if cid in list_cid_RS]  # get index of data that's RS

            labels_CS = [lab for lab in labels if lab not in list_cid_RS]

            dict_cluster = dict()
            for ind, cid in enumerate(labels_CS):
                if cid in dict_cluster.keys():
                    dict_cluster[cid] = np.append(dict_cluster[cid], [data[ind]], axis=0)
                else:
                    dict_cluster[cid] = np.array([data[ind]])
            dict_CS = {cid: {'n': len(data), 'sum': np.sum(data, axis=0), 'sumsq': np.sum(data ** 2, axis=0)} \
                       for cid, data in dict_cluster.items()}

            labels = [-1 if lab in list_cid_RS else lab for lab in labels]
            return dict_CS, labels


def getMahalanobisDist(dict_cluster, data):
    dict_cluster_sum = {cid: {'centroid': val['sum']/val['n'], 'sd': (val['sumsq']/val['n'] - (val['sum']/val['n'])**2)**0.5} \
                        for cid, val in dict_cluster.items()}

    temp_result = list()
    for cid, summary in dict_cluster_sum.items():
        centroid = summary['centroid']
        sd = summary['sd']
        maha_dist = np.sum(((data - centroid) / sd) ** 2, axis=1) ** 0.5
        temp_result.append(maha_dist.reshape(len(maha_dist), 1))

    result_dist = np.concatenate(temp_result, axis=1)
    list_cids = list(dict_cluster_sum.keys())
    return result_dist, list_cids


def updateClusterDS(dict_cluster, dict_data_to_merge):
    for cid, data in dict_data_to_merge.items():
        dict_cluster[cid]['n'] = dict_cluster[cid]['n'] + len(data)
        dict_cluster[cid]['sum'] = dict_cluster[cid]['sum'] + np.sum(data, axis=0)
        dict_cluster[cid]['sumsq'] = dict_cluster[cid]['sumsq'] + np.sum(data ** 2, axis=0)
    return dict_cluster


def updateClusterCS(dict_cluster, dict_data_to_merge):
    for cid, data in dict_data_to_merge.items():
        dict_cluster[cid]['n'] = dict_cluster[cid]['n'] + len(data['data'])
        dict_cluster[cid]['sum'] = dict_cluster[cid]['sum'] + np.sum(data['data'], axis=0)
        dict_cluster[cid]['sumsq'] = dict_cluster[cid]['sumsq'] + np.sum(data['data'] ** 2, axis=0)
        dict_cluster[cid]['true_ind'].extend(data['true_ind'])
    return dict_cluster


def assignPointsToClusters(data, dict_bfr, cluster_name, n_dim, merge_scope_sd, true_ind):
    if cluster_name == 'ds':
        # get mahalanobis distance
        result_dist, list_cids = getMahalanobisDist(dict_bfr['ds'], data)

        # set up vars
        dict_ind_merge = dict()
        list_ind_not_merge = list()
        dict_data_to_merge = dict()

        # loop through data and decide to merge or not
        for ind, dist in enumerate(result_dist):
            min_dist = min(dist)
            if min_dist < merge_scope_sd * n_dim**0.5:
                cid_min_dist = list_cids[np.argmin(dist)]
                dict_ind_merge[true_ind[ind]] = cid_min_dist
                if cid_min_dist in dict_data_to_merge.keys():
                    dict_data_to_merge[cid_min_dist] = np.append(dict_data_to_merge[cid_min_dist], [data[ind]], axis=0)
                else:
                    dict_data_to_merge[cid_min_dist] = np.array([data[ind]])
            else:
                list_ind_not_merge.append(ind)

        # update dict_bfr
        dict_bfr['ds'] = updateClusterDS(dict_bfr['ds'], dict_data_to_merge)

        return dict_bfr, dict_ind_merge, list_ind_not_merge

    elif cluster_name == 'cs':
        if len(dict_bfr['cs']) > 0:
            # get mahalanobis distance
            result_dist, list_cids = getMahalanobisDist(dict_bfr['cs'], data)

            # set up vars
            # dict_ind_merge = dict()
            list_ind_not_merge = list()
            dict_data_to_merge = dict()

            # loop through data and decide to merge or not
            for ind, dist in enumerate(result_dist):
                min_dist = min(dist)
                if min_dist < merge_scope_sd * n_dim ** 0.5:
                    cid_min_dist = list_cids[np.argmin(dist)]
                    # dict_ind_merge[true_ind[ind]] = cid_min_dist
                    if cid_min_dist in dict_data_to_merge.keys():
                        dict_data_to_merge[cid_min_dist]['data'] = np.append(dict_data_to_merge[cid_min_dist]['data'], [data[ind]], axis=0)
                        dict_data_to_merge[cid_min_dist]['true_ind'].append(true_ind[ind])
                    else:
                        dict_data_to_merge[cid_min_dist] = {}
                        dict_data_to_merge[cid_min_dist]['data'] = np.array([data[ind]])
                        dict_data_to_merge[cid_min_dist]['true_ind'] = [true_ind[ind]]
                else:
                    list_ind_not_merge.append(ind)

            # update dict_bfr
            dict_bfr['cs'] = updateClusterCS(dict_bfr['cs'], dict_data_to_merge)

            return dict_bfr, list_ind_not_merge
        else:
            return dict_bfr, list(range(len(data)))



def getMahalanobisDist_single(data1, data2):
    centroid = data1['sum'] / data1['n']
    sd = (data1['sumsq'] / data1['n'] - (data1['sum'] / data1['n']) ** 2) ** 0.5
    maha_dist = np.sum((((data2['sum'] / data2['n']) - centroid) / sd) ** 2) ** 0.5
    return maha_dist, sd


def mergeCS(dict_cluster, n_dim, merge_scope_sd):
    # print(dict_cluster)
    if len(dict_cluster) > 1:
        list_merge_cid = list()
        list_pair_merge_cid = list()
        # check if maha dist between CS is less than 2*sqrt(d)
        for cid1, data1 in dict_cluster.items():
            for cid2, data2 in dict_cluster.items():
                if cid1 != cid2 and cid1 not in list_merge_cid and cid2 not in list_merge_cid:
                    maha_dist, sd = getMahalanobisDist_single(data1, data2)
                    if maha_dist < merge_scope_sd * n_dim**0.5:
                        list_merge_cid.extend([cid1, cid2])
                        pair = (cid1, cid2)
                        list_pair_merge_cid.append(pair)

        # merge 2 CS Clusters
        max_cs_cid = max(dict_cluster.keys()) + 1
        for ind, (c1, c2) in enumerate(list_pair_merge_cid):
            dict_cluster[max_cs_cid+ind] = {'n': dict_cluster[c1]['n'] + dict_cluster[c2]['n'], \
                                            'sum': dict_cluster[c1]['sum'] + dict_cluster[c2]['sum'], \
                                            'sumsq': dict_cluster[c1]['sumsq'] + dict_cluster[c2]['sumsq'], \
                                            'true_ind': dict_cluster[c1]['true_ind'] + dict_cluster[c2]['true_ind']}

            dict_cluster.pop(c1)
            dict_cluster.pop(c2)
        return dict_cluster
    else:
        return dict_cluster


def getIntermediateStatus(dict_bfr):
    # get num_DS
    num_DS = 0
    for k, val in dict_bfr['ds'].items():
        num_DS += val['n']

    # get num_cluster_CS
    num_cluster_CS = len(dict_bfr['cs'])
    num_CS = 0
    for k, val in dict_bfr['cs'].items():
        num_CS += val['n']

    # get num_RS
    num_RS = len(dict_bfr['rs'])

    return (num_DS, num_cluster_CS, num_CS, num_RS)


def find_actual_cid(dict_data, dict_data_group, n_dim):
    # list_cid_map = list()
    list_cid_map = [str(dict_data[ind]['actual_cluster_no']) + '_' + str(bfr_cid) for ind, bfr_cid in dict_data_group.items()]

    # for ind, bfr_cid in dict_data_group.items():
    #     list_cid_map.append(str(dict_data[ind]['actual_cluster_no']) + '_' + str(bfr_cid))
        # if np_cid_map is None:
        #     np_cid_map = np.array([str(dict_data[ind]['actual_cluster_no']) + '_' + str(bfr_cid)])
        # else:
        #     np_cid_map = np.append(np_cid_map, [str(dict_data[ind]['actual_cluster_no']) + '_' + str(bfr_cid)])

    np_cid_map = np.array(list_cid_map)
    np_cid_map = np_cid_map[np_cid_map != '-1_-1']
    list_merge_cid, cnt = np.unique(np_cid_map, return_counts=True)
    cnt_merge_cid = list(zip(list(list_merge_cid), list(cnt)))

    selection_merge_cid = sorted(cnt_merge_cid, key=lambda x: -x[1])[:n_dim]

    dict_cid_map = dict()
    for merge_cid, _ in selection_merge_cid:
        actual_cid = int(merge_cid.split('_')[0])
        bfr_cid = int(merge_cid.split('_')[1])
        dict_cid_map[bfr_cid] = actual_cid

    dict_data_group_new = {ind: (dict_cid_map[bfr_cid] if bfr_cid in dict_cid_map.keys() else bfr_cid) for ind, bfr_cid in dict_data_group.items()}
    return dict_data_group_new


def getAccuracy(dict_data, dict_data_group):
    num_total = 0
    num_match = 0
    for ind, data in dict_data.items():
        if data['actual_cluster_no'] != 1:
            num_total += 1
            if data['actual_cluster_no'] == dict_data_group[ind]:
                num_match += 1
    return num_match/num_total


def writeOutput(output_file_path, dict_data_group, result_intermediate):
    with open(output_file_path, 'w+') as o:
        o.write('The intermediate results:\n')
        o.write('\n'.join('Round ' + str(round) + ': ' + ','.join(str(item) for item in out) for round, out in result_intermediate.items()))
        o.write('\n\n' + 'The clustering results:' + '\n')
        o.write('\n'.join(str(ind) + ',' + str(cid) for ind, cid in dict(sorted(dict_data_group.items())).items()))
        o.close()


if __name__ == '__main__':
    time_start = time.time()

    # set up input params
    input_file_path = 'hw6_clustering.txt'
    n_cluster = 10
    output_file_path = 'output.txt'

    # input_file_path = sys.argv[1]
    # n_cluster = int(sys.argv[2])
    # output_file_path = sys.argv[3]

    # set up BFR params
    num_load = 5
    multiplier = 5
    merge_scope_sd = 2

    # variables to track bfr results
    dict_data = dict()
    dict_bfr = {'ds': {}, 'cs': {}, 'rs': []}
    # dict_bfr = dict()
    dict_data_group = dict()
    result_intermediate = dict()

    # read data
    for line in open(input_file_path):
        data = line.split(',')
        index, cluster_no, coord = int(data[0]), int(data[1]), list(map(float, data[2:]))
        dict_data[index] = {'actual_cluster_no': cluster_no, 'coord': coord}

    n_dim = len(dict_data[list(dict_data.keys())[0]]['coord'])

    acc = 0
    pct_discard = 0
    acc_thres = 0.985
    pct_discard_thres = 0.99
    # num_iter = 0

    while not(acc >= acc_thres and pct_discard >= pct_discard_thres) and time.time() - time_start < 500:
        # num_iter += 1
        dict_bfr = {'ds': {}, 'cs': {}, 'rs': []}
        # dict_bfr = dict()
        dict_data_group = dict()
        result_intermediate = dict()

        # split data into 5 chunks
        # random.seed(553)
        list_ind_shuffle = list(dict_data.keys())
        random.shuffle(list_ind_shuffle)
        chunk_size = math.ceil(len(list_ind_shuffle) / num_load)
        list_ind_shuffle = [list_ind_shuffle[i*chunk_size:(i+1)*chunk_size] for i in range(num_load)]

        # read data chunk by chunk
        for chunk_no in range(num_load):
            if chunk_no == 0:
                # get data for this chunk
                true_ind_temp_data = list_ind_shuffle[chunk_no]
                temp_data = np.array([dict_data[ind]['coord'] for ind in true_ind_temp_data])

                # 1st kmeans to separate RS
                list_ind_RS = custom_kmeans(temp_data, n_cluster*multiplier, 'init_getRS')
                true_ind_RS = [true_ind_temp_data[ind] for ind in list_ind_RS]

                # 2nd kmeans to cluster the rest
                temp_data_DS = np.delete(temp_data, list_ind_RS, 0)
                true_ind_DS = [ind for ind in true_ind_temp_data if ind not in true_ind_RS]

                dict_DS, labels_DS = custom_kmeans(temp_data_DS, n_cluster, 'init_group')
                dict_bfr['ds'] = dict_DS # add DS params to dict
                # print(dict_bfr)

                # add cluster number to dict
                for ind, cid in enumerate(labels_DS):
                    dict_data_group[true_ind_DS[ind]] = cid
                # print(dict_data_group)

                # 3rd kmeans to group RS into CS
                temp_data_RS = temp_data[list_ind_RS]
                dict_CS, labels = custom_kmeans(temp_data_RS, n_cluster*multiplier, 'getCS')

                dict_bfr['cs'] = dict_CS
                for lab, true_ind in list(zip(labels, true_ind_RS)):
                    if lab == -1:
                        dict_bfr['rs'].append(true_ind)
                    else:
                        if 'true_ind' in dict_bfr['cs'][lab].keys():
                            dict_bfr['cs'][lab]['true_ind'].append(true_ind)
                        else:
                            dict_bfr['cs'][lab]['true_ind'] = [true_ind]

                # get intermediate status
                result_intermediate[chunk_no+1] = getIntermediateStatus(dict_bfr)
                # print(result_intermediate)

            else:
                # get data for this chunk
                true_ind_temp_data = list_ind_shuffle[chunk_no]
                temp_data = np.array([dict_data[ind]['coord'] for ind in true_ind_temp_data])

                # check with DS and merge
                dict_bfr, dict_ind_merge, list_ind_not_merge = assignPointsToClusters(temp_data, dict_bfr, 'ds', n_dim, merge_scope_sd, true_ind_temp_data)
                dict_data_group.update(dict_ind_merge) # update cluster number for data assigned to DS

                # check with CS and merge
                true_ind_not_merge_DS = [true_ind_temp_data[ind] for ind in list_ind_not_merge]
                temp_data_not_merge_DS = temp_data[list_ind_not_merge]

                dict_bfr, list_ind_not_merge = assignPointsToClusters(temp_data_not_merge_DS, dict_bfr, 'cs', n_dim, merge_scope_sd, true_ind_not_merge_DS)
                # print(dict_bfr)

                # if not assigned to DS/CS, assign to RS
                true_ind_RS = [true_ind_not_merge_DS[ind] for ind in list_ind_not_merge]
                dict_bfr['rs'].extend(true_ind_RS)
                temp_data_RS = np.array([dict_data[ind]['coord'] for ind in dict_bfr['rs']])

                # kmeans on RS
                dict_CS, labels = custom_kmeans(temp_data_RS, n_cluster * multiplier, 'getCS')
                # print('CS', len(dict_CS))
                # print(dict_CS, labels)

                # update cs in dict_bfr
                if len(dict_bfr['cs']) > 0:
                    max_cs_label = max(dict_bfr['cs'].keys()) + 1
                    dict_CS = {label + max_cs_label: val for label, val in dict_CS.items()}
                    dict_bfr['cs'].update(dict_CS)
                else:
                    dict_bfr['cs'] = dict_CS
                    max_cs_label = 0

                # print([(k, v['n']) for k, v in dict_bfr['cs'].items()])
                # print(sorted([k for k, v in dict_bfr['cs'].items()]))
                # print(sorted(list(set(labels))))
                # print(len(dict_bfr['rs']))
                # print(len(set(dict_bfr['rs'])))

                for lab, true_ind in list(zip(labels, dict_bfr['rs'])):
                    if lab != -1:
                        dict_bfr['rs'].remove(true_ind)
                        if 'true_ind' in dict_bfr['cs'][lab+max_cs_label].keys():
                            dict_bfr['cs'][lab+max_cs_label]['true_ind'].append(true_ind)
                        else:
                            dict_bfr['cs'][lab+max_cs_label]['true_ind'] = [true_ind]
                # print(dict_bfr['cs'])
                # print([(k, v['n'], len(v['true_ind'])) for k, v in dict_bfr['cs'].items()])

                # merge CS
                dict_bfr['cs'] = mergeCS(dict_bfr['cs'], n_dim, merge_scope_sd)

                # get intermediate status
                result_intermediate[chunk_no + 1] = getIntermediateStatus(dict_bfr)

                # if last run, merge CS to DS
                if chunk_no == num_load - 1:
                    data_CS = list()
                    if len(dict_bfr['cs']) > 0:
                        for id, data in dict_bfr['cs'].items():
                            data_CS.append(list(data['sum']/data['n']))
                        result_dist, list_cids = getMahalanobisDist(dict_bfr['ds'], data_CS)

                        # merge each CS to DS with min dist
                        for ind, dist in enumerate(result_dist):
                            cid_DS = list_cids[np.argmin(dist)]
                            cid_CS = list(dict_bfr['cs'].keys())[ind]

                            # update dict_bfr ds
                            dict_bfr['ds'][cid_DS]['n'] = dict_bfr['ds'][cid_DS]['n'] + dict_bfr['cs'][cid_CS]['n']
                            dict_bfr['ds'][cid_DS]['sum'] = dict_bfr['ds'][cid_DS]['sum'] + dict_bfr['cs'][cid_CS]['sum']
                            dict_bfr['ds'][cid_DS]['sumsq'] = dict_bfr['ds'][cid_DS]['sumsq'] + dict_bfr['cs'][cid_CS]['sumsq']

                            # update dict_data_group
                            dict_ind_merge_CS = {true_ind: cid_DS for true_ind in dict_bfr['cs'][cid_CS]['true_ind']}
                            dict_data_group.update(dict_ind_merge_CS)

                        # update dict_bfr cs
                        dict_bfr['cs'] = {}

                    # update rs to dict_data_group
                    for ind_RS in dict_bfr['rs']:
                        dict_data_group[ind_RS] = -1



            # print(chunk_no)

        dict_data_group = find_actual_cid(dict_data, dict_data_group, n_dim)
        acc = getAccuracy(dict_data, dict_data_group)
        pct_discard = result_intermediate[5][0]/(result_intermediate[5][0]+result_intermediate[5][3])
        print('accuracy: ', acc, ' /%discard point: ', pct_discard)

    writeOutput(output_file_path, dict_data_group, result_intermediate)

    print('Duration: ', time.time() - time_start)
