import os
import sys
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from graphframes import *
from itertools import combinations

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

def getEdges(rdd_graph_group, filter_thres):
    dict_graph = dict(rdd_graph_group.collect())
    list_nodes = sorted(dict_graph.keys())
    list_edges = list()
    set_vertex = set()
    for comb in combinations(list_nodes, 2):
        set_bus_a = dict_graph[comb[0]]
        set_bus_b = dict_graph[comb[1]]
        num_co_bus = len(set_bus_a & set_bus_b)
        if num_co_bus >= filter_thres:
            list_edges.append(comb)
            reverse_comb = (comb[1], comb[0])
            list_edges.append(reverse_comb)
            set_vertex.add(comb[0])
            set_vertex.add(comb[1])

    return list(set_vertex), list_edges


def writeOutput(list_comm, output_file_path):
    with open(output_file_path, 'w+') as o:
        o.write('\n'.join([str(tup).strip('[').strip(']') for tup in list_comm]))
        o.close()


if __name__ == '__main__':
    time_start = time.time()

    # input params
    filter_thres = float(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # filter_thres = 7
    # input_file_path = 'input/ub_sample_data.csv'
    # output_file_path = 'output/output1.txt'

    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task1').set('spark.executor.memory', '4g').set(
        'spark.driver.memory', '4g')
    sc = SparkContext(conf=spark_config)
    sc.setLogLevel('OFF')
    ss = SparkSession(sc)

    # read data into RDD
    rdd_raw = sc.textFile(input_file_path)
    rdd_header = rdd_raw.first()
    rdd_graph = rdd_raw.filter(lambda item: item != rdd_header)

    # group by userid
    rdd_graph_group = rdd_graph. \
        map(lambda r: (r.split(',')[0], r.split(',')[1])). \
        groupByKey(). \
        map(lambda item: (item[0], set(item[1])))

    # get list of nodes and edges
    list_nodes, list_edges = getEdges(rdd_graph_group, filter_thres)

    # convert list of nodes and edges to Spark DF
    list_nodes_tup = [(node,) for node in list_nodes]
    df_nodes = sc.parallelize(list_nodes_tup).toDF(['id'])
    df_edges = sc.parallelize(list_edges).toDF(['src', 'dst'])

    # find communities using GraphFrames
    graph = GraphFrame(df_nodes, df_edges)
    result_comm = graph.labelPropagation(maxIter=5)

    # read communities to rdd -> group
    rdd_comm = result_comm.rdd.\
        map(lambda r: (r[1], r[0])).\
        groupByKey().\
        map(lambda r: sorted(list(r[1]))).\
        sortBy(lambda r: (len(r), r))

    # write results
    writeOutput(rdd_comm.collect(), output_file_path)

    print('Duration: ', time.time() - time_start)

