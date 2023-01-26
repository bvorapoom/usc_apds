import os
import sys
import time
from pyspark import SparkContext, SparkConf
from itertools import combinations


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
            set_vertex.add(comb[0])
            set_vertex.add(comb[1])

    return list(set_vertex), list_edges


class graphGN(object):
    def __init__(self, nodes, edges, neighbors):
        self.initial_nodes = nodes
        self.initial_edges = edges
        self.initial_neighbors = neighbors

        # self.current_nodes = nodes
        self.current_edges = self.initial_edges.copy()
        self.current_neighbors = self.initial_neighbors.copy()


    def computeBetweenness(self):
        dict_total_betweenness = dict()
        for node in self.initial_nodes:
            tree = self._buildTree(node)
            dict_temp_betweenness = self._computeEdgeCredits(tree)
            for edge, btwn in dict_temp_betweenness.items():
                dict_total_betweenness[edge] = dict_total_betweenness.get(edge, 0) + btwn/2

        return dict_total_betweenness


    def getBestCommunities(self):
        # get current graph's betweenness
        self.dict_btwn_current = self.computeBetweenness()
        # first cut
        self._cutEdgeHighestBtwn(self.dict_btwn_current)
        communities, modularity = self._computeModularity()
        best_communities, max_modularity = communities, modularity
        # cut entire graph
        while len(self.current_edges) > 0:
            self.dict_btwn_current = self.computeBetweenness()
            self._cutEdgeHighestBtwn(self.dict_btwn_current)
            communities, modularity = self._computeModularity()
            if modularity > max_modularity:
                max_modularity = modularity
                best_communities = communities


        return best_communities, max_modularity


    def _buildTree(self, root_node):
        dict_tree = dict()
        dict_tree[root_node] = (0, list())

        list_node_to_check = list()
        list_node_checked = list()
        list_node_to_check.append(root_node)

        while len(list_node_to_check) > 0:
            parent_node = list_node_to_check.pop(0)
            list_node_checked.append(parent_node)
            for child_node in self.current_neighbors[parent_node]:
                if child_node not in list_node_checked:
                    dict_tree[child_node] = (dict_tree[parent_node][0] + 1, [parent_node])
                    list_node_checked.append(child_node)
                    list_node_to_check.append(child_node)
                elif dict_tree[child_node][0] == dict_tree[parent_node][0] + 1:
                    dict_tree[child_node][1].append(parent_node)

        return dict_tree


    def _getNumShortestPath(self, dict_tree):
        list_tree_formatted = sorted([(elem[0], node, elem[1]) for node, elem in dict_tree.items()])
        dict_shortest_path = dict()
        for level, current_node, list_parent_node in list_tree_formatted:
            if len(list_parent_node) > 0:
                sum_shortest_path = 0
                temp_dict = dict()
                for parent_node in list_parent_node:
                    temp_sum = dict_shortest_path[parent_node][0]
                    sum_shortest_path += temp_sum
                    temp_dict[parent_node] = temp_sum
                dict_shortest_path[current_node] = (sum_shortest_path, temp_dict)
            else:
                dict_shortest_path[current_node] = (1, {})
        return dict_shortest_path


    def _computeEdgeCredits(self, dict_tree):
        dict_node_credits = {node: 1 for node, _ in dict_tree.items()}
        dict_edge_credits = dict()
        list_tree_formatted = sorted([(elem[0], node, elem[1]) for node, elem in dict_tree.items()], reverse=True)
        dict_num_shortest_path = self._getNumShortestPath(dict_tree)
        for level, current_node, list_parent_node in list_tree_formatted:
            num_parents = len(list_parent_node)
            if num_parents > 0:
                temp_dict_shortest_path = dict_num_shortest_path[current_node]
                total_shortest_path = temp_dict_shortest_path[0]
                for parent_node in list_parent_node:
                    edge = (current_node, parent_node)
                    edge = tuple(sorted(edge))
                    credit = dict_node_credits[current_node] / total_shortest_path * temp_dict_shortest_path[1][parent_node]
                    dict_node_credits[parent_node] += credit
                    dict_edge_credits[edge] = credit

        return dict_edge_credits


    def _cutEdgeHighestBtwn(self, dict_btwn):
        max_btwn = max(dict_btwn.values())
        list_edge_to_cut = [edge for edge, btwn in dict_btwn.items() if btwn == max_btwn]

        # loop through edge to cut
        for edge_to_cut in list_edge_to_cut:
            self.current_edges.remove(edge_to_cut)
            try:
                self.current_neighbors[edge_to_cut[0]].remove(edge_to_cut[1])
            except:
                pass
            try:
                self.current_neighbors[edge_to_cut[1]].remove(edge_to_cut[0])
            except:
                pass

    def _computeModularity(self):
        communities = self._findCommunity()
        m = len(self.initial_edges)
        modularity = 0

        for comm in communities:
            for pair in combinations(comm, 2):
                ki = len(self.initial_neighbors[pair[0]])
                kj = len(self.initial_neighbors[pair[1]])
                if pair in self.initial_edges:
                    A = 1
                else:
                    A = 0
                temp_mod = A - (ki * kj / (2 * m))
                modularity += temp_mod
        modularity = modularity / (2 * m)
        return communities, modularity


    def _findCommunity(self):
        community = list()
        list_node_unchecked = self.initial_nodes.copy()
        list_node_checked = list()

        while len(list_node_unchecked) > 0:
            init_node = list_node_unchecked.pop(0)
            list_node_checked.append(init_node)
            temp_community = set()
            temp_community.add(init_node)
            temp_list_node_unchecked = list()
            temp_list_node_unchecked.extend(self.current_neighbors[init_node])
            while len(temp_list_node_unchecked) > 0:
                temp_node = temp_list_node_unchecked.pop(0)
                list_node_unchecked.remove(temp_node)
                list_node_checked.append(temp_node)
                temp_community.add(temp_node)
                for neighbor in self.current_neighbors[temp_node]:
                    if neighbor not in list_node_checked and neighbor not in temp_list_node_unchecked:
                        temp_list_node_unchecked.append(neighbor)

            community.append(sorted(list(temp_community)))
        return community


def writeOutputBetweenness(dict_betweenness, output_file_path):
    dict_betweenness = {edge: round(btwn, 5) for edge, btwn in dict_betweenness.items()}
    list_betweenness = sorted(dict_betweenness.items(), key=lambda r: (-r[1], r[0]))
    with open(output_file_path, 'w+') as o:
        o.write('\n'.join([str(tup)[1:-1] for tup in list_betweenness]))
        o.close()

def writeOutputCommunity(community, output_file_path):
    list_comm = sorted(community, key=lambda r: (len(r), r))
    with open(output_file_path, 'w+') as o:
        o.write('\n'.join([str(tup).strip('[').strip(']') for tup in list_comm]))
        o.close()


if __name__ == '__main__':
    time_start = time.time()

    # input params
    # filter_thres = sys.argv[1]
    # input_file_path = sys.argv[2]
    # output_file_path_betweenness = sys.argv[3]
    # output_file_path_comm = sys.argv[4]

    filter_thres = 7
    input_file_path = 'input/ub_sample_data.csv'
    output_file_path_betweenness = 'output/output2_btw.txt'
    output_file_path_comm = 'output/output2_comm.txt'

    # set spark
    spark_config = SparkConf().setMaster('local').setAppName('task1').set('spark.executor.memory', '4g').set(
        'spark.driver.memory', '4g')
    sc = SparkContext(conf=spark_config)
    sc.setLogLevel('OFF')

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

    # get neighbors of each node
    rdd_neighbor = sc.parallelize(list_edges).\
        flatMap(lambda r: [r, (r[1], r[0])]).\
        groupByKey().\
        map(lambda r: (r[0], list(set(r[1]))))
    dict_neighbor = dict(rdd_neighbor.collect())

    # create graphGN object
    graph = graphGN(list_nodes, list_edges, dict_neighbor)

    # compute initial betweenness
    initial_btwn = graph.computeBetweenness()
    writeOutputBetweenness(initial_btwn, output_file_path_betweenness)

    # get best communities
    best_comm, max_mod = graph.getBestCommunities()
    writeOutputCommunity(best_comm, output_file_path_comm)

    print('Duration: ', time.time() - time_start)


