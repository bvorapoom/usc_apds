from pymongo import MongoClient
import pandas as pd
import numpy as np

def init_mongo():
    # uri = 'mongodb://ThanawanL:551project@ac-mxkrvs0-shard-00-00.jmrywg2.mongodb.net:27017,ac-mxkrvs0-shard-00-01.jmrywg2.mongodb.net:27017,ac-mxkrvs0-shard-00-02.jmrywg2.mongodb.net:27017/?ssl=true&replicaSet=atlas-ixjcde-shard-0&authSource=admin&retryWrites=true&w=majority'
    uri = 'mongodb+srv://ThanawanL:551projectnew@551project.jmrywg2.mongodb.net/?retryWrites=true&w=majority'
    client = MongoClient(uri)
    db = client.MongoDB
    return db


def convert_collection_to_dataframe(db, cl_name):
    df = pd.DataFrame(list(db[cl_name].find({},{"_id": 0})))
    return df


def convert_id_name(how, value, df_file):
    if how == 'id2name':
        return str(df_file[df_file.id == value]['name'].values[0])
    elif how == 'name2id':
        return int(df_file[df_file.name == value]['id'].values[0])


def check_directory_exist(path):
    list_node = list(map(str, path.strip('/').split('/')))

    db = init_mongo()
    df_file = convert_collection_to_dataframe(db, 'File')
    df_dir = convert_collection_to_dataframe(db, 'Directory')

    try:
        list_node_id = list(map(lambda item: convert_id_name('name2id', item, df_file), list_node))

        exist = True
        for current_id, parent_id in zip(list_node_id[::-1], list_node_id[::-1][1:] + [0]):
            if len(df_dir[df_dir.child == current_id]['parent']) == 0:
                exist = False
                break
            else:
                check_parent_id = df_dir[df_dir.child == current_id]['parent'].values[0]
                if check_parent_id != parent_id:
                    exist = False
                    break
        return exist
    except:
        return False



# print(check_directory_exist('/test'))


def make_directory(path):
    db = init_mongo()
    list_node = list(map(str, path.strip('/').split('/')))

    df_file = convert_collection_to_dataframe(db, 'File')
    if len(df_file) == 0:
        this_id = 1
    else:
        this_id = int(df_file['id'].max() + 1)

    # print(convert_id_name('id2name', 2, df_file))

    # if create root folder, just create. Else, check if prev directory exists
    if len(list_node) == 1:
        # add data to File collection
        db['File'].insert_one({'id': this_id, 'name': list_node[0]})
        # add data to Directory collection
        db['Directory'].insert_one({'parent': 0, 'child': this_id})
        return 'Make directory: ' + path + ' successfuly'
    else:
        if check_directory_exist(path):
            return 'Directory: ' + path + ' already exists'
        else:
            parent_path = '/'.join(list_node[:-1])
            if check_directory_exist(parent_path):
                # add data to File collection
                db['File'].insert_one({'id': this_id, 'name': list_node[-1]})
                # add data to Directory collection
                parent_id = convert_id_name('name2id', list_node[-2], df_file)
                db['Directory'].insert_one({'parent': parent_id, 'child': this_id})
                return 'Make directory: ' + path + ' successfuly'
            else:
                return 'Parent directory: ' + parent_path + ' does not exist'


def list_directory_content(path):
    db = init_mongo()
    list_node = list(map(str, path.strip('/').split('/')))

    df_file = convert_collection_to_dataframe(db, 'File')
    df_dir = convert_collection_to_dataframe(db, 'Directory')

    if check_directory_exist(path):
        this_id = convert_id_name('name2id', list_node[-1], df_file)
        list_child_id = list(df_dir[df_dir.parent == this_id]['child'])
        list_child_name = list(map(lambda item: convert_id_name('id2name', item, df_file), list_child_id))
        return list_child_name
    else:
        return 'Directory: ' + path + ' does not exist'

# x = list_directory_content('/data')
# print(x)

def upload_file(path, file_path, num_partition=None, col_partition_by=None):

    list_node = list(map(str, path.strip('/').split('/')))
    file_name = file_path.strip('/').split('/')[-1]

    if check_directory_exist(path):
        if check_directory_exist(path.strip('/') + '/' + file_name):
            return file_name + ' already exists under directory: ' + path
        else:
            db = init_mongo()
            df_file = convert_collection_to_dataframe(db, 'File')

            this_id = int(df_file['id'].max() + 1)
            parent_id = convert_id_name('name2id', list_node[-1], df_file)

            # read data into dataframe & split into partitions
            df_data = pd.read_csv(file_path)
            df_data = df_data.replace({np.nan: None})

            if col_partition_by is not None:
                assert col_partition_by in df_data.columns, 'Columns to partition by does not exist'
                df_sort = df_data.sort_values(col_partition_by, ignore_index=True)
            else:
                df_sort = df_data.sort_values(df_data.columns[0], ignore_index=True)

            if num_partition is None:
                # if num partition not specified, default to 5
                num_partition = 5
                df_split = np.array_split(df_sort, num_partition)
            else:
                df_split = np.array_split(df_sort, num_partition)

            # upload each partition into mongo's collections
            for i, df_sp in enumerate(df_split, start=1):
                cl_name = str(this_id) + '_' + str(i)
                db.create_collection(cl_name)

                db[cl_name].insert_many(df_sp.to_dict('records'))

            # update metadata File
            db['File'].insert_one({'id': this_id, 'name': file_name})
            # update metadata Directory
            db['Directory'].insert_one({'parent': parent_id, 'child': this_id})
            # update metadata Partition
            db['Partition'].insert_many([{'fileid': this_id,'partitionid': i} for i in range(1, num_partition+1)])

            return 'File: ' + file_name + ' uploaded successfully'

    else:
        return 'Directory: ' + path + ' does not exist'


# x = upload_file('/data/car_accidents', 'data/cleaned/data_accident_2016_2021.csv', num_partition=None, col_partition_by=None)
# print(x)

def get_partition_locations(path):
    list_node = list(map(str, path.strip('/').split('/')))

    db = init_mongo()
    df_file = convert_collection_to_dataframe(db, 'File')
    df_partition = convert_collection_to_dataframe(db, 'Partition')
    file_id = convert_id_name('name2id', list_node[-1], df_file)

    list_partition = list(df_partition[df_partition.fileid == file_id]['partitionid'])
    list_collection_name = [str(file_id) + '_' + str(ptt_id) for ptt_id in list_partition]

    return list_collection_name


# x = get_collection_name_from_file('/data/car_accidents/data_accident_2016_2021.csv')
# print(x)

def display_file_content(path):
    if check_directory_exist(path):
        db = init_mongo()
        list_collection_name = get_partition_locations(path)
        list_dfs = [convert_collection_to_dataframe(db, cl_name) for cl_name in list_collection_name]
        df_merge = pd.concat(list_dfs)
        return df_merge.reset_index()
    else:
        return 'File path does not exist'

# x = display_file_content('/data/pop/data_pop_2015_2019.csv')
# print(x)


def remove_file(path):
    if check_directory_exist(path):
        db = init_mongo()
        df_file = convert_collection_to_dataframe(db, 'File')
        list_node = list(map(str, path.strip('/').split('/')))

        file_name = list_node[-1]
        file_id = convert_id_name('name2id', file_name, df_file)


        list_collection_name = get_partition_locations(path)

        # drop collection
        for cl_name in list_collection_name:
            db[cl_name].drop()

        # update metadata File
        db['File'].delete_one({'id': file_id})
        # update metadata Directory
        db['Directory'].delete_one({'child': file_id})
        # update metadata Partition
        db['Partition'].delete_many({'fileid': file_id})

        return 'Delete file: ' + list_node[-1] + ' successfully'
    else:
        return 'File path does not exist'

# x = remove_file('/data/pop/data_pop_2015_2019.csv')
# print(x)

def read_partition(path, partition_num):
    partition_locations = get_partition_locations(path)

    if partition_num in range(1, len(partition_locations)+1):
        for cl_name in partition_locations:
            if cl_name.endswith('_' + str(partition_num)):
                target_cl_name = cl_name
                break
        db = init_mongo()
        df_partition = convert_collection_to_dataframe(db, target_cl_name)

        return df_partition
    else:
        return 'Input partition number: ' + str(partition_num) + ' out of range'

# x = read_partition('/data/pop/data_pop_2015_2019.csv', 3)
# print(x)


def terminal_command(command):
    command_list = ['mkdir', 'ls', 'cat', 'rm', 'rmdir','put', 'getPartitionLocations', 'readPartition']

    components = command.split()

    if components[0] in command_list:
        if components[0] == 'mkdir':
            assert len(components) == 2, 'Incorrect number of arguments'
            return make_directory(components[1])
        elif components[0] == 'ls':
            assert len(components) == 2, 'Incorrect number of arguments'
            return list_directory_content(components[1])
        elif components[0] == 'put':
            assert len(components) >= 3, 'Missing arguments to upload file to Firebase'
            file_path = components[1]
            dir_path = components[2]
            num_partition = None
            col_partition_by = None
            if len(components) == 4:
                try:
                    num_partition = int(components[3])
                    col_partition_by = None
                except:
                    col_partition_by = components[3]
            if len(components) == 5:
                col_partition_by = components[4]
                num_partition = int(components[3])
            return upload_file(dir_path, file_path, num_partition, col_partition_by)
        elif components[0] == 'getPartitionLocations':
            assert len(components) == 2, 'Incorrect number of arguments'
            return get_partition_locations(components[1])
        elif components[0] == 'readPartition':
            assert len(components) == 3, 'Incorrect number of arguments'
            return read_partition(components[1], int(components[2]))
        elif components[0] == 'cat':
            assert len(components) == 2, 'Incorrect number of arguments'
            return display_file_content(components[1])
        elif components[0] == 'rm':
            return remove_file(components[1])
        # elif components[0] == 'rmdir':
        #     return remove_directory(firebase_url, components[1])
    else:
        return 'Please input valid command'


def get_structure():
    db = init_mongo()
    df_file = convert_collection_to_dataframe(db, 'File')
    df_dir = convert_collection_to_dataframe(db, 'Directory')
    df_partition = convert_collection_to_dataframe(db, 'Partition')

    # get all paths of leaves
    set_parent = set(df_dir['parent'])
    set_child = set(df_dir['child'])
    list_leaf = list(set_child - set_parent)

    path_list = list()
    for leaf in list_leaf:
        temp_path = list()
        parent = df_dir[df_dir.child == leaf]['parent']
        if len(parent) == 0:
            temp_path = [leaf]
        else:
            parent_id = parent.values[0]
            current_id = leaf
            temp_path.append(current_id)
            while parent_id != 0:
                current_id = parent_id
                parent_id = df_dir[df_dir.child == current_id]['parent'].values[0]
                temp_path.append(current_id)

        path_list.append(temp_path[::-1])


    # convert path_list from id to name
    pathlist_named = [list(map(lambda item: convert_id_name('id2name', item, df_file), path)) for path in path_list]

    # get partitions
    pathlist_with_ptt = list()
    for path in pathlist_named:
        temp = path.copy()
        full_path = '/'.join(path)
        id = convert_id_name('name2id', path[-1], df_file)
        if id in list(df_partition['fileid']):
            list_ptt = ['p' + str(ptt_id) for ptt_id in list(df_partition[df_partition.fileid == id]['partitionid'])]
            temp[-1] = {temp[-1]: list_ptt}
        else:
            temp[-1] = {temp[-1]: {}}

        pathlist_with_ptt.append(temp)


    # convert path_list to JSON format
    def list_to_dict(list_path):
        val = list_path[-1]
        for i in range(len(list_path)-2, -1, -1):
            temp = {list_path[i]: val}
            val = temp
        return val

    final_structure = dict()
    for num, path in enumerate(pathlist_with_ptt):
        if num == 0:
            final_structure = list_to_dict(path)
        else:
            temp = final_structure
            for n, name in enumerate(path):
                if isinstance(name, dict):
                    temp.update(name)
                else:
                    if name not in temp.keys():
                        temp.update(list_to_dict(path[n:]))
                        break
                    else:
                        temp = temp[name]


    return final_structure


if __name__ == '__main__':
    pass
    # db = init_mongo()
    # temp = db.list_collection_names()
    # print(temp)
    # command = 'mkdir /data'
    # command = 'mkdir data/car_accidents'
    # command = 'mkdir data/employment'
    # command = 'mkdir data/gdp'
    # command = 'mkdir data/pop'
    # command = 'mkdir data/weather'
    # command = 'mkdir test/test234'

    # command = 'put data/cleaned/data_accident_2016_2021.csv /data/car_accidents 6 year'  # specify both col_partition and num_partition
    # command = 'put data/cleaned/data_employment_2015_2019.csv /data/employment 4'  # specify only num_partition
    # command = 'put data/cleaned/data_gdp_2015_2020.csv /data/gdp year'  # specify only col_partition
    # command = 'put data/cleaned/data_pop_2015_2019.csv /data/pop'  # not specify both optional args
    # command = 'put data/cleaned/data_weather_2016_2021.csv /data/weather 7 year'  # not specify both optional args



    # output = terminal_command(command)
    # print(output)