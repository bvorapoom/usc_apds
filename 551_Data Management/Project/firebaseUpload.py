import json
import sys
import requests
import pandas as pd
import numpy as np


def check_path_existence(type, firebase_url, list_node):
    if type == 'dir':
        path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node]) + '.json?writeSizeLimit=unlimited'
    elif type == 'file':
        path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-1]]) + '/file:' + list_node[-1].replace('.', ':') + '.json?writeSizeLimit=unlimited'

    res_check = requests.get(path)
    return json.loads(res_check.text) is not None


def make_directory(firebase_url, dir_path):
    list_node = list(map(str, dir_path.strip('/').split('/')))

    # check if previous path exist
    prev_path = firebase_url + '/'.join(list_node[:-1]) + '.json?writeSizeLimit=unlimited'
    if len(list_node) == 1 or check_path_existence('dir', firebase_url, list_node[:-1]):
        current_path = firebase_url + '/'.join(list_node) + '.json?writeSizeLimit=unlimited'
        if check_path_existence('dir', firebase_url, list_node):
            return 'path: /' + '/'.join(list_node) + ' already exists'
        else:
            content = json.dumps({list_node[-1]: ""})
            res_mkdir = requests.patch(prev_path, content)
            if res_mkdir.status_code == 200:
                update_metadata(action='mkdir', firebase_url=firebase_url, dir_path=dir_path)
                return 'Directory: /' + '/'.join(list_node) + ' created successfully'
            else:
                return 'Fail to create directory'
    else:
        return 'path: /' + '/'.join(list_node[:-1]) + ' does not exist'


def list_directory_content(firebase_url, dir_path):
    list_node = ['dir:' + str(node) for node in dir_path.strip('/').split('/')]
    metadata_request_path = firebase_url + 'metadata/' + '/'.join(list_node) + '.json?writeSizeLimit=unlimited'
    res_meta = requests.get(metadata_request_path)
    list_contents = json.loads(res_meta.text)
    list_contents = [item.split(':')[1] for item in list_contents.keys()]
    return list_contents


def display_file_content(firebase_url, dir_path):
    try:
        file_path = firebase_url + dir_path.strip('/').replace('.', ':') + '.json?writeSizeLimit=unlimited'
        res_file = requests.get(file_path)
        list_dfs = [pd.DataFrame.from_records(p_data) for p_num, p_data in json.loads(res_file.text).items()]
        df_merge = pd.concat(list_dfs)
        return df_merge.reset_index()
    except:
        return 'File path does not exist'


def remove_file(firebase_url, dir_path):
    list_node = list(map(str, dir_path.strip('/').replace('.', ':').split('/')))
    current_path = firebase_url + '/'.join(list_node) + '.json?writeSizeLimit=unlimited'
    if check_path_existence('file', firebase_url, list_node):
        if len(list_node) == 1: # if to remove root node, just delete
            res_del = requests.delete(current_path)
            if res_del.status_code == 200:
                update_metadata(action='rm', case=1, firebase_url=firebase_url, list_node=list_node)
                return 'Delete file: ' + list_node[-1].replace(':', '.') + ' successfully'
            else:
                return 'Fail to delete file: ' + list_node[-1].replace(':', '.')

        else:
            # check if there are other dirs or files in the path
            prev_meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-1]]) + '.json?writeSizeLimit=unlimited'
            res_prev_meta_path = requests.get(prev_meta_path)
            # if yes, just delete
            if len(json.loads(res_prev_meta_path.text)) > 1:
                res_del = requests.delete(current_path)
                if res_del.status_code == 200:
                    update_metadata(action='rm', case=2, firebase_url=firebase_url, list_node=list_node)
                    return 'Delete file: ' + list_node[-1].replace(':', '.') + ' successfully'
                else:
                    return 'Fail to delete file: ' + list_node[-1].replace(':', '.')

            # if no, use put to update value of prev path to "" so that the previous path is not deleted too
            else:
                prev_path = firebase_url + '/'.join(list_node[:-1]) + '.json?writeSizeLimit=unlimited'
                prev_2_path = firebase_url + '/'.join(list_node[:-2]) + '.json?writeSizeLimit=unlimited'
                res_del = requests.put(prev_path, '"temp"')
                res_del = requests.patch(prev_2_path, '{"' + list_node[-2] + '": ""}')
                if res_del.status_code == 200:
                    update_metadata(action='rm', case=3, firebase_url=firebase_url, list_node=list_node)
                    return 'Delete file: ' + list_node[-1].replace(':', '.') + ' successfully'
                else:
                    return 'Fail to delete file: ' + list_node[-1].replace(':', '.')

    else:
        return 'Path does not exist'


def remove_directory(firebase_url, dir_path):
    list_node = list(map(str, dir_path.strip('/').replace('.', ':').split('/')))
    current_path = firebase_url + '/'.join(list_node) + '.json?writeSizeLimit=unlimited'
    if check_path_existence('dir', firebase_url, list_node):
        if len(list_node) == 1:  # if to remove root node, just delete
            res_del = requests.delete(current_path)
            if res_del.status_code == 200:
                update_metadata(action='rmdir', case=1, firebase_url=firebase_url, list_node=list_node)
                return 'Delete directory: ' + list_node[-1].replace(':', '.') + ' successfully'
            else:
                return 'Fail to delete directory: ' + list_node[-1].replace(':', '.')

        else:
            # check if there are other dirs or files in the path
            prev_meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-1]]) + '.json?writeSizeLimit=unlimited'
            res_prev_meta_path = requests.get(prev_meta_path)
            # if yes, just delete
            if len(json.loads(res_prev_meta_path.text)) > 1:
                res_del = requests.delete(current_path)
                if res_del.status_code == 200:
                    update_metadata(action='rmdir', case=2, firebase_url=firebase_url, list_node=list_node)
                    return 'Delete directory: ' + list_node[-1] + ' successfully'
                else:
                    return 'Fail to delete directory: ' + list_node[-1]

            # if no, use put to update value of prev path to "" so that the previous path is not deleted too
            else:
                prev_path = firebase_url + '/'.join(list_node[:-1]) + '.json?writeSizeLimit=unlimited'
                prev_2_path = firebase_url + '/'.join(list_node[:-2]) + '.json?writeSizeLimit=unlimited'
                res_del = requests.put(prev_path, '"temp"')
                res_del = requests.patch(prev_2_path, '{"' + list_node[-2] + '": ""}')
                if res_del.status_code == 200:
                    update_metadata(action='rmdir', case=3, firebase_url=firebase_url, list_node=list_node)
                    return 'Delete directory: ' + list_node[-1].replace(':', '.') + ' successfully'
                else:
                    return 'Fail to delete directory: ' + list_node[-1].replace(':', '.')

    else:
        return 'Path does not exist'


def upload_file(firebase_url, dir_path, file_path, num_partition=None, col_partition_by=None):
    list_node = list(map(str, dir_path.strip('/').split('/')))
    current_path = firebase_url + '/'.join(list_node) + '.json?writeSizeLimit=unlimited'
    if check_path_existence('dir', firebase_url, list_node[:-1]):
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

        dict_result = dict()
        dict_metadata = dict()
        file_name = file_path.strip('/').split('/')[-1].replace('.', ':')
        upload_path = firebase_url + '/'.join(list_node) + '/' + file_name + '.json?writeSizeLimit=unlimited'
        for i, df_p in enumerate(df_split):
            dict_result['p' + str(i+1) + ':' + file_name] = df_p.to_dict(orient='records')
            dict_metadata['p' + str(i+1)] = '/' + '/'.join(list_node) + '/' + file_name + '/' + 'p' + str(i+1) + ':' + file_name

        res_put = requests.put(upload_path, json.dumps(dict_result))
        if res_put.status_code == 200:
            update_metadata(action='put', firebase_url=firebase_url, dir_path=dir_path, file_name=file_name,
                            dict_metadata=dict_metadata)
            return 'File: ' + file_name + ' uploaded successfully'
        else:
            return 'Fail to upload file'

    else:
        return 'path: /' + '/'.join(list_node) + ' does not exist'


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict and key.split(':')[0] == 'dir':
            yield from recursive_items(value)
        else:
            yield (key, value)


def get_partition_locations(firebase_url, file_name):
    metadata_path = firebase_url + 'metadata/.json?writeSizeLimit=unlimited'
    res_metadata = requests.get(metadata_path)
    dict_meta = json.loads(res_metadata.text)

    file_name_reformat = 'file:' + file_name.replace('.', ':')

    partition_locations = None
    for key, value in recursive_items(dict_meta):
        if key == file_name_reformat:
            partition_locations = value
            break

    if partition_locations is not None:
        return partition_locations
    else:
        return 'Cannot find file: ' + file_name + ' in the database'



def read_partition(firebase_url, file_name, partition_num):
    partition_locations = get_partition_locations(firebase_url, file_name)

    if partition_num in range(1, len(partition_locations)+1):
        dir_path = firebase_url + partition_locations['p' + str(partition_num)].strip('/') + '.json?writeSizeLimit=unlimited'
        res_partition = requests.get(dir_path)
        df_partition = pd.DataFrame.from_records(json.loads(res_partition.text))
        return df_partition
    else:
        return 'Input partition number: ' + str(partition_num) + ' out of range'


def update_metadata(action, **kwargs):
    if action == 'mkdir':
        list_node = ['dir:' + str(node) for node in kwargs['dir_path'].strip('/').split('/')]
        metadata_path = kwargs['firebase_url'] + 'metadata/' + '/'.join(list_node[:-1]) + '.json?writeSizeLimit=unlimited'
        meta = json.dumps({list_node[-1]: ""})
        res_metadata = requests.patch(metadata_path, meta)
    elif action == 'put':
        list_node = ['dir:' + str(node) for node in kwargs['dir_path'].strip('/').split('/')]
        metadata_path = kwargs['firebase_url'] + 'metadata/' + '/'.join(list_node) + '/file:' + kwargs['file_name'] + '.json?writeSizeLimit=unlimited'
        meta = json.dumps(kwargs['dict_metadata'])
        res_metadata = requests.patch(metadata_path, meta)
    elif action == 'rm':
        list_node = kwargs['list_node']
        firebase_url = kwargs['firebase_url']
        if kwargs['case'] == 1:
            root_meta_path = firebase_url + 'metadata/' + '.json?writeSizeLimit=unlimited'
            res_prev_meta_path = requests.get(root_meta_path)
            # check number of elements in root node
            if len(json.loads(res_prev_meta_path.text)) > 1:
                meta_path = firebase_url + 'metadata/dir:' + list_node[0] + '.json?writeSizeLimit=unlimited'
                res_meta_del = requests.delete(meta_path)
            else:
                meta_path = firebase_url + 'metadata.json?writeSizeLimit=unlimited'
                res_meta_del = requests.delete(meta_path)
                meta_path = firebase_url + '.json?writeSizeLimit=unlimited'
                res_meta_del = requests.put(meta_path, '{"metadata": ""}')
        elif kwargs['case'] == 2:
            meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-1]]) + '/file:' + list_node[-1].replace('.', ':') + '.json?writeSizeLimit=unlimited'
            res_meta_del = requests.delete(meta_path)
        elif kwargs['case'] == 3:
            prev_meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-1]]) + '.json?writeSizeLimit=unlimited'
            prev_2_meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-2]]) + '.json?writeSizeLimit=unlimited'
            res_meta_del = requests.put(prev_meta_path, '"temp"')
            res_meta_del = requests.patch(prev_2_meta_path, '{"dir:' + list_node[-2] + '": ""}')
    elif action == 'rmdir':
        list_node = kwargs['list_node']
        firebase_url = kwargs['firebase_url']
        if kwargs['case'] == 1:
            root_meta_path = firebase_url + 'metadata/' + '.json?writeSizeLimit=unlimited'
            res_prev_meta_path = requests.get(root_meta_path)
            # check number of elements in root node
            if len(json.loads(res_prev_meta_path.text)) > 1:
                meta_path = firebase_url + 'metadata/dir:' + list_node[0] + '.json?writeSizeLimit=unlimited'
                res_meta_del = requests.delete(meta_path)
            else:
                meta_path = firebase_url + 'metadata.json?writeSizeLimit=unlimited'
                res_meta_del = requests.delete(meta_path)
                meta_path = firebase_url + '.json?writeSizeLimit=unlimited'
                res_meta_del = requests.put(meta_path, '{"metadata": ""}')
        elif kwargs['case'] == 2:
            meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node]) + '.json?writeSizeLimit=unlimited'
            res_meta_del = requests.delete(meta_path)
        elif kwargs['case'] == 3:
            prev_meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-1]]) + '.json?writeSizeLimit=unlimited'
            prev_2_meta_path = firebase_url + 'metadata/' + '/'.join(['dir:' + node for node in list_node[:-2]]) + '.json?writeSizeLimit=unlimited'
            res_meta_del = requests.put(prev_meta_path, '"temp"')
            res_meta_del = requests.patch(prev_2_meta_path, '{"dir:' + list_node[-2] + '": ""}')


def terminal_command(command, firebase_url):
    command_list = ['mkdir', 'ls', 'cat', 'rm', 'rmdir','put', 'getPartitionLocations', 'readPartition']

    components = command.split()

    if components[0] in command_list:
        if components[0] == 'mkdir':
            assert len(components) == 2, 'Incorrect number of arguments'
            return make_directory(firebase_url, components[1])
        elif components[0] == 'ls':
            assert len(components) == 2, 'Incorrect number of arguments'
            return list_directory_content(firebase_url, components[1])
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
            return upload_file(firebase_url, dir_path, file_path, num_partition, col_partition_by)
        elif components[0] == 'getPartitionLocations':
            assert len(components) == 2, 'Incorrect number of arguments'
            return get_partition_locations(firebase_url, components[1])
        elif components[0] == 'readPartition':
            assert len(components) == 3, 'Incorrect number of arguments'
            return read_partition(firebase_url, components[1], int(components[2]))
        elif components[0] == 'cat':
            assert len(components) == 2, 'Incorrect number of arguments'
            return display_file_content(firebase_url, components[1])
        elif components[0] == 'rm':
            return remove_file(firebase_url, components[1])
        elif components[0] == 'rmdir':
            return remove_directory(firebase_url, components[1])
    else:
        return 'Please input valid command'


def format_structure(structure):
    formatted_structure = dict()
    for key, val in structure.items():
        type, name = key.split(':', 1)
        if type == 'dir':
            if len(val) == 0 or val == '':
                formatted_structure[name] = {}
            else:
                formatted_structure[name] = format_structure(val)
        elif type == 'file':
            formatted_structure[name] = list(val.keys())
    return formatted_structure

def get_structure():
    firebase_url = 'https://dsci551-project-40a58-default-rtdb.firebaseio.com/'
    metadata_request_path = firebase_url + 'metadata/.json?writeSizeLimit=unlimited'
    res_file = requests.get(metadata_request_path)
    structure = json.loads(res_file.text)
    structure = format_structure(structure)

    return structure


if __name__ == '__main__':
    # command = ' '.join(sys.argv[1:])

    # create directories
    command = 'mkdir data/'
    command = 'mkdir data/car_accidents'
    command = 'mkdir data/employment'
    command = 'mkdir data/gdp'
    command = 'mkdir data/pop'
    command = 'mkdir data/weather'
    #
    # list content in directory
    command = 'ls data/'
    #
    # upload data: put <path_data> <path_firebase> <number of partition>(optional) <column to partition>(optional)
    command = 'put data/cleaned/data_accident_2016_2021.csv /data/car_accidents 6 year' #specify both col_partition and num_partition
    command = 'put data/cleaned/data_employment_2015_2019.csv /data/employment 4' # specify only num_partition
    command = 'put data/cleaned/data_gdp_2015_2020.csv /data/gdp year' # specify only col_partition
    command = 'put data/cleaned/data_pop_2015_2019.csv /data/pop' # not specify both optional args
    command = 'put data/cleaned/data_weather_2016_2021.csv /data/weather' # not specify both optional args
    #
    # get partition locations of a file (from metadata)
    command = 'getPartitionLocations data_accident_2016_2021.csv'
    #
    # read a particular partition of a file
    # command = 'readPartition data_accident_2016_2021.csv 3'
    #
    # display file content
    # command = 'cat /data/car_accidents/data_accident_2016_2021.csv'
    #
    # remove file
    # command = 'rm /data/car_accidents/data_accident_2016_2021.csv'
    #
    # remove directories
    # command = 'rmdir /data/car_accidents'
    # command = 'rmdir /data'

    firebase_url = 'https://dsci551-project-40a58-default-rtdb.firebaseio.com/'
    current_path = firebase_url + 'test' + '.json?writeSizeLimit=unlimited'
    res_del = requests.delete(current_path)

    # dir_path = 'test/'
    # list_node = ['dir:' + str(node) for node in dir_path.strip('/').split('/')]
    # metadata_path = firebase_url + 'metadata/' + '/'.join(list_node[:-1]) + '.json?writeSizeLimit=unlimited'
    # meta = json.dumps({list_node[-1]: ""})
    # res_metadata = requests.patch(metadata_path, meta)

    #
    # result = terminal_command(command, firebase_url)
    # print(result)
















