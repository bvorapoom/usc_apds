import sqlalchemy
import pymysql
import paramiko
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder
import regex as re
from sqlalchemy import insert
pymysql.install_as_MySQLdb()
import pandas as pd
import regex as re
import numpy as np
import json


#ssh config
mypkey = paramiko.RSAKey.from_private_key_file('dsci551.pem')
ssh_host = 'ec2-54-176-10-221.us-west-1.compute.amazonaws.com'
ssh_user = 'ec2-user'
ssh_port = 22

#mysql config         
sql_hostname = 'localhost'
sql_username = 'root'
sql_password = 'Dsci-551'
sql_main_database = 'project'
sql_port = 3306
host = '127.0.0.1'



tunnel = SSHTunnelForwarder(
      (ssh_host, ssh_port),
      ssh_username=ssh_user,
      ssh_pkey=mypkey,
      remote_bind_address=(sql_hostname, sql_port))

tunnel.start()

engine = sqlalchemy.create_engine('mysql+pymysql://'+sql_username+':'+sql_password+'@'+host+':'+str(tunnel.local_bind_port)+'/'+sql_main_database)
connection = engine.connect()

def upload_csv(df, table_name):
  df.to_sql(table_name, engine, index=False)

# Have three tables in MySQL to store metadata
# 1. File(file_id, file_name)
# 2. Directory(parent_id, child_id)
# 3. Partition(file_id, partition_id)

def file_table2dict():
  df = pd.read_sql('SELECT * FROM File', con=connection)
  file_dict = df.to_dict('records')
  return file_dict

def directory_table2dict():
  df = pd.read_sql('SELECT * FROM Directory', con=connection)
  dir_dict = df.to_dict('records')
  return dir_dict

def get_file_name_by_id(id):
  file_dict = file_table2dict()
  for file in file_dict:
    if file['file_id'] == id:
      return file['file_name']

def check_directory_exist(dir_path):
  file_dict = file_table2dict()
  dir_dict = directory_table2dict()

  dir_list = dir_path.split("/")
  if dir_list[0]== "":
      dir_list = dir_list[1:]
  
  if dir_list[-1]=="":
    dir_list.pop()

  count = 0
  last_exist = 0
  parent = 0
  for i in range(len(dir_list)):
      #whether there exist a file whose parent is 0
      for file in file_dict:
        if file['file_name'] == dir_list[i]:
          for pair in dir_dict:
            if pair['child_id'] == file['file_id']:
              if pair['parent_id'] == parent:
                parent = pair['child_id']
                count += 1
  
  last_exist = parent

  return count, parent

def make_directory(dir_path):
  dir_list = dir_path.split("/")
  if dir_list[0]== "":
      dir_list = dir_list[1:]
  
  if dir_list[-1]=="":
    dir_list.pop()

  metadata = sqlalchemy.MetaData(engine)

  file_table = sqlalchemy.Table('File', metadata, autoload=True)
  ins_file = file_table.insert()

  directory_table = sqlalchemy.Table('Directory', metadata, autoload=True)
  ins_dir = directory_table.insert()

  count, exist_file = check_directory_exist(dir_path)
  
  if len(dir_list) - 1 == count:
    print("Creating")
    #make directory for the file
    file_name = dir_list[-1]
    connection.execute(ins_file, file_name=file_name)

    #select the last row's file_id as the child_id
    result = connection.execute('SELECT file_id FROM File ORDER BY file_id DESC LIMIT 1')
    new_file_id = result.first()[0]

    parent_id = exist_file
    #insert the (parent_id, child_id) into the Directory Table
    connection.execute(ins_dir, parent_id = parent_id, child_id = new_file_id)
    return str(dir_path) + 'created successfully'
  elif len(dir_list) == count:
    return str(dir_path) + " path already exists"
  else:
    return "Cannot create directory for " + str(dir_path)

def list_content(dir_path):
  dir_list = dir_path.split("/")

  if dir_list[0]== "":
      dir_list = dir_list[1:]
  if dir_list[-1]=="":
    dir_list.pop()

  count, exist_file = check_directory_exist(dir_path)
  dir_dict = directory_table2dict()

  res = list();

  if count == len(dir_list):
    for pair in dir_dict:
      if pair['parent_id'] == exist_file:
        file_name = get_file_name_by_id(pair['child_id'])
        res.append(file_name)
    return res
  else:
    return "The path does not exist"


def remove_file(dir_path):
  dir_list = dir_path.split("/")
  if dir_list[0] == "":
    dir_list = dir_list[1:]
  if dir_list[-1] == "":
    dir_list.pop()

  count, exist_file = check_directory_exist(dir_path)

  if count == len(dir_list):
    file_name = get_file_name_by_id(exist_file)
    # print(exist_file)
    try:
      result = connection.execute('Drop Table ' + file_name)
      metadata = connection.execute(f'delete from partition_locations where file_id = {exist_file}')
      file_table = connection.execute(f'delete from File where file_id = {exist_file}')
      directory_table = connection.execute(f'delete from Directory where child_id = {exist_file}')
      return 'Remove succesfully'
    except:
      return 'file does not exist'
  else:
    return "The path does not exist"

# displaying contents

def list_to_dict(list_path):
    val = list_path[-1]
    for i in range(len(list_path)-2, -1, -1):
        temp = {list_path[i]: val}
        val = temp
    return val

def path_json(list_path):
  final_structure = dict()
  for num, path in enumerate(list_path):
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

  json_path = json.dumps(final_structure)
  return json_path

def display_contents(userpath):

  file_id = check_directory_exist(f'{userpath}')[1]

  df = pd.read_sql("SELECT child_id,parent_id FROM Directory", con=connection)
  df=  df.applymap(str)
  df['Dictionery'] = list(zip(df['child_id'], df['parent_id']))
  ancestry = df['Dictionery']

  def ancestors(p):
    return (ancestors(children[p]) if p in children else []) + [p]

  parents = set()
  children = {}
  for c,p in ancestry:
      parents.add(p)
      children[c] = p

  pathlist = [] 
  for k in (set(children.keys()-parents)):    
    pathlist.append(ancestors(k))

  parent_child = []

  for x in pathlist:
    last_element = int(x[-1])
    if file_id == last_element:
      df = pd.read_sql(f'SELECT  * FROM partition_locations', con=connection)
      df = df[df.file_id == file_id]
      display_contents = pd.DataFrame()
      for i in range(len(df)):
        read_partition = pd.read_sql(f'SELECT  * FROM {df.iloc[i, 0]}_{df.iloc[i, 1]}', con=connection)
        display_contents = pd.concat([display_contents, read_partition], axis=0)
  display_contents = display_contents.reset_index(drop= True)
  return display_contents

def getting_pathlist():
  df = pd.read_sql("SELECT child_id,parent_id FROM Directory", con=connection)
  df=  df.applymap(str)
  df['Dictionery'] = list(zip(df['child_id'], df['parent_id']))
  ancestry = df['Dictionery']
  
  def ancestors(p):
    return (ancestors(children[p]) if p in children else []) + [p]

  parents = set()
  children = {}
  for c,p in ancestry:
      parents.add(p)
      children[c] = p
  pathlist = [] 
  for k in (set(children.keys()-parents)):    
    pathlist.append(ancestors(k))
  for x in pathlist:
    x.pop(0)
  return pathlist

def filepath_json():

  path_list= getting_pathlist()

  df_file = pd.read_sql("SELECT * FROM File", con=connection)
  df_file.set_index('file_id', inplace=True)

  pathlist_named = list()
  for dir in path_list:
    temp = list()
    for id in dir:
      name = df_file.loc[int(id), 'file_name']
      temp.append(name)
    pathlist_named.append(temp)

  df_ptt = pd.read_sql("SELECT * FROM partition_locations", con=connection)
  df_ptt.set_index('file_id', inplace=True)

  pathlist_with_ptt = list()
  for path in pathlist_named:
    temp = path.copy()
    full_path = '/'.join(path)
    id = check_directory_exist(full_path)[1]
    if id in list(df_ptt.index):
      list_ptt = ['p'+str(ptt_id) for ptt_id in list(df_ptt.loc[id, 'partition_id'])]
      temp[-1] = {temp[-1]: list_ptt}
    else:
      temp[-1] = {temp[-1]: {}}
    pathlist_with_ptt.append(temp)
  json = path_json(pathlist_with_ptt)
  return json


def terminal_commands(command):
  command_list = ['mkdir', 'ls', 'cat', 'rm', 'rmdir','put', 'getPartitionLocations', 'readPartition']
  components = command.split()

  if components[0] in command_list:
    if components[0] == 'mkdir':
      assert len(components) == 2, 'Incorrect number of arguments'
      return make_directory(components[1])
    elif components[0] == 'ls':
      assert len(components) == 2, 'Incorrect number of arguments'
      return list_content(components[1])
    elif components[0] == 'put':
      assert len(components) >= 3, 'Missing arguments to upload file to MySQL'
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

      return put(file_path, dir_path, num_partition, col_partition_by)
    elif components[0] == 'getPartitionLocations':
        assert len(components) == 2, 'Incorrect number of arguments'
        return getPartitionLocations(components[1])
    elif components[0] == 'readPartition':
        assert len(components) == 3, 'Incorrect number of arguments'
        return readPartition(components[1], int(components[2]))
    elif components[0] == 'cat':
        assert len(components) == 2, 'Incorrect number of arguments'
        return display_contents(components[1])
    elif components[0] == 'rm':
        return remove_file(components[1])
  else:
    return "Please input valid command"


def put(file,dir_path,num_partition= None, col_partition_by= None):
  # uploading data
  file_name = re.findall(r'(\w+)',file)[0]
  df = pd.read_csv(f'{file}', low_memory=False)
  upload_csv(df, f'{file_name}')

  file_directory_path = str(dir_path)+"/"+ str(file_name)

  make_directory(file_directory_path)

  #getting file_id
  file_id = check_directory_exist(f'{file_directory_path}')[1]

  df_data = pd.read_sql(f'SELECT * FROM {file_name}', con=connection)

  if col_partition_by is not None:
    assert col_partition_by in df_data.columns, "Columsn to partition by does not exist"
    df_sort = df_data.sort_values(df.data.columns[0], ignore_index = True)
  else:
    df_sort = df_data.sort_values(df_data.columns[0], ignore_index = True)

  if num_partition is None:
    num_parititons = 5
    df_split = np.array_split(df_sort,num_parititons)
  else:
    df_split = np.array_split(df_sort, num_partition)

  partitionid_list= []
  counter =0
  for x in df_split:
     counter +=1
     file_partition_instance = [file_id,counter]
     upload_csv(x,f'{file_id}_{counter}')
     partitionid_list.append(file_partition_instance)

  partitionid_fileid_dataframe = pd.DataFrame(partitionid_list, columns = ['file_id' , 'partition_id'])
    # checking if the dataframe already exists or to implement a new table
  try:
    table_exists = pd.read_sql(f'SELECT * FROM partition_locations', con=connection)
    for x in partitionid_list:
      connection.execute(f"INSERT INTO  `project`.`partition_locations` (`file_id` ,`partition_id`) VALUES ('{x[0]}',  '{x[1]}')")
    return "File upload Done!"
  except:
    upload_csv(partitionid_fileid_dataframe, "partition_locations")
    return "File upload Done!"

#put("weather.csv","data/pop", 2)

def getPartitionLocations(file):
  file_id = check_directory_exist(f'{file}')[1]
  partition_locations = pd.read_sql(f'SELECT file_id, partition_id FROM partition_locations WHERE file_id = {file_id}', con=connection)
  return partition_locations

# getPartitionLocations("data/gdp/gdp")

def readPartition(file,partition):
  file_id = check_directory_exist(f'{file}')[1]
  read_partition = pd.read_sql(f'SELECT  * FROM {file_id}_{partition}', con=connection)
  return read_partition

# readPartition("data/gdp/gdp",2)

# command = 'mkdir data/'
# terminal_commands(command)
#
# command = 'mkdir data/car_accidents'
# terminal_commands(command)
#
# command = 'mkdir data/employment'
# terminal_commands(command)
#
# command = 'mkdir data/gdp'
# terminal_commands(command)
#
# command = 'mkdir data/pop'
# terminal_commands(command)
#
# command = 'mkdir data/weather'
# terminal_commands(command)
#
# command = 'ls data/gdp'
# terminal_commands(command)

# command = 'cat data/pop/pop'
# x = terminal_commands(command)
# print(x)

#
# command = 'mkdir data/car_accidents/abc'
# terminal_commands(command)
#
# command = 'put accident.csv data/car_accidents 4'
# terminal_commands(command)


# tunnel.stop()

# x = getting_pathlist()
# print(x)
