from lxml import etree
import pandas as pd
from datetime import datetime
import sys

# pd.set_option('display.max_columns', 15)


def read_inode_from_tree_to_df(tree):
    dict_raw = dict()
    for i, inode in enumerate(tree.xpath('//INodeSection/inode')):
        dict_temp = dict()

        target_tag = ['id', 'type', 'name', 'mtime', 'permission', 'blocks']
        for tag in target_tag:
            if tag == 'blocks':
                num_blocks = len(inode.xpath('blocks/block'))
                num_bytes = sum(map(int, inode.xpath('blocks/block/numBytes/text()')))
                dict_temp['num_blocks'] = num_blocks
                dict_temp['num_bytes'] = num_bytes
            else:
                val = inode.xpath(str(tag) + '/text()')
                if len(val) == 0:
                    dict_temp[tag] = None
                else:
                    dict_temp[tag] = val[0]

        dict_raw[i] = dict_temp

    df_raw = pd.DataFrame.from_dict(dict_raw, orient='index')

    df_selection = df_raw[['id', 'type', 'name', 'mtime', 'permission', 'num_blocks', 'num_bytes']]
    df_selection.columns = ['raw_' + col for col in df_selection.columns]
    return df_selection


def convert_timestamp_to_datetime(timestamp, format='%m/%d/%Y %H:%M'):
    dt = datetime.utcfromtimestamp(int(timestamp) / 1000)
    return dt.strftime(format)


def clean_numeric_values(val):
    if pd.isna(val):
        return 0
    else:
        return int(val)


def get_permission(raw_permission, filetype):
    output = ''

    # get the first char of permission
    if str(filetype).lower() == 'directory':
        output += 'd'
    elif str(filetype).lower() == 'file':
        output += '-'

    def convert_to_rwx(val):
        pm_out = ''
        pm_code = ['r', 'w', 'x']
        for i, v in enumerate(val):
            if int(v) == 1:
                pm_out += pm_code[i]
            else:
                pm_out += '-'
        return pm_out

    permission = str(raw_permission)[-3:]
    for pm in permission:
        pm_bit = bin(int(pm))[2:].zfill(3)
        output += convert_to_rwx(pm_bit)

    return output


def get_parent_child_path_df(tree):
    df_path = pd.DataFrame(columns=['child', 'parent'])
    for dir in tree.xpath('//INodeDirectorySection/directory'):
        dir_parent = dir.xpath('parent/text()')[0]
        for child in dir.xpath('child'):
            df_path.loc[len(df_path), :] = [child.text, dir_parent]
    return df_path


def get_full_path(id, df, df_path):
    full_path = ''
    current_id = id
    current_path = df[df['raw_id'] == str(current_id)]['raw_name'].values[0]
    if current_path is None:
        full_path = '/'
    else:
        while current_path is not None:
            full_path = '/' + current_path + full_path
            next_parent_id = df_path[df_path['child'] == str(current_id)]['parent'].values[0]
            current_path = df[df['raw_id'] == str(next_parent_id)]['raw_name'].values[0]
            current_id = next_parent_id
    return full_path


def write_to_tsv(df, output_path):
    df.to_csv(output_path, sep='\t')


if __name__ == '__main__':
    input_xml_path = sys.argv[1]
    output_tsv_path = sys.argv[2]

    tree = etree.parse(input_xml_path)

    df = read_inode_from_tree_to_df(tree)

    df_path = get_parent_child_path_df(tree)
    # get full path of each file or directory
    df['Path'] = df['raw_id'].apply(get_full_path, df=df, df_path=df_path)

    # convert raw_permission to permission in rwx format
    df['Permission'] = df.apply(lambda x: get_permission(x.raw_permission, x.raw_type), axis=1)

    # converting raw mtime (timestamp) to datetime format
    df['ModificationTime'] = df['raw_mtime'].apply(convert_timestamp_to_datetime, format='%m/%d/%Y %H:%M')

    # clean num_blocks and num_bytes column: replacing nan
    df['BlocksCount'] = df['raw_num_blocks'].apply(clean_numeric_values)
    df['FileSize'] = df['raw_num_bytes'].apply(clean_numeric_values)

    # columns selection
    df_final = df[['Path', 'ModificationTime', 'BlocksCount', 'FileSize', 'Permission']]

    # write output to tsv file
    write_to_tsv(df_final, output_tsv_path)