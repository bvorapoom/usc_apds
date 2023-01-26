import json
import sys
import requests
import re
from collections import Counter

firebase_path = 'https://dsci551-hw-4a0a3-default-rtdb.firebaseio.com/hw1/inverted_index'
json_suffix = '.json'


def query_data(url):
    res = requests.get(url)
    return res


def clean_search_keyword(search_kw):
    return re.sub(r'[^a-zA-Z0-9]+', ' ', search_kw).lower().split()


def filter_search_results(kw_list):
    list_filtered_car = []
    for k in kw_list:
        url_kw = firebase_path + '/' + k + json_suffix
        res_kw = json.loads(query_data(url_kw).text)
        list_filtered_car.extend(res_kw)
    return list_filtered_car


if __name__ == '__main__':
    # clean search keyword
    search_kw = sys.argv[1]
    kw_list = clean_search_keyword(search_kw)
    list_filtered_car = filter_search_results(kw_list)
    if len(list_filtered_car) == 0:
        print('No cars found')
    else:
        # format to sort by frequency of keywords
        list_filtered_car = Counter(list_filtered_car)
        list_filtered_car_formatted = [k for k, v in sorted(list_filtered_car.items(), key=lambda x: (-x[1], -x[0]))]
        print('IDs of the car are:',\
              list_filtered_car_formatted, sep='\n')
