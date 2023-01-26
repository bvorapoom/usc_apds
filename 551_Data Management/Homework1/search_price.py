import json
import sys
import requests

firebase_path = 'https://dsci551-hw-4a0a3-default-rtdb.firebaseio.com/hw1/'
key = 'raw'
json_suffix = '.json'


def query_data_by_price_range(url, lower_bound, upper_bound):
    url_with_params = url + '?orderBy="price"&startAt=' + str(lower_bound) + '&endAt=' + str(upper_bound)
    res = requests.get(url_with_params)
    return res


if __name__ == '__main__':
    lower_bound = sys.argv[1]
    upper_bound = sys.argv[2]
    # query cars by price range from Firebase
    res = query_data_by_price_range(firebase_path + key + json_suffix, lower_bound, upper_bound)
    res_dict = json.loads(res.text)
    if len(res_dict) == 0:
        print('No cars found with the given range')
    else:
        list_car_id = [v['car_ID'] for k, v in res_dict.items()]
        print('IDs for the car price range are:', \
              sorted(list(map(int, list_car_id))), sep='\n')
