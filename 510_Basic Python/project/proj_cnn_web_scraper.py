from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import argparse



def get_html_from_url(url):
    ''' get soup from input url
    params:
        url : str, url of the web that will get soup from
    returns: 
        BeautifulSoup object
    '''
    try:
        content = requests.get(url)
        soup = BeautifulSoup(content.content, 'html.parser')
        return soup
    except:
        return 'Incorrect URL for getting HTML'
    
    
def get_list_of_thailand_cities_from_cnn():
    ''' get list of Thailand cities from CNN website
    returns:
        Dataframe object : list of Thailand cities
    '''
    url = 'https://www.cnn.com/travel/article/thailand-travel-77-provinces-guide/index.html'
    soup = get_html_from_url(url)
    thailand_cities_list = []
    tags = soup.select('div.Paragraph__component > span > h3 > strong')
    for tag in tags:
        return_text = tag.get_text().strip()
        city_name = re.search(r'\d{1,2}\.\s([a-zA-Z\s]+):?', return_text)
        if city_name is not None:
            if city_name.group(1) == 'Sa Kaeow':
                thailand_cities_list.append('Sa Kaeo')
            else:
                thailand_cities_list.append(city_name.group(1))
                
    return thailand_cities_list
    



