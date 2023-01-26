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
    
    

def gen_wiki_url_from_city_list(base_wiki_url, city_list):
    ''' generate wikipedia URLs for each Thailand city
    params:
        base_wiki_url : str
        city_list: list, list of Thailand cities
    returns:
        dictionary: key = city name, value = wikipedia URL for that city
    '''
    wiki_url_dict = {}
    for city in city_list:
        wiki_url = base_wiki_url + city.replace(' ', '_') + '_Province'
        wiki_url_dict[city] = wiki_url
    return wiki_url_dict


def get_info_from_wiki_url(wiki_url):
    ''' scrape info (population, area, density) from wikipedia URL given
    params:
        wiki_url : str, wikipedia URL that will be scraped (needs to be Thailand city wikipedia link)
    returns:
        area : float, area of Thailand city
        pop : float, population of Thailand city
        density : float, population density of Thailand city
    '''
    soup = get_html_from_url(wiki_url)
    
    # get area of the province
    tag_area_title = soup.select('table > tbody > tr.mergedtoprow > th:-soup-contains("Area")')
    tag_area_list = tag_area_title[0].find_parent('tr').find_next_siblings('tr', limit = 3)
    
    for tag_area in tag_area_list:
        if any(substr in tag_area.find('th').get_text() for substr in ['Total', 'City', 'Province']):
            matched_tag_area = tag_area
            
    tag_area_val = matched_tag_area.find('td')
    area = tag_area_val.get_text()
    area = float(re.search(r'([\d\,]+).*?', area).group(0).replace(',',''))
    
    # get total population and population density of the province
    tag_pop_title = soup.select('table > tbody > tr.mergedtoprow > th:-soup-contains("Population")')
    tag_pop_list = tag_pop_title[0].find_parent('tr').find_next_siblings('tr', limit = 5)
    
    for tag_pop in tag_pop_list:
        if any(substr in tag_pop.find('th').get_text() for substr in ['Total', 'City', 'Province']):
            matched_tag_pop_total = tag_pop
        elif tag_pop.find('th').get_text().endswith('Density'):
            matched_tag_density = tag_pop
            
    tag_pop_val = matched_tag_pop_total.find('td')
    pop = tag_pop_val.get_text()
    pop = float(re.search(r'([\d\,]+).*?', pop).group(0).replace(',',''))
        
    tag_density_val = matched_tag_density.find('td')
    density = tag_density_val.get_text()
    density = float(re.search(r'([\d\,]+).*?', density).group(0).replace(',',''))
    
    
    return area, pop, density
        
    
def popmain_get_population_data_all_thailand_cities(city_list):
    ''' get population/area/density info by scraping wikipedia for all Thailand cities
    params:
        city_list : list, list of Thailand cities
    returns:
        Dataframe that contains city, area, population, density info
    '''
    base_wiki_url = 'https://en.wikipedia.org/wiki/'
    wiki_url_dict = gen_wiki_url_from_city_list(base_wiki_url, city_list)
    
    df_cols = ['City', 'Area (km2)', 'Population', 'Population Density (/km2)']
    info_df = pd.DataFrame(columns = df_cols)
    
    for city, wiki_url in wiki_url_dict.items():
        area, pop, density = get_info_from_wiki_url(wiki_url)
        info_df.loc[len(info_df.index)] = [city, area, pop, density]
        
    return info_df


