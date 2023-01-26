from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .master("local[1]") \
    .appName("hw5") \
    .getOrCreate()


# 3a
country_df = spark.read.json('country.json')
country_rdd = country_df.rdd
result = country_rdd.map(lambda r: r['GNP']).filter(lambda r: r >= 10000 and r <= 20000).count()
print(result)


# 3b
country_df = spark.read.json('country.json')
country_rdd = country_df.rdd
result = country_rdd.map(lambda r: (r['Continent'], r['GNP'])).groupByKey().mapValues(max).collect()
print(result)


# 3c
country_df = spark.read.json('country.json')
country_rdd = country_df.rdd
city_df = spark.read.json('city.json')
city_rdd = city_df.rdd
result = country_rdd.map(lambda r: (r['Capital'], r['Name'])).\
    join(city_rdd.map(lambda r: (r['ID'], r['Name']))).\
    map(lambda r: r[1]). \
    sortByKey(ascending=False).\
    take(20)
print(result)


# 3d
city_df = spark.read.json('city.json')
city_rdd = city_df.rdd
result = city_rdd.filter(lambda r: r['CountryCode'] == 'USA').map(lambda r: r['Population']).max()
print(result)


# 3e
countrylanguage_df = spark.read.json('countrylanguage.json')
countrylanguage_rdd = countrylanguage_df.rdd
result = countrylanguage_rdd.filter(lambda r: r['IsOfficial'] == 'T').\
    map(lambda r: (r['CountryCode'], r['Language'])).groupByKey().mapValues(list).\
    filter(lambda r: 'English' in r[1] and 'French' in r[1]).\
    map(lambda r: r[0]).collect()
print(result)