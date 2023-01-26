from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .master("local[1]") \
    .appName("hw5") \
    .getOrCreate()

# 2a
import pyspark.sql.functions as fc

countryLang = spark.read.json('countrylanguage.json')
countryLang.filter('IsOfficial == "T"').groupBy('Language').agg(fc.count('*').alias('cnt')).orderBy('cnt', ascending=False).limit(10).show(truncate=False)

# 2b

country = spark.read.json('country.json')
city = spark.read.json('city.json')
country.filter('Continent == "North America" and GNP >= 100000').select('Capital', country.Name.alias('CountryName')).join(city.select('ID', city.Name.alias('CapitalName')), country.Capital == city.ID).select('CountryName', 'CapitalName').show(truncate=False)


# 2c

country = spark.read.json('country.json')
countryLang = spark.read.json('countrylanguage.json')
country.join(countryLang, (country.Continent == 'North America') & (country.Code == countryLang.CountryCode) & (countryLang.Language == 'English') & (countryLang.IsOfficial == 'T')).select('Name').show(truncate=False)

# 2d
import pyspark.sql.functions as fc

city = spark.read.json('city.json')
city.filter('CountryCode == "USA"').agg(fc.max('Population')).show(truncate=False)

# 2e

countryLang = spark.read.json('countrylanguage.json')
countryLang.filter('Language == "English" and IsOfficial == "T"').select('CountryCode').intersect(countryLang.filter('Language == "French" and IsOfficial == "T"').select('CountryCode')).show(truncate=False)

