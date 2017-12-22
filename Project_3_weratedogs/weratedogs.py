#%%
import requests
import os
from bs4 import BeautifulSoup
import glob,csv
import pandas as pd

userid=os.getlogin()
os.chdir(r"c:\\Users\\"+userid+r"\\desktop\\Udacity-Data-Analyst-Nanodegree\\Project_3_weratedogs") 
#%%
#Gather 1
folder_name = 'images'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
twitter_archive=pd.read_csv('twitter-archive-enhanced.csv')
twitter_archive.head(2)
#file=".\image-predictions.tsv"
#
#with open(file) as fd:
#    rd = csv.reader(fd, delimiter="\t", quotechar='"')
#    for row in rd:
#        print(row)


#%%
#Gather 2

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)

with open(url.split('/')[-1], mode='wb') as file:
    file.write(response.content)
    
images = pd.read_csv('image-predictions.tsv', sep='\t')
images.head(2)

#%%
#Gather 3
import tweepy

consumer_key = 	'F0wLeqL2m6XSMl8DWU6S7mlJm'
consumer_secret = 'PSYpGQHbDcxWmHLQvp5XODKL4wUVdddvl00mL99C1wk58Pnau3'

access_token = '106335650-XVaCgWgxQwwjW0vZH5pAftO6CwKJr2Gz3824NAsI'
access_token_secret = 'usQWjVz2pujFZFjPJdIDCPB6lXTApYRbmLZjAOAVjvKEy'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

ublic_tweets = api.home_timeline()  

for tweet in ublic_tweets:  

   print (tweet.text)
   
   
#%%
tweet_list=[]
tweet_ids = images['tweet_id']
for tweet_id in tweet_ids:
   tweet_status=api.get_status(tweet_id)._json
   favorite_count = tweet_status['favorite_count']
   retweet_count = tweet_status['retweet_count']
   tweet_list.append({'tweet_id': int(tweet_id),
                        'favorites': int(favorite_count),
                        'retweet_count': int(retweet_count)})
   

RateLimitError: [{'code': 88, 'message': 'Rate limit exceeded'}]