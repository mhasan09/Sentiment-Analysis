from textblob import TextBlob
import tweepy
key= 'imF8YpanOMKifjbGw7MoCfB7L'
secret= '35nTQBo2uLyyPtKlKy34FAHRrVN8RTRKqz0RDPD8YH4E9AEgZy'
access_token = '1114391929440165888-A3medjceBaPcnXouLGZn5NKJmrjLVF'
access_token_secret = 'QLbIXKQnQS4oOxPq9386SbenAz0BReCbvtITRUgGYYQ2S'
auth= tweepy.OAuthHandler(key,secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)
tweets=api.search('endgame')
for i in tweets:
    print(i.text)
    analysis = TextBlob(i.text)
    print(analysis.sentiment)