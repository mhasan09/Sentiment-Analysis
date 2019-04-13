from textblob import TextBlob
import tweepy
key= ''
secret= ''
access_token = ''
access_token_secret = ''
auth= tweepy.OAuthHandler(key,secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)
tweets=api.search('endgame')
for i in tweets:
    print(i.text)
    analysis = TextBlob(i.text)
    print(analysis.sentiment)
