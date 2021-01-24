#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:15:35 2021

@author: Katarina
"""

import twitter
import time
import re

api = twitter.Api(consumer_key="KEY",
                  consumer_secret="SECRET_KEY",
                  access_token_key="KEY",
                  access_token_secret="SECRET_KEY")

collected_tweets = []
masterfile = open('masterfile.txt', mode='w')
masterfile.write('<MASTERFILE>')
tweet_count = 0
call_count = 0

with open('queries.txt', mode='r') as queryfile:
    for query in queryfile:
        if tweet_count > 150000:
            break
        elif tweet_count%1000 == 0:
            print("Amount of tweets found: ", tweet_count)
        call_count += 1
        if call_count%179 == 0:
            time.sleep(1000)
        query = query.strip()
        tweets = api.GetSearch(term=query, count=100, lang='ru', result_type='recent')
        collected_tweets.extend(tweets)
        for tweet in tweets:
            masterfile.write('<tweet>')
            masterfile.write(tweet.text)
            masterfile.write('</tweet>\n')
            tweet_count += 1
            if re.search('[А-Яа-я]+', str(tweet.text)) is not None:
                words_in_tweet = re.split(' ', str(tweet.text))
                for word in words_in_tweet:
                    word = re.search('[А-Яа-я]+', word)
                    if word is not None:
                        word = word.group()
                        word = word.lower()
                        call_count += 1
                        if call_count%179 == 0:
                            time.sleep(1000)
                        try:
                            new_tweets = api.GetSearch(term=word, count=100, lang='ru', result_type='recent')
                            for new_tweet in new_tweets:
                                masterfile.write('<tweet>')
                                masterfile.write(new_tweet.text)
                                masterfile.write('</tweet>\n')
                                tweet_count += 1
                                if tweet_count%1000 == 0:
                                    print("Amount of tweets found: ", tweet_count)
                        except:
                            print("an error occurred. Amount of tweets currently collected: ", tweet_count)
#next time: keep track of which words I already used as queries to avoid duplicates
masterfile.write('</MASTERFILE>')
masterfile.close()
print("Amount of tweets collected:", tweet_count, "\nAmount of calls made to the Twitter API:", call_count)
#print(collected_tweets)
        