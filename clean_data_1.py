#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:12:29 2021

@author: Katarina
"""

#input: masterfile with tweets
#output: masterfile with useless tweets removed/other tweets cleaned up

import xml.etree.ElementTree as ET
import re
from pymystem3 import Mystem

INPUTFILE = open('masterfile1.txt', mode='r')
IMPERFECTIVE_OUTPUTFILE = open('imperfective_masked_cleaned_masterfile.txt', mode='w')
PERFECTIVE_OUTPUTFILE = open('perfective_masked_cleaned_masterfile.txt', mode='w')
GARBAGEFILE = open('unprocessed_tweets.txt', mode='w')
mystem = Mystem()

def extract_tweets():
    tweets = []
    tree = ET.parse(INPUTFILE)
    root = tree.getroot()
    tweet_elements = [tweet for tweet in root.iter('tweet')]
    tweets = [tweet.text for tweet in tweet_elements]
    return(tweets)
    
def remove_duplicates(tweets):
    tweets = list(set(tweets))
    return(tweets)

def mask_mentions_and_remove_short_tweets(tweets):
    for index, old_tweet in enumerate(tweets):
        tweet = re.sub('[<|>]', '\"', old_tweet)
        words = re.split('\s', tweet)
        length = 0
        for word_position, word in enumerate(words):
            if re.search('@[A-Za-z0-9_]', word) is not None:
                words[word_position] = 'MENTION'
            elif re.search('^#', word) is not None:
                pass
            elif re.search('http', word) is not None:
                words[word_position] = 'URL'
            elif re.search('^RT$', word) is not None:
                pass
            else:
                length += 1
        if length < 3:
            GARBAGEFILE.write(tweet)
            GARBAGEFILE.write("\n\n\n")
            tweets.remove(old_tweet)
        else:
            space = ' '
            tweet = space.join(words)
            tweets[index] = tweet
    return(tweets)

def delete_tweets_without_verbs(tweets):
    for tweet in tweets:
        verb_count = 0
        try:
            #lemmas = mystem.lemmatize(tweet)
            full_analysis = mystem.analyze(tweet)
            for word_information in full_analysis:
                #word = word_information['text']
                if 'analysis' in word_information:
                    analysis = word_information['analysis']
                    if(len(analysis) > 0):
                        analysis = analysis[0]
                        if 'gr' in analysis:
                            pos_tag = analysis['gr']
                            if re.search('^V', pos_tag) is not None:
                                verb_count += 1
        except:
            print("an error occurred. This tweet could not be parsed adequately.")
        if verb_count < 1:
            GARBAGEFILE.write(tweet)
            GARBAGEFILE.write("\n\n\n")
            tweets.remove(tweet)
    return(tweets)
    
def mask_verb(word_information):
    analysis = word_information['analysis'][0]
    lemma = analysis['lex']
    masked_verb = 'TARGETVERB-' + lemma
    return(masked_verb)
    
def get_aspect(word_information):
    analysis = word_information['analysis'][0]
    morf_razbor = analysis['gr']
    if re.search('несов', morf_razbor) is not None:
        return(False)
    else:
        return(True)

def mask_tweets(tweets):
    imperfective_tweets = []
    perfective_tweets = []
    for original_tweet in tweets:
        verb_index = 0
        new_tweet = ""
        verb = ""
        try:
            full_analysis = mystem.analyze(original_tweet)
            for index, word_information in enumerate(full_analysis):
                if 'analysis' in word_information:
                    analysis = word_information['analysis']
                    if(len(analysis) > 0):
                        analysis = analysis[0]
                        if 'gr' in analysis:
                            if 'lex' in analysis:
                                if re.search('блять', analysis['lex']) is None:
                                    pos_tag = analysis['gr']
                                    if re.search('^V', pos_tag) is not None:
                                        if verb_index < 3: #take the second or only verb
                                            verb_index = index
            if verb_index > 0:
                imperfective = False
                for index, word_information in enumerate(full_analysis):
                    if index == verb_index:
                        verb = mask_verb(word_information)
                        imperfective = get_aspect(word_information)
                        new_tweet = new_tweet + verb
                    else:
                        if 'text' in word_information:
                            word = word_information['text']
                            new_tweet = new_tweet + word
                if imperfective:
                    imperfective_tweets.append(new_tweet)
                else:
                    perfective_tweets.append(new_tweet)
        except:
            print("An error occurred. The tweet could not be parsed adequately.")
            tweets.remove(original_tweet)
        
    tweet_lists = [imperfective_tweets, perfective_tweets]
    return(tweet_lists)

def write_to_masterfile_and_print(tweet_lists):
    imperfective_tweets = tweet_lists[0]
    perfective_tweets = tweet_lists[1]
    IMPERFECTIVE_OUTPUTFILE.write('<IMPERFECTIVE_MASTERFILE>')
    for tweet in imperfective_tweets:
        tweet = '<tweet>' + tweet + '</tweet>\n'
        IMPERFECTIVE_OUTPUTFILE.write(tweet)
    IMPERFECTIVE_OUTPUTFILE.write('</IMPERFECTIVE_MASTERFILE>')
    PERFECTIVE_OUTPUTFILE.write('<PERFECTIVE_MASTERFILE>')
    for tweet in perfective_tweets:
        tweet = '<tweet>' + tweet + '</tweet>\n'
        PERFECTIVE_OUTPUTFILE.write(tweet)
    PERFECTIVE_OUTPUTFILE.write('</PERFECTIVE_MASTERFILE>')
    print("Total amount of tweets: ", len(imperfective_tweets)+len(perfective_tweets))
    print("Imperfective tweets: ", len(imperfective_tweets))
    print("Perfective tweets: ", len(perfective_tweets))

def main():
    tweets = extract_tweets()
    print("Total amount of tweets before cleanup: ", len(tweets))
    tweets = remove_duplicates(tweets)
    print("Total amount of tweets after removing duplicates: ", len(tweets))
    tweets = mask_mentions_and_remove_short_tweets(tweets)
    print("Total amount of tweets after removing short tweets: ", len(tweets))
    tweets = delete_tweets_without_verbs(tweets)
    print("Total amount of tweets after removing tweets without verbs: ", len(tweets))
    tweet_lists = mask_tweets(tweets)
    write_to_masterfile_and_print(tweet_lists)
    INPUTFILE.close()
    IMPERFECTIVE_OUTPUTFILE.close()
    PERFECTIVE_OUTPUTFILE.close()
    GARBAGEFILE.close()

if __name__ == '__main__':
    main()
