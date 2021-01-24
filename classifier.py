#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CLASSIFIER

Created on Thu Jan 21 17:16:32 2021

@author: Katarina
"""
import nltk
import nltk.classify
import re
from pymystem3 import Mystem
import random
from wiki_ru_wordnet import WikiWordnet

mystem = Mystem()
wwnet = WikiWordnet()
ALL_FEATURES = {}

def load_all_data():
    #outputs a tuple consisting (trainingdata, testdata)
    #each of the elements of that tuple is a list of tuples (tweet, label)
    training_data = []
    test_data = []
    n_train = 0
    with open('imperfective_masked_cleaned_masterfile.txt', encoding='utf-8') as file:
        lines = ""
        for line in file:
            if re.search('@[a-zA-Z0-9_]+', line) is not None:
                line = re.sub('@[a-zA-Z0-9_]+', 'mention', line)
            lines = lines+line
        tweets = re.findall('<tweet>(.*)', lines)
        n_train = int(0.8*len(tweets))
        for tweet in tweets[:n_train]:
            tweet_with_label = (tweet, 'imperfective')
            training_data.append(tweet_with_label)
        for tweet in tweets[n_train:]:
            tweet_with_label = (tweet, 'imperfective')
            test_data.append(tweet_with_label)
            
    with open('perfective_masked_cleaned_masterfile.txt', encoding='utf-8') as file:
        lines = ""
        for line in file:
            if re.search('@[a-zA-Z0-9_]+', line) is not None:
                line = re.sub('@[a-zA-Z0-9_]+', 'mention', line)
            lines = lines+line
        tweets = re.findall('<tweet>(.*)', lines)
        for tweet in tweets[:n_train]:
            tweet_with_label = (tweet, 'perfective')
            training_data.append(tweet_with_label)
        for tweet in tweets[n_train:]:
            tweet_with_label = (tweet, 'perfective')
            test_data.append(tweet_with_label)
    
    data = (training_data, test_data)
    return(data)

def normalize_tweet(tweet):
    tweet = tweet.lower()
    words = re.split(' ', tweet)
    tweet = ""
    for word in words:
        if re.search('^([ха]{2,})$', word.lower()) is not None:
            if re.search('х', word.lower()) is not None:
                word = 'хаха'
            else:
                word = 'Ааа'
            tweet = tweet + " " + word
        else:
            tweet = tweet + " " + word
    return(tweet)

def get_char_n_grams(tweet, features):
    tweet = re.sub('targetverb', 'tv', tweet)
    longer_char_grams = True
    for i, char in enumerate(tweet):
        if i > 3:
            feature = "CHAR_4GRAM_" + tweet[i-3] + tweet[i-2] + tweet[i-1] + tweet[i]
            if feature in features:
                features[feature] += 1
            else:
                features[feature] = 1
            if i > 4:
                feature = "CHAR_5GRAM_" + tweet[i-4] + tweet[i-3] + tweet[i-2] + tweet[i-1] + tweet[i]
                if feature in features:
                    features[feature] += 1
                else:
                    features[feature] = 1
                if i > 5:
                    feature = "CHAR_6GRAM_" + tweet[i-5] + tweet[i-4] + tweet[i-3] + tweet[i-2] + tweet[i-1] + tweet[i]
                    if feature in features:
                        features[feature] += 1
                    else:
                        features[feature] = 1    
                    if longer_char_grams:
                        if i > 6:
                            feature = "CHAR_7GRAM_"+ tweet[i-6] + tweet[i-5] + tweet[i-4] + tweet[i-3] + tweet[i-2] + tweet[i-1] + tweet[i]
                            if feature in features:
                                features[feature] += 1
                            else:
                                features[feature] = 1
                            if i > 7:
                                feature = "CHAR_8GRAM_" + tweet[i-7] + tweet[i-6] + tweet[i-5] + tweet[i-4] + tweet[i-3] + tweet[i-2] + tweet[i-1] + tweet[i]
    return(features)

def get_n_grams(features, word_list, tag):
    for i, word in enumerate(word_list):
        unigram = tag+ "_UNI_" + word_list[i]
        if unigram in features:
            features[unigram] += 1
        else:
            features[unigram] = 1
        if i > 0:
            bigram = tag + "_BI_" + word_list[i-1] + "_" + word_list[i]
            if bigram in features:
                features[bigram] += 1
            else:
                features[bigram] = 1
            if i > 1:
                trigram = tag + "TRI_" + word_list[i-2] + "_" + word_list[i-1] + "_" + word_list[i]
                if trigram in features:
                    features[trigram] += 1
                else:
                    features[trigram] = 1
                skipgram = tag + "SKIP_" + word_list[i-2] + "_SKIP_" + word_list[i]
                if skipgram in features:
                    features[skipgram] += 1
                else:
                    features[trigram] = 1
    return(features)

def get_word_n_grams(tweet, features):
    words = re.split(' ', tweet)
    words_clean = []
    for word in words:
        word = word.lower()
        if re.search('targetverb', word) is None:
            word_letters = re.search('[а-яa-z]*', word)
            if word_letters is not None:
                word_letters = word_letters.group()
                words_clean.append(word_letters)
                word_letters = "UNI_" + word_letters
                if word_letters in features:
                    features[word_letters] += 1
                else:
                    features[word_letters] = 1
    features = get_n_grams(features, words_clean, "WORDS")
    return(features)

def get_morphological_n_grams(features, words_analysis):
    pos_tags = []
    for word in words_analysis:
        if len(word['analysis']) > 0:
            analysis = word['analysis'][0]
            if 'gr' in analysis:
                pos_tag = re.search('[A-Z]', analysis['gr'])
                pos_tags.append(pos_tag.group())
        elif 'text' in word:
            if re.search('target', word['text']) is None:
                pos_tags.append(word['text'])
                
    features = get_n_grams(features, pos_tags, "POS")
    return(features)

def get_morphological_features(full_analysis, features):
    words_analysis = []
    all_lemmas = []
    for word_analysis in full_analysis:
        if 'analysis' in word_analysis:
            analysis = word_analysis['analysis']
            words_analysis.append(word_analysis)
            if len(analysis) > 0:
                analysis = analysis[0]
                if 'lex' in analysis:
                    lemma = analysis['lex']
                    all_lemmas.append(lemma)
                if 'qual' in analysis:
                    feature = "QUAL_" + analysis['qual']
                    '''
                    if feature in features:
                        features[feature] += 1
                    else:
                        features[feature] = 1
                    '''
            elif 'text' in word_analysis:
                if re.search('target', word_analysis['text']) is None:
                    all_lemmas.append(word_analysis['text'])
    #features['TWEET_LENGTH'] = len(full_analysis)
    #features = get_morphological_n_grams(features, words_analysis)
    features = get_n_grams(features, all_lemmas, "LEM")
    return(features)

def get_verbs(annotation):
    verbs = []
    for word_analysis in annotation:
        if 'analysis' in word_analysis:
            analysis = word_analysis['analysis']
            if len(analysis) > 0:
                analysis = analysis[0]
                if 'gr' in analysis:
                    gr = analysis['gr']
                    if re.search('^V', gr) is not None:
                        verbs.append(gr)
                        #if re.search('несов', gr) is not None:
    return(verbs)

def add_other_verb_features(other_verbs, tag, features):
    feature = tag + "_VERBS"
    features[feature] = len(other_verbs)
    if len(other_verbs) > 0:
        n_perfective = 0
        for verb in other_verbs:
            if re.search('несов', verb) is not None:
                n_perfective += 1
                feature = tag + '_VERBS_PERF_PRESENT'
                features[feature] = 1
        feature = tag + '_PERF_VERB_RATIO'
        features[feature] = n_perfective / len(other_verbs)
        if len(other_verbs)-n_perfective > 0:
            feature = tag + '_NON_PERF_VERB_PRESENT'
            features[feature] = 1
    return features

def get_other_aspects(tweet, features):
    split_tweet = re.split('targetverb-[а-я]*', tweet.lower())
    if len(split_tweet) == 2:
        tweet_without_targetverb = split_tweet[0]+split_tweet[1]
        annotation = mystem.analyze(tweet_without_targetverb)
        other_verbs = get_verbs(annotation)
        features = add_other_verb_features(other_verbs, 'ALL_OTHER', features)
        
        annotation = mystem.analyze(split_tweet[0])
        verbs_left_context = get_verbs(annotation)
        features = add_other_verb_features(verbs_left_context, 'LEFT', features)
        
        annotation = mystem.analyze(split_tweet[1])
        verbs_right_context = get_verbs(annotation)
        features = add_other_verb_features(verbs_right_context, 'RIGHT', features)    
    return(features)

def get_wordnet_information(features, word, tag):
    synset = wwnet.get_synsets(word)
    if len(synset) > 0:
        synset = synset[0]
    else:
        feature = tag + 'NO_SYNSET_VERB'
        features[feature] = 1
        return(features)
    
    found_highest = False
    i = 0
    while not found_highest:
        hypernym = wwnet.get_synsets(synset)
        if len(hypernym) > 0:
            i += 1
            synset = hypernym[0].lemma()
        else:
            words = synset.get_words()
            for w in words:
                word = w
                break
            word = word.lemma()
            feature = tag + "LVL_" + str(i) + "_" + word
            if feature in features:
                features[feature] += 1
            else:
                features[feature] = 1
            found_highest = True            
    return(features)

def get_synsets_other_words(tweet, features):
    tweet = re.sub("targetverb-([А-Яа-я]*)", "", tweet)
    words_in_tweet = re.split(' ', tweet)
    for word in words_in_tweet:
        word = re.sub('\W', '', tweet)
        features = get_wordnet_information(features, word, 'WORDNET_WORD_')
    return(features)

def get_verb_semantics(tweet, features):
    if re.search("targetverb-([А-Яа-я]*)", tweet) is not None:
        verb = re.search("targetverb-([А-Яа-я]*)", tweet)
        verb = verb.group(1)
        feature = 'TARGETVERB_' + verb
        features[feature] = 1
        features['LENGTH_VERB'] = len(verb)
        verb_no_prefix = verb
        if re.search('^пере', verb) is not None:
            features['PREFIX_pere'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^по', verb) is not None:
            features['PREFIX_po'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^с', verb) is not None:
            features['PREFIX_s'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^за', verb) is not None:
            features['PREFIX_za'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^до', verb) is not None:
            features['PREFIX_do'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^на', verb) is not None:
            features['PREFIX_na'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^р[ао][зс]', verb) is not None:
            features['PREFIX_raz'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^без', verb) is not None:
            features['PREFIX_bez'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^о', verb) is not None:
            features['PREFIX_o'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^вы', verb) is not None:
            features['PREFIX_vy'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^в[цкнгшщхъфвпрлджчсмтьб]', verb) is not None:
            features['PREFIX_v'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^де', verb) is not None:
            features['PREFIX_de'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^при', verb) is not None:
            features['PREFIX_pri'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^под', verb) is not None:
            features['PREFIX_pod'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^у', verb) is not None:
            features['PREFIX_u'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^от', verb) is not None:
            features['PREFIX_ot'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        elif re.search('^недо', verb) is not None:
            features['PREFIX_nedo'] = 1
            verb_no_prefix = re.sub('^пере', '', verb)
        else:
            features['PREFIX_NONE'] = 1
        #check if there is a negation (very crudely)
        if re.search(' не', tweet) is not None:
            features['NEGATION'] = 1
        features = get_wordnet_information(features, verb, 'WORDNET_VERB_PREF_')
        if not verb_no_prefix == verb:
            features = get_wordnet_information(features, verb_no_prefix, 'WORDNET_VERB_NOPREF_')
        return(features)
    else:
        return(None)

def extract_features(tweet):
    #input: a tweet 
    #output: a dictionary {feature:count, feature2:count, feature3:count}
    features = {}
    tweet = normalize_tweet(tweet)
    features = get_char_n_grams(tweet, features)
    #features = get_word_n_grams(tweet, features)
    full_analysis = mystem.analyze(tweet)
    features = get_morphological_features(full_analysis, features)
    #features = get_other_aspects(tweet, features)
    #features = get_synsets_other_words(tweet, features)
    features = get_verb_semantics(tweet, features)
    if re.search("targetverb-([А-Яа-я]*)", tweet) is None:
        features = None
    if features is not None:
        for feature in features.keys():
            if feature in ALL_FEATURES:
                ALL_FEATURES[feature] += 1
            else:
                ALL_FEATURES[feature] = 1
    return(features)

def prune_features(features_training_data):
    print("Total amount of features extracted before pruning: ", len(ALL_FEATURES))
    deleted = 0
    for tweet in features_training_data:
        for feature in list(tweet[0].keys()):
            if(int(ALL_FEATURES[feature])) < 2:
                del tweet[0][feature]
                deleted += 1
                #del ALL_FEATURES[feature]
    print("Total amount of features left after removing single features: ", len(ALL_FEATURES)-deleted)
    return(features_training_data)

def print_fold_accuracy(fold_number, k_folds, n_correctly_classified, n_total_in_test):
    print("Fold number ", fold_number+1, "out of ", k_folds, " folds.")
    print("Number of tweets classified: ", n_total_in_test)
    print("Number of tweets correctly classified: ", n_correctly_classified)
    print("Accuracy: ", n_correctly_classified/n_total_in_test)
    return(n_correctly_classified/n_total_in_test)

def print_fold_precision(true_positives, false_positives):
    precision = true_positives / (true_positives + false_positives)
    print("Precision: ", precision)
    return(precision)

def print_fold_recall(true_positives, false_negatives):
    recall = true_positives / (true_positives + false_negatives)
    print("Recall: ", recall)
    return(recall)

def print_fold_F1(precision, recall):
    f1 = 2*((precision*recall) / (precision + recall))
    print("F1: ", f1, "\n\n\n\n")
    return(f1)

def print_averaged_results(cum_accuracy, k_folds, precision, recall, f1):
    average_accuracy = cum_accuracy/k_folds
    average_precision = precision/k_folds
    average_recall = recall/k_folds
    average_f1 = f1/k_folds
    print("Total number of folds classified: ", k_folds)
    print("Average accuracy over ", k_folds, " folds: ", average_accuracy)
    print("Average precision over ", k_folds, " folds: ", average_precision)
    print("Average recall over ", k_folds, " folds: ", average_recall)
    print("Average F1 over ", k_folds, " folds: ", average_f1)

def try_features(features):
    #takes a list of tuples (features, label) (each tuple represents a tweet) 
    #and uses k folds cross validation to calculate accuracy over
    k_folds = 10
    subset_size = int(len(features) / k_folds)
    cum_accuracy = 0
    cum_precision = 0
    cum_recall = 0
    cum_f1 = 0
    for i in range(k_folds):
        testing_this_round = features[(i*subset_size):((i+1)*subset_size)]
        training_this_round = features[:i*subset_size] + features[(i+1)*subset_size:]
        #train classifier on training list
        classifier = nltk.classify.NaiveBayesClassifier.train(training_this_round)
        #classify instances of testing list
        n_correctly_classified = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for tweet_tuple in testing_this_round:
            predicted = classifier.classify(tweet_tuple[0]) #tweet_tuple[0] contains the features for the tweet tuple
            if predicted == tweet_tuple[1]:
                n_correctly_classified += 1
            if predicted == 'imperfective':
                if tweet_tuple[1] == 'imperfective':
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if tweet_tuple[1] == 'imperfective':
                    false_negatives += 1
        #print results for this fold
        accuracy = print_fold_accuracy(i, k_folds, n_correctly_classified, len(testing_this_round))
        precision = print_fold_precision(true_positives, false_positives)
        recall = print_fold_recall(true_positives, false_negatives)
        f1 = print_fold_F1(precision, recall)
        cum_accuracy += accuracy 
        cum_precision += precision
        cum_recall += recall
        cum_f1 += f1
    print_averaged_results(cum_accuracy, k_folds, cum_precision, cum_recall, cum_f1)

def test_on_testset(test_data, train_data):
    features_test_data = []
    for instance in test_data:
        features = extract_features(instance[0])
        if features is not None:
            label = instance[1]
            features_test_data.append((features, label))
    random.shuffle(features_test_data)
    classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    n_correctly_classified = 0
    
    for tweet_tuple in features_test_data:
        predicted = classifier.classify(tweet_tuple[0])
        if predicted == tweet_tuple[1]:
            n_correctly_classified += 1
        if predicted == 'imperfective':
            if tweet_tuple[1] == 'imperfective':
                true_positives += 1
            else:
                false_positives += 1
        else:
            if tweet_tuple[1] == 'imperfective':
                false_negatives += 1
    
    accuracy = n_correctly_classified/len(features_test_data)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2*((precision*recall) / (precision + recall))
    
    print("100 most informative features:")
    classifier.show_most_informative_features(100)
    
    print("\n\nRESULTS CLASSIFICATION TEST SET\n\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

def main():
    print("This is the right version! only final feature groups included, no pruning")
    data = load_all_data()
    raw_training_data = data[0]
    features_training_data = []
    for i, instance in enumerate(raw_training_data):
        features = extract_features(instance[0])
        if features is not None:
            label = instance[1]
            features_training_data.append((features, label))
    random.shuffle(features_training_data)
    #features_training_data = prune_features(features_training_data)
    try_features(features_training_data)
    #test_data = data[1]
    #use_testset 
    #test_on_testset(data[1], features_training_data)
    #print most informative features
    '''
    classifier.show_most_informative_features(100)
    see classifier4.py (in SSI folder) for more
    '''
    
if __name__ == '__main__':
    main()