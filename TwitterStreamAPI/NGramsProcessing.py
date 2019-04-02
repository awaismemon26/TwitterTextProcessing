
import pandas as pd
import numpy as np
import spacy
import re
import string
import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#load reviews data
reviews = pd.read_csv('D:\\LearningDevelopmentProject\\PreprocessedData\\TwitterData.csv')
#extract only reviews
comments = reviews['tweets']
comments = comments.astype('str')

#function to remove non-ascii characters
def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)
#remove non-ascii characters
comments = comments.map(lambda x: _removeNonAscii(x))


#get stop words of all languages
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
#function to detect language based on # of stop words for particular language
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    lang = max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
    if lang == 'english':
        return True
    else:
        return False
#filter for only english comments
eng_comments=comments[comments.apply(get_language)]

#drop duplicates
eng_comments.drop_duplicates(inplace=True)

#load spacy
nlp = spacy.load('en')

#function to clean and lemmatize comments
def clean_comments(text):
    #remove punctuations
    regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
    nopunct = regex.sub(" ", str(text))
    #use spacy to lemmatize comments
    doc = nlp(nopunct, disable=['parser','ner'])
    lemma = [token.lemma_ for token in doc]
    return lemma

#apply function to clean and lemmatize comments
lemmatized = eng_comments.map(clean_comments)

#make sure to lowercase everything
lemmatized = lemmatized.map(lambda x: [word.lower() for word in x])

#turn all comments' tokens into one single list
unlist_comments = [item for items in lemmatized for item in items]

#print(unlist_comments)


bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()


bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(unlist_comments)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(unlist_comments)

#bigramFinder.apply_freq_filter(100)

bigram_freq = bigramFinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)



bigramFreqTable.head().reset_index(drop=True)


#get english stopwords
en_stopwords = set(stopwords.words('english'))


#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or '' in ngram or ' 'in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords:
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False


#filter bigrams
filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]


exportBigramFreqTable = filtered_bi.to_csv(r'D:\\LearningDevelopmentProject\\PreprocessedData\\TwitterDataBigrams.csv',index = None, header=True)
