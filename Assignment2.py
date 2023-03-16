import os
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re


def read_stop_words() -> list[str]:
    with open('Stopwords.txt', encoding="utf-8") as f:
        text = f.read()
        # print(text.split('\n'))
    return text.split('\n')


def read_files():
    files = os.listdir("corpus")
    corpus = [''] * len(files)
    for file in files:
        with open(os.path.join("corpus", file), encoding='utf-8') as f:
            corpus[int(file.split(".")[0][-1]) - 1] = f.read()
    return corpus


def remove_stopwords(words: list[str], stopwords: list[str]) -> list[str]:
    return [word for word in words if word not in stopwords]


def preprocess(corpus: list[str], stopwords: list[str]) -> dict:
    pre_processed_corpus = {}
    stemmer = PorterStemmer()
    for i, doc in enumerate(corpus):
        # Remove numbers
        doc = re.sub(r'\d+', '', doc)
        # Remove punctuation
        doc = re.sub(r'[^\w\s]', '', doc)
        # tokenize
        tokenized_doc = word_tokenize(doc)
        # remove stop words
        cleaned_doc = remove_stopwords(tokenized_doc, stopwords)
        # stemming
        stemmed_doc = [stemmer.stem(word) for word in cleaned_doc]
        print(i)
        print(stemmed_doc)
        pre_processed_corpus[i] = stemmed_doc

    return pre_processed_corpus


def TF():
    pass


def IDF():
    pass


def TF_IDF():
    pass


preprocess(read_files(), read_stop_words())
