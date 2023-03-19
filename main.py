import os
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import math


def read_stop_words() -> list[str]:
    with open('Stopwords.txt', encoding="utf-8") as f:
        text = f.read()
        # print(text.split('\n'))
    return text.split('\n')


def read_files(directory):
    files = os.listdir(directory)
    corpus = [''] * len(files)
    for file in files:
        with open(os.path.join(directory, file), encoding='utf-8') as f:
            corpus[int(file.split(".")[0][-1]) - 1] = f.read()
    return corpus


def remove_stopwords(words: list[str], stopwords: list[str]) -> list[str]:
    return [word for word in words if word not in stopwords]


def preprocess(corpus: list[str], stopwords: list[str]) -> list:
    pre_processed_corpus = [0] * (len(corpus))
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
        # print(i)
        # print(stemmed_doc)
        pre_processed_corpus[i] = stemmed_doc

    return pre_processed_corpus


# weighted term frequency
def TF(word: str, doc: list[str]) -> float:
    tf = doc.count(word)
    wtf = 0 if tf == 0 else (1 + math.log10(tf))
    return wtf


# weighted Inverse document frequency
def IDF(word: str, corpus: list) -> float:
    n = len(corpus)
    df = 0
    for doc in corpus:
        if word in doc:
            df += 1
    idf = math.log10(n / df)
    return idf


def cosine_normalize(tfs: list) -> list:
    magnitude = math.sqrt(sum([num * num for num in tfs]))
    c_normalized = [(num / magnitude) for num in tfs]
    return c_normalized


def TFIDF_COS(corpus):
    vocabulary = list(set([word for doc in corpus for word in doc]))
    tfidf = {}
    cosine = {}
    for word in vocabulary:
        tfidf_word = []
        tfs = []
        for doc in corpus:
            tf = TF(word, doc)
            tfs.append(tf)  # term frequency to calculate cosine similarity later
            tfidf_word.append(tf * IDF(word, corpus))  # tfidf score
        tfidf[word] = tfidf_word
        cosine[word] = cosine_normalize(tfs)
    return tfidf, cosine


def Print_Result(result: list[tuple], query_num):
    print(''.join(test_data[query_num]))

    print("about ", len([score for score in result if score[1] != 0]), " results :")
    for res in result:
        if res[1] != 0:
            print("document", res[0] + 1)
    print('-' * 50)


def Search(query_docs, table, num_docs):
    i = 0
    for query in query_docs:
        similarity = [0] * num_docs
        # get vocab of query document
        vocabulary = list(set(query))
        for word in vocabulary:
            if word in table:
                similarity = [sum(x) for x in zip(similarity, table[word])]

        sorted_scores = [(i, score) for (i, score) in enumerate(similarity)]
        sorted_scores.sort(key=lambda x: x[1], reverse=True)
        Print_Result(sorted_scores, i)
        i += 1


if __name__ == '__main__':
    # preprocess and calculation phase
    Num_docs = len(os.listdir('corpus'))
    data = preprocess(read_files('corpus'), read_stop_words())
    test_data = read_files('test')
    queries = preprocess(test_data, read_stop_words())
    tfidf_table, cosine_table = TFIDF_COS(data)

    choice = input('1-TFIDF\n2-Cosine Similarity\n')
    if choice == '1':
        Search(queries, tfidf_table, Num_docs)
    else:
        Search(queries, cosine_table, Num_docs)
