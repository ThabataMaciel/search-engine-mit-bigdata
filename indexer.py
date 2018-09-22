#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:54:15 2018
@author: thabata
"""
import csv
from math import log10
from nltk.probability import FreqDist
import logging
import time

def indexer(path, config_vector):
    """
calls functions to read inverted index and creates TF.IDF matrix
    """
    begin = time.time()
    global logger_global
    log_path = path+'/3-INDEXER/ix.log'
    log('indexer', log_path)
    logger_global = logging.getLogger('indexer')
    logger_global.info('Processing Indexer Module...')

    outfile_indexer = 0
    for config in config_vector:
        if str(config[0]) == 'LEIA':            
            inv_ix = read_CSV(path+config[1].strip())
            
        elif str(config[0]) == 'ESCREVA':
            outfile_indexer = path+config[1]
    
    if not outfile_indexer:
        outfile_indexer = path+'/3-INDEXER/indexer_out.csv'
        logger_global.warning('Log file for Indexer not specified. '
                             'Applying default: '+outfile_indexer)

    end = time.time() - begin
    logger_global.info('Inverted index read succesfully '
                      'in %s s' % str(end))

    write_tfidf(outfile_indexer, inv_ix, *TFIDF(inv_ix))

    end = time.time() - begin
    logger_global.info('End of Inverted Index Generator Module. '
                   'Total of %s elapsed.' % str(end))



def read_CSV(filename):
    """
reads inverted index from csv file
returns dictionary: inv_ix[token] = array of documents containing token
    """
    import ast
    logger_global.info('Reading '+filename+' file...')
    inv_ix = {}
    
    with open(filename,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            inv_ix[row[0]] = ast.literal_eval(row[1])
    
    return inv_ix



def log(name, log_file):
    """
instantiates the logging
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(streamHandler)



def get_doc_list(inv_ix):
    """
get document keys list from inverted index
returns array
    """
    doc_list = []
    for _, partial_list in inv_ix.items():
        doc_list += partial_list
    doc_list = list(set(doc_list))
    return doc_list



def TFIDF(inv_ix):
    """
calculates TF*IDF for documents and outsources definition of document list
returns nested dictionaries: tfidf[token][document key]
returns array of documents keys
    """
    init = time.time()
    logger_global.info('Calculating TF*IDF...')
    
    doc_list = get_doc_list(inv_ix)
    
    idf = IDF(inv_ix, len(doc_list))
    tf = TF(inv_ix)
    tfidf = {}
    for token, keys in tf.items():
        tfidf[token] = {}
        for key in keys:
            tfidf[token][key] = tf[token][key] * idf[token]

    end = time.time() - init
    logger_global.info('Indexer operation finished in %s s' % str(end))
    
    return tfidf, doc_list



def TF(inv_ix):
    """
calculates TF for documents and tokens
returns nested dictionary: tf[token][document key]
TF(term,doc) = frequency of term in doc / total terms in doc
    """
    tf = {}
    maxf = 0
    for token, list_keys in inv_ix.items():
        tf[token] = {}
        for key in list_keys:
            tf[token][key] = inv_ix[token].count(key)
            maxf = tf[token][key] if tf[token][key] > maxf else maxf
    
    for token, _ in tf.items():
        for key, _ in tf[token].items():
            tf[token][key] /= maxf
    return tf



def IDF(inv_ix, how_many_docs):
    """
calculates IDF for tokens in inverted index
returns dictionary: idf[token] = value
IDF(term) = log(number of docs / number of docs with the term)
    """
    idf = {}
    for token, list_keys in inv_ix.items():
        idf[token] = log10(how_many_docs/len(set(list_keys)))
    
    return idf



def write_tfidf(filepath, inv_ix, tfidf, doc_list):
    """
write tfidf in csv file
    """
    init = time.time()
    logger_global.info('Writing TF*IDF on file...')
    
    # transposing tfidf nested dictionaries for printing
    tfidf_t = \
        {d : {k:tfidf[k][d] for k in tfidf if d in tfidf[k]} for d in doc_list}
    
    with open(filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        #csv_writer.writerow(['doc keys']+
        #                    [token for token, _ in inv_ix.items()])
        for key in doc_list:
            csv_writer.writerow([key]+
             [[(token, tfidf_t[key][token]) for token in tfidf_t[key].keys()]])
    
    end = time.time() - init
    logger_global.info('Write operation finished in %s s' % str(end))



if __name__ == '__main__':

    import os
    PATH = os.path.dirname(os.path.abspath(__file__))

    config_file = '/3-INDEXER/ix.cfg'
    
    with open(PATH+config_file.strip(), 'r') as configuration:
        config_vector=[]
        for line in configuration:
            line = line.strip()
            config_vector.append(line.split('='))

    indexer(PATH, config_vector)