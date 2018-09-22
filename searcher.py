#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:54:15 2018
@author: thabata
"""
import csv
from math import sqrt
import logging
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def searcher(path, config_vector):
    """
calls functions to read inverted index and creates TF.IDF matrix
    """

    begin = time.time()
    
    #instantiate logging 
    global logger_global
    log_path = path+'/5-SEARCHER/se.log'
    log('searcher', log_path)
    logger_global = logging.getLogger('searcher')
    logger_global.info('Processing Searcher Module...')

    for config in config_vector:
        if str(config[0]) == 'MODELO':            
            doc_tfidf = read_TFIDF(path+config[1].strip())
        
        elif str(config[0]) == 'CONSULTAS':            
            queries = read_QUERIES(path+config[1].strip())
            
        elif str(config[0]) == 'RESULTADOS':
            outfile_results = path+config[1]
    
    if not outfile_results:
        outfile_results = path+'/5-SEARCHER/results.csv'
        logger_global.warning('Outfile not specified. '
                             'Applying default: '+outfile_results)

    end = time.time() - begin
    logger_global.info('Documents and queries read succesfully '
                      'in %s s' % str(end))
    
    similarities = cosine_similarity(doc_tfidf,
                                     calculate_queries_tfidf(queries))
    write_results(outfile_results, similarities)

    end = time.time() - begin
    logger_global.info('End of Searcher Module. '
                   'Total of %s elapsed.' % str(end))



def read_TFIDF(filename):
    """
reads tfidf matrix from csv file
returns TFIDF array of dictionaries:
    [ {tfidf['doc keys'], tfidf[token1], tfidf[token2], ...} , {}, ... ]
    """
    import ast
    logger_global.info('Reading '+filename+' file...')
    init = time.time()
    tfidf = []
    
    with open(filename,"r") as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=';'))
        for row in csv_reader:
            tfidf_dict = {}
            tfidf_dict['doc keys'] = row[0]
            for pair in ast.literal_eval(row[1]):
                tfidf_dict[pair[0]] = pair[1]
            tfidf.append(tfidf_dict)
    logger_global.info('TFIDF matrix read in %s s' % str(time.time()-init))
    return tfidf



def read_QUERIES(filename):
    """
reads queries from csv file
returns dictionary: queries[query key] = array of tokenized query
    """
    import ast
    logger_global.info('Reading queries file...')
    init = time.time()
    queries = {}
    
    with open(filename,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            queries[row[0]] = ast.literal_eval(row[1])

    logger_global.info('Queries file read in %s s' % str(time.time()-init))
    return queries



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



def calculate_queries_tfidf(queries):
    """
calculates array of TF*IDF matrices for queries
assumes TF*IDF = 1 for all words in query
returns array of dictionaries:
    [ {tfidf['qu keys'], tfidf[token1], tfidf[token2], ...} , {}, ... ]
    """
    init = time.time()
    logger_global.info('Calculating TF*IDF for queries...')
    
    queries_tfidf = []
    for query_key, query_array in queries.items():
        query_tfidf = {}
        query_tfidf['qu keys'] = query_key
        for token in query_array:
            query_tfidf[token] = 1.
        queries_tfidf.append(query_tfidf)
        
    end = time.time() - init
    logger_global.info('Queries TF*IDF completed in %s s' % str(end))
    
    return queries_tfidf



def denominator(document, query):
    """
calculates the norms of doc and query tfidf's
returns the denominator for the cosine similarity equation
    """
    document_norm = 0.
    for key, item in document.items():
        if key != 'doc keys': document_norm += float(item) * float(item)
    document_norm = sqrt(document_norm)
    
    query_norm = 0.
    for key, item in query.items():
        if key in list(document.keys()):
            query_norm += float(item) * float(item)
    query_norm = sqrt(query_norm)
    
    return document_norm * query_norm



def cosine_similarity_minion(doc_tfidf, query):
    """
actually calculates cosine similarity
returns array of ordered tuples: (rank, doc number, similarity)
    """
    init = time.time()
    logger_global.info('... for query '+query['qu keys'])
    similarity = {}
    no_match = []
    for document in doc_tfidf:
        doc_num = document['doc keys']
        numerator = 0.
        for token, _ in document.items():
            try: numerator += float(document[token]) * query[token]
            except KeyError: pass
        
        try: similarity[doc_num] = numerator/denominator(document, query)
        except ZeroDivisionError:
            similarity[doc_num] = 0.
            no_match.append(document['doc keys'])
    
    logger_global.warning('... has nothing in common with ' + \
                          str(len(no_match)) + ' documents')
     
    ranked_similarity = sorted(similarity.items(),
                               key = lambda x: x[1],
                               reverse=True)
    
    result = []
    for rank in range(len(ranked_similarity)):
        result.append((rank+1,
                       ranked_similarity[rank][0],
                       ranked_similarity[rank][1]))
        
    logger_global.info('finished in %s s' % str(time.time() - init))
    return result

    

def cosine_similarity(doc_tfidf, qu_tfidf):
    """
organizes the cosine similarity calculation for all queries
returns array of array of ordered tuples:
    [ [ query_key, [(rank, doc, simil), ...] ], ... ]    
    """
    init = time.time()
    logger_global.info('Calculating similarity...')
    
    similarities = []
    for query in qu_tfidf:
        similarities.append((query['qu keys'],
                             cosine_similarity_minion(doc_tfidf, query)))
        
    end = time.time() - init
    logger_global.info('Similarity calculation completed in %s s' % str(end))
    
    return similarities
    


def write_results(filepath, results):
    """
write tfidf in csv file
    """
    init = time.time()
    logger_global.info('Writing results on file...')
    
    with open(filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        for row in results:
            csv_writer.writerow(row)
    
    end = time.time() - init
    logger_global.info('Write operation finished with %s s' % str(end))



if __name__ == '__main__':

    import os
    PATH = os.path.dirname(os.path.abspath(__file__))

    config_file = '/5-SEARCHER/se.cfg'
    
    with open(PATH+config_file.strip(), 'r') as configuration:
        config_vector=[]
        for line in configuration:
            line = line.strip()
            config_vector.append(line.split('='))

    searcher(PATH, config_vector)