#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 09:51:37 2018
@author: thabata
"""

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from xml.dom.minidom import parse
import logging
import time

def query_processor(path, config_vector):
    """
calls functions to read queries and outputs
    """

    begin = time.time()
    
    #instantiate logging 
    global logger_global
    log_path = path+'/4-QUERY_PROCESSOR/qp.log'
    log('query_processor', log_path)
    logger_global = logging.getLogger('query_processor')
    logger_global.info('Processing Queries Module...')

    queries_array = []
    queries_keys = []
    
    #stop words
    stop_words = set(stopwords.words('english'))
    
    use_mode = config_vector[0][1]
    for config in config_vector:
        if str(config[0]) == 'LEIA':            
            partial_qus_array, partial_qus_keys, results = \
                                    read_XML(path+str(config[1]).strip())
            
            logger_global.info('Tokenizing documents...')
            if use_mode == 'STEMMER':
                stemmer = PorterStemmer()
                queries_array += tokenizer(partial_qus_array,
                                              stop_words,
                                              stemmer)
                queries_keys += partial_qus_keys
            elif use_mode == 'NOSTEMMER':
                queries_array += tokenizer(partial_qus_array,
                                              stop_words,
                                              None)
                queries_keys += partial_qus_keys                
            else: print("ERROR: Use mode undefined.")
            
        elif str(config[0]) == 'CONSULTAS':
            outfile_queries = path+config[1].strip()
        
        elif str(config[0]) == 'ESPERADOS':
            outfile_expected_results = path+config[1].strip()
    
    if not (outfile_queries or outfile_expected_results):
        outfile_queries = path+'/4-QUERY_PROCESSOR/queries_out.csv'
        outfile_expected_results = path+ \
                            '/4-QUERY_PROCESSOR/expected_results_out.csv'
        logger_global.warning('Outfiles not specified. '
                             'Applying default ones:'
                             ' '+outfile_queries+','
                             ' '+outfile_expected_results)

    end = time.time() - begin
    start = time.time()
    logger_global.info('All %s queries read and tokenized successfully '
                      'in %s s' % (str(len(queries_array)), str(end)))
    
    logger_global.info('Writing Queries on file...')
    write_csv(outfile_queries, queries_keys, queries_array)
    
    logger_global.info('Writing Expected Results on file...')
    write_csv(outfile_expected_results, queries_keys, results)
    
    end = time.time() - start
    logger_global.info('Write operation finished with %s s' % str(end))
    
    end = time.time() - begin
    logger_global.info('End of Query Processor Module. '
                   'Total of %s elapsed.' % str(end))



def read_XML(filename):
    """
reads data from xml files
returns two arrays: queries, their keys and array of results arrays
The results arrays are arrays of tuples: (doc number, votes)
    """
    logger_global.info('Reading '+filename+' file...')
    init = time.time()
    
    queries = []
    queries_keys = []
    results = []
    dom_tree = parse(filename)
    query_xml = dom_tree.documentElement.getElementsByTagName("QUERY")
    
    for qu in query_xml:
        qu_num = qu.getElementsByTagName('QueryNumber')[0].childNodes[0].data
        queries.append(
                qu.getElementsByTagName('QueryText')[0].childNodes[0].data
                )
        queries_keys.append(qu_num)
        
        records = qu.getElementsByTagName('Records')[0]
        items = records.getElementsByTagName('Item')
        result = []
        for item in items:
            score = item.getAttribute('score')
            doc_num = item.childNodes[0].data
            result.append((doc_num, score))
        results.append(result)
    
    finish = time.time() - init
    logger_global.info('%s queries read succesfully in %s s.' % 
                          (str(len(queries)), str(finish)))
    return queries, queries_keys, results



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



def tokenizer(docs, stop, stemmer):
    """
tokenize documents and pre-process
- with stemmer (or not)
- removes small words (1-2 chars)
- removes numbers
returns array of arrays (list of tokens in each document)
    """
    init = time.time()
    import re
    regex = re.compile('^\d*[.,]?\d*$')
    
    tok_docs = []
    for doc in docs:
        toks = [stemmer.stem(tok) for tok in word_tokenize(doc)
                                        if not tok in stop
                                        if len(tok)>2
                                        if regex.match(tok)==None]
        toks = [tok.upper() for tok in toks]
        tok_docs.append(toks)

    finish = time.time() - init
    logger_global.info('%s records tokenized succesfully in %s s.' % 
                          (str(len(docs)), str(finish)))

    return tok_docs



def write_csv(filepath, csv_keys, csv_values):
    """
writes csv file
    """
    f = open(filepath, 'w+')
    for aux in range(len(csv_keys)):
        f.write(csv_keys[aux]+";%s\n" % csv_values[aux])     
    f.close()



if __name__ == '__main__':

    import os
    PATH = os.path.dirname(os.path.abspath(__file__))

    config_file = '/4-QUERY_PROCESSOR/qp.cfg'
    
    with open(PATH+config_file.strip(), 'r') as configuration:
        config_vector=[]
        for line in configuration:
            line = line.strip()
            config_vector.append(line.split('='))

    query_processor(PATH, config_vector)