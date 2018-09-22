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

def inverted_index_generator(path, config_vector):
    """
calls functions to read data and create inverted index
    """

    begin = time.time()
    global logger_global
    log_path = path+'/2-INVERTED_INDEX/iv.log'
    log('inverted_index_generator', log_path)
    logger_global = logging.getLogger('inverted_index_generator')
    logger_global.info('Processing Inverted Index Generator Module...')

    docs_array = []
    docs_keys = []
    
    stop_words = set(stopwords.words('english'))
    
    use_mode = config_vector[0][1]
    outfile_inverted_index = 0
    for config in config_vector:
        if str(config[0]) == 'LEIA':            
            partial_docs_array, partial_docs_keys = \
                                    read_XML(path+str(config[1]).strip())
            
            logger_global.info('Tokenizing documents...')
            if use_mode == 'STEMMER':
                stemmer = PorterStemmer()
                docs_array += tokenizer(partial_docs_array,
                                              stop_words,
                                              stemmer)
                docs_keys += partial_docs_keys
            elif use_mode == 'NOSTEMMER':
                docs_array += tokenizer(partial_docs_array,
                                              stop_words,
                                              None)
                docs_keys += partial_docs_keys                
            else: print("ERROR: Use mode undefined.")
            
        elif str(config[0]) == 'ESCREVA':
            outfile_inverted_index = path+config[1].strip()
    
    if not outfile_inverted_index:
        outfile_inverted_index = path+'/2-INVERTED_INDEX/inverted_index_out.csv'
        logger_global.warning('Log file for Inverted Index not specified. '
                             'Applying default: '+outfile_inverted_index)

    end = time.time() - begin
    start = time.time()
    logger_global.info('All %s documents read and tokenized successfully '
                      'in %s s' % (str(len(docs_array)), str(end)))
    
    write_inverted_index(outfile_inverted_index,
                         inverted_index_minion(docs_array, docs_keys),
                         docs_keys)
    
    end = time.time() - start
    logger_global.info('Write operation finished with %s s' % str(end))
    
    end = time.time() - begin
    logger_global.info('End of Inverted Index Generator Module. '
                   'Total of %s elapsed.' % str(end))



def read_XML(filename):
    """
reads data from xml files
returns two arrays: documents and their keys
    """
    logger_global.info('Reading '+filename+' file...')
    init = time.time()
    
    docs = []
    docs_keys = []
    dom_tree = parse(filename)
    records = dom_tree.documentElement.getElementsByTagName("RECORD")
    
    for rec in records:
        rec_num = rec.getElementsByTagName('RECORDNUM')[0].childNodes[0].data
        try:
            docs.append(
                rec.getElementsByTagName('ABSTRACT')[0].childNodes[0].data
                )
            docs_keys.append(rec_num)
            
        except IndexError:
            try:
                docs.append(
                    rec.getElementsByTagName('EXTRACT')[0].childNodes[0].data
                    )
                docs_keys.append(rec_num)
                    
            except IndexError:
                logger_global.warning(
                        "Document["+rec_num+"] \
                        doesn't have abstract neither extract!")
    
    finish = time.time() - init
    logger_global.info('%s records read succesfully in %s s.' % 
                          (str(len(docs)), str(finish)))
    return docs, docs_keys



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
        tok_docs.append(toks)

    finish = time.time() - init
    logger_global.info('%s records tokenized succesfully in %s s.' % 
                          (str(len(docs)), str(finish)))

    return tok_docs



def inverted_index_minion(docs, docs_keys):
    """
actually creates the inverted index (the minion always does all the work)
returns dictionary - keys as index for doc_num array; values as token arrays
    """
    logger_global.info('Making inverted index...')
    init = time.time()
    
    inv_ix = {}
    i=0
    for doc, key in zip(docs, docs_keys):
        for token in doc:
            try: inv_ix[token].append(key)
            except: inv_ix[token] = [key]
            i+=1
    
    finish = time.time() - init
    logger_global.info('Inverted index created in %s s' % str(finish))
    return inv_ix



def write_inverted_index(filepath, inv_ix, docs_keys):
    """
write inverted index in csv file
    """
    logger_global.info('Writing Inverted Index on file...')
    init = time.time()
    
    f = open(filepath, 'w+')
    for key in inv_ix.keys():
        f.write(key.upper()+";%s\n" % inv_ix[key])     
    f.close()
    
    finish = time.time() - init
    logger_global.info('Inverted index written in %s s' % str(finish))



if __name__ == '__main__':

    import os
    PATH = os.path.dirname(os.path.abspath(__file__))

    config_file = '/2-INVERTED_INDEX/iv.cfg'
    
    with open(PATH+config_file.strip(), 'r') as configuration:
        config_vector=[]
        for line in configuration:
            line = line.strip()
            config_vector.append(line.split('='))

    inverted_index_generator(PATH, config_vector)