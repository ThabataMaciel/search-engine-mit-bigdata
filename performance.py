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

def performance_evaluator(path, config_vector):
    """
calls functions to read inverted index and creates TF.IDF matrix
    """

    begin = time.time()
    
    #instantiate logging 
    global logger_global
    log_path = path+'/6-PERFORMANCE/pe.log'
    log('performance', log_path)
    logger_global = logging.getLogger('performance')
    logger_global.info('Processing Performance Evaluator Module...')

    for config in config_vector:
        if str(config[0]) == 'RESULTADOS':            
            results = read_results(path+config[1].strip())
            
        elif str(config[0]) == 'ESPERADOS':
            answers = read_answers(path+config[1].strip())
        
        elif str(config[0]) == 'DESEMPENHO':
            outfile_performance = path+config[1]
    
    if not (outfile_performance):
        outfile_performance = path+'/6-PERFORMANCE/performance.csv'
        logger_global.warning('Outfile not specified. '
                             'Applying default: '+outfile_performance)

    end = time.time() - begin
    logger_global.info('Results and answers read succesfully '
                      'in %s s' % str(end))
    
    logger_global.info('Calculating performance...')
    
    performance_raw = precision_recall(answers, results)
    performance = interpolated_precision_recall(performance_raw)
    curve_11 = eleven_point_curve(performance)
    ndcg(answers, results)
    
    end = time.time() - end
    logger_global.info('Performance calculated succesfully '
                      'in %s s' % str(end))

    write_performance(outfile_performance, performance)
    #show_precision_recall(performance, '00005', 20)
    #print_precision_recall_curves(performance)
    show_11_point_curve(curve_11)

    end = time.time() - begin
    logger_global.info('End of Performance Evaluator Module. '
                       'Total of %s elapsed.' % str(end))



def read_results(filename):
    """
reads results from csv file
returns dictionary: 
    results[query key] = array of results tuples (rank, docnum, sim)
    """
    import ast
    logger_global.info('Reading results file...')
    init = time.time()
    results = {}
    
    with open(filename,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            results[row[0]] = ast.literal_eval(row[1])

    logger_global.info('Results file read in %s s' % str(time.time()-init))
    return results



def read_answers(filename):
    """
reads answers (expected results) from from csv file
returns dictionary: 
    answers[query key] = array of answer tuples (docnum, score)
    """
    import ast
    logger_global.info('Reading answer file...')
    init = time.time()
    answers = {}
    
    with open(filename,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            answers[row[0]] = ast.literal_eval(row[1])

    logger_global.info('Answers file read in %s s' % str(time.time()-init))
    return answers



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



def precision_recall(answers, results):
    """
answers[query key] = array of answer tuples (docnum, score)
results[query key] = array of results tuples (rank, docnum, sim)
performance[query_key] = array of performance tuple (precision, recall)
    """
    performance = {}
    for query_key, result in results.items():
        answer = answers[query_key]   
        performance[query_key] = []
    
        n = len(answer)
        t, p = 1, 0
        for my_doc in result:
            my_doc_num = int(my_doc[1].strip())
            for right_doc in answer:
                right_doc_num = int(right_doc[0].strip())
                if my_doc_num == right_doc_num: p += 1
            performance[query_key].append((p/t, p/n))
            t += 1
    
    return performance        
        
    
def interpolated_precision_recall(performance):
    """
interpolates for maximum precision for all queries
keeps recall domain intact
    """
    curve = {}
    for query_key, result in performance.items():
        curve[query_key] = []
        for i in range(len(result)-1):
            curve[query_key].append(result[i])
            if result[i+1][0] > result[i][0]:
                while (curve[query_key][-1][0] < result[i+1][0]):
                    del curve[query_key][-1]
                    if len(curve[query_key]) == 0: break
        curve[query_key].append(result[-1])
                
    return curve
                


def eleven_point_curve(performance):
    """
1) computes interpolated precision at eleven recall levels
2) averages over all queries to get eleven-point precision/recall curve
returns array of tuples (precision, recall)
    """
    curve = {}
    for query_key, pr in performance.items():
        result = list(pr)
        curve[query_key] = []
        curve[query_key].append((result[0][0],0.0))
        for recall in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            while (result[0][1] < recall) and (len(result) > 1):
                del result[0]
            curve[query_key].append((result[0][0],recall))
        curve[query_key].append((result[-1][0],1.0))

    curve_avg = []    
    for i in range(0, 11, 1):
        precision = [pr[i][0] for pr in curve.values()]
        curve_avg.append((sum(precision) / len(precision), float(i)/10.))
        
    
    return curve_avg



def show_precision_recall(performance, key, K):
    """
shows scatter plot for a precision recall curve for one query
logs precision @ K
    """
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    recall = [x[1] for x in performance[key]]
    precision = [x[0] for x in performance[key]]
    plt.plot(recall, precision, 'o', color='black');
    
    #logger_global.info('Precision for query {} at K={} is {}'.format(
    #                                        key,
    #                                        K,
    #                                        precision[K-1]))


# =============================================================================
#def print_precision_recall_curves(performance):
#    import matplotlib.pyplot as plt
#    plt.style.use('seaborn-whitegrid')
#    
#     for key, row in performance.items():
#         recall = [x[1] for x in performance[key]]
#         precision = [x[0] for x in performance[key]]
#         plt.plot(recall, precision, 'o', color='black');
#         plt.savefig(key+'.png')
#         plt.close()
#   
#     see = ['00010', '00011', '00015', '00031',
#            '00033', '00037', '00044', '00057',
#            '00090', '00092']
#     plt.figure(figsize=(20,10))
#     plt.rcParams.update({'font.size': 18})
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     for key in see:
#         recall = [x[1] for x in performance[key]]
#         precision = [x[0] for x in performance[key]]
#         plt.plot(recall, precision, label=key)
#         plt.legend(loc='upper right')
#         
#     plt.savefig('Bottom10.png', dpi=100)
#     plt.close()
# =============================================================================

def show_11_point_curve(curve):
    """

    """
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    recall = [x[1] for x in curve]
    precision = [x[0] for x in curve]
    plt.figure(figsize=(20,10))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.rcParams.update({'font.size': 18})
    plt.plot(recall, precision, color='black')
    plt.savefig('11PointCurve.png', dpi=100)
    plt.close()



def ndcg(answers, results):
    """
answers[query key] = array of answer tuples (docnum, score)
results[query key] = array of results tuples (rank, docnum, sim)
dcg[query key] = array of dcg values ordered by my document rank
ndcg[query key] = array of ndcg values ordered by my document rank
plots dcg and ndcg in bar plots
    """
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    dcg = {}
    ndcg = {}
    for query_key, result in results.items():
        
        answer = answers[query_key]
        dcg[query_key] = []
        icg = []
        dcg_value = 0.
        for my_doc in result:
            my_doc_num = int(my_doc[1].strip())
            for right_doc in answer:
                right_doc_num = int(right_doc[0].strip())
                if my_doc_num == right_doc_num:
                    relevance = [int(r) for r in list(right_doc[1].strip())]
                    relevance = sum(relevance)
                    rank = my_doc[0]
                    dcg_value += relevance / rank
                    icg.append(relevance)
                    dcg[query_key].append(dcg_value)
        
        icg.sort(reverse=True)
        idcg_value = 0.
        ndcg[query_key] = []
        i = 1
        for icg_value in icg:
            idcg_value += icg_value/i
            ndcg[query_key].append(dcg[query_key][i-1]/idcg_value)
            i += 1
            
    x = list(ndcg.keys())
    y = [ndcg[key][-1] for key in x]
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 18})
    plt.bar(x, y)
    plt.xticks([])
    plt.savefig('NDCG.png', dpi=100)
    plt.close()
    
    logger_global.info('Maximum NDCG is '+str(max(y))+' for '+ x[y.index(max(y))])
    logger_global.info('Minimum NDCG is '+str(min(y))+' for '+ x[y.index(min(y))])


def write_performance(filepath, results):
    """
writes performance in csv file
    """
    init = time.time()
    logger_global.info('Writing performance on file...')
    
    with open(filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        for key, row in results.items():
            csv_writer.writerow([key, row])
    
    end = time.time() - init
    logger_global.info('Write operation finished with %s s' % str(end))



if __name__ == '__main__':

    import os
    PATH = os.path.dirname(os.path.abspath(__file__))

    config_file = '/6-PERFORMANCE/pe.cfg'
    
    with open(PATH+config_file.strip(), 'r') as configuration:
        config_vector=[]
        for line in configuration:
            line = line.strip()
            config_vector.append(line.split('='))

    performance_evaluator(PATH, config_vector)