/*
 * Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 *
 * openGauss is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
 
 

import logging  # Setting up the loggings to monitor gensim
import pickle
from time import time  # To time our operations

from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

with open('more_sentences_train_query.pkl', 'rb') as f:
    content = f.read()
    sentences = pickle.loads(content)
print('New Sentences Loaded')
w2v_model = Word2Vec.load("word2vec_updated.model")
print('Model Loaded')

t = time()
w2v_model.build_vocab(sentences, progress_per=10000, update=True)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.save("word2vec_updated.model")
print('model saved')

path = get_tmpfile("wordvectors_updated.kv")
w2v_model.wv.save(path)
print('word saved')
'''
model.build_vocab(new_sentences, update=True)
model.train(new_sentences)
'''
