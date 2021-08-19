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
 
 

import unittest
from src.feature_extraction.extract_features import *
from src.training.train_and_test import *

class TestFeatureExtraction(unittest.TestCase):
    def test(self):
        data_dir = '/home/sunji/cost_estimation/test_files_open_source/imdb_data_csv'
        tmp_dir = '/home/sunji/cost_estimation/test_files_open_source'
        dataset = load_dataset(data_dir)
        column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, table_names = prepare_dataset(dataset)
        sample_num = 1000
        sample = prepare_samples(dataset, sample_num, table_names)
        feature_extractor(tmp_dir+'/plans.json', tmp_dir+'/plans_seq.json')
        add_sample_bitmap(tmp_dir+'/plans_seq.json', tmp_dir+'/plans_seq_sample.json', dataset, sample, sample_num)

if __name__ == '__main__':
    unittest.main()
