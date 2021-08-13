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
 
 

import numpy as np

def get_batch_job(batch_id, istest=False, directory='/home/sunji/learnedcardinality/job'):
    if istest:
        suffix = 'test_'
    else:
        suffix = ''
    target_cost_batch = np.load(directory+'/target_cost_'+suffix+str(batch_id)+'.np.npy')
    target_cardinality_batch = np.load(directory+'/target_cardinality_'+suffix+str(batch_id)+'.np.npy')
    operators_batch = np.load(directory+'/operators_'+suffix+str(batch_id)+'.np.npy')
    extra_infos_batch = np.load(directory+'/extra_infos_'+suffix+str(batch_id)+'.np.npy')
    condition1s_batch = np.load(directory+'/condition1s_'+suffix+str(batch_id)+'.np.npy')
    condition2s_batch = np.load(directory+'/condition2s_'+suffix+str(batch_id)+'.np.npy')
    samples_batch = np.load(directory+'/samples_'+suffix+str(batch_id)+'.np.npy')
    condition_masks_batch = np.load(directory+'/condition_masks_'+suffix+str(batch_id)+'.np.npy')
    mapping_batch = np.load(directory+'/mapping_'+suffix+str(batch_id)+'.np.npy')
    return target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch,\
           condition2s_batch, samples_batch, condition_masks_batch, mapping_batch