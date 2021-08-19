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
 
 

#!/usr/bin/python

["database_connection"]

local_pg = {
    "host": "localhost",
    "database": "",
    "user": "",
    "password": ""
}


["encoder_reducer"]

enc_rdc_model_path = "./result/job/encoder_reducer/2021.4.27-1/"
enc_rdc_trainset_dir = "./dataset/JOB/trainset/"


["DQN"]

dqn_budget = 500
