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
 
 

from src.feature_extraction.node_features import *


def plan2seq(root, alias2table):
    sequence = []
    join_conditions = []
    node, join_condition = extract_info_from_node(root, alias2table)
    if join_condition is not None:
        join_conditions += join_condition
    sequence.append(node)
    if 'Plans' in root:
        for plan in root['Plans']:
            next_sequence, next_join_conditions = plan2seq(plan, alias2table)
            sequence += next_sequence
            join_conditions += next_join_conditions
    sequence.append(None)
    return sequence, join_conditions
