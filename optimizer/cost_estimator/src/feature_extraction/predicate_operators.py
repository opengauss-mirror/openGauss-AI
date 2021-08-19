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
 
 

class Operator(object):
    def __init__(self, opt):
        self.op_type = 'Bool'
        self.operator = opt

    def __str__(self):
        return 'Operator: ' + self.operator


class Comparison(object):
    def __init__(self, opt, left_value, right_value):
        self.op_type = 'Compare'
        self.operator = opt
        self.left_value = left_value
        self.right_value = right_value

    def __str__(self):
        return 'Comparison: ' + self.left_value + ' ' + self.operator + ' ' + self.right_value
