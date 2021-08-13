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
 
 

import os
import sqlparse
import org.apache.calcite.rel.rules as R
from org.apache.calcite.adapter.jdbc import JdbcSchema
from org.apache.calcite.jdbc         import CalciteConnection
from org.apache.calcite.plan         import RelOptUtil, RelOptRule
from org.apache.calcite.plan.hep     import HepMatchOrder, HepPlanner, HepProgram, HepProgramBuilder
from org.apache.calcite.rel          import RelRoot, RelNode
from org.apache.calcite.rel.rel2sql  import RelToSqlConverter

from org.apache.calcite.rel.rules    import FilterJoinRule,AggregateExtractProjectRule,FilterMergeRule

from org.apache.calcite.schema       import SchemaPlus
from org.apache.calcite.sql          import SqlNode, SqlDialect
from org.apache.calcite.sql.dialect  import CalciteSqlDialect, PostgresqlSqlDialect
from org.apache.calcite.tools        import FrameworkConfig, Frameworks, Planner, RelBuilderFactory
from org.apache.calcite.util         import SourceStringReader


def parse_quote(sql):

    new_sql = ""

    for token in sqlparse.parse(sql)[0].flatten():
        if token.ttype is sqlparse.tokens.Name and token.parent and not isinstance(token.parent.parent, sqlparse.sql.Function):
            new_sql += '\"' + token.value + '\"'
        elif token.value != ';':
            new_sql += token.value

    return new_sql


def SQL2RA(planner, sql):
    planner.close()
    planner.reset()
    sql_node = planner.parse(SourceStringReader(sql))
    sql_node = planner.validate(sql_node)
    rel_root = planner.rel(sql_node)
    rel_node = rel_root.project()
    program = HepProgramBuilder().addMatchOrder(HepMatchOrder.TOP_DOWN).build()
    # program = HepProgramBuilder().addRuleInstance(rule).build()
    # test = FilterJoinRule()
    # test = jpype.JClass(FilterJoinRule)
    # program = HepProgramBuilder().addRuleInstance(test).build()

    hep_planner = HepPlanner(program)
    hep_planner.setRoot(rel_node)
    rel_node = hep_planner.findBestExp()
    return rel_node


def RA2SQL(dialect, ra):
    converter = RelToSqlConverter(dialect)
    return converter.visitInput(ra, 0).asStatement().toSqlString(dialect).getSql()


# rewrite query with a rewrite rule

def rewrite(sql, dialect, planner):

    #print(' origin sql '.center(60, '-'))
    #print(sql)

    #print(' formatted sql '.center(60, '-'))
    format_sql = parse_quote(sql)
    #print(format_sql)

    #print(' rewritten logic plan '.center(60, '-'))
    ra = SQL2RA(planner, format_sql)
    #print(RelOptUtil.toString(ra))

    #print(' rewritten sql '.center(60, '-'))
    sql = RA2SQL(dialect, ra)
    #print(sql)

    return sql
