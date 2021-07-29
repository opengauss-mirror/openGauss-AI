import os
import sqlparse
import time

import jpype as jp
from jpype.types import *
import jpype.imports

''' Configure JAVA environment for JPype '''
base_dir = os.path.abspath(os.curdir)
local_lib_dir = os.path.join(base_dir, 'libs')

# For the first use: uncomment if `classpath.txt` need update
# Otherwise: commoent "_ = os.popen('mvn dependency:build-classpath -Dmdep.outputFile=classpath.txt').read()"
_ = os.popen('mvn dependency:build-classpath -Dmdep.outputFile=classpath.txt').read()

classpath = open(os.path.join(base_dir, 'classpath.txt'), 'r').readline().split(':')
classpath.extend([os.path.join(local_lib_dir, jar) for jar in os.listdir(local_lib_dir)])
# print('\n'.join(classpath))

if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), classpath=classpath)

# check JVM classpth
# check_classpath = str(jp.java.lang.System.getProperty('java.class.path'))
# print('\n'.join(check_classpath.split(':')))


from javax.sql import DataSource
from java.sql import Connection, DriverManager
from java.util import ArrayList, List

from org.postgresql import Driver as PostgreSQLDriver

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

from config import CONFIG,parse_cmd_args
from rewrite_by_rule import rewrite
from database import execute_sql, fetch_execution_time


''' Implement SQL2RA & RA2SQL '''
def initialize_rewriter():
    try:
        if planner: pass
    except:
        conn = DriverManager.getConnection('jdbc:calcite:')
        calcite_conn = conn.unwrap(CalciteConnection)
        root_schema = calcite_conn.getRootSchema()
        # database config
        data_source = JdbcSchema.dataSource(CONFIG['url'], CONFIG['driver'], CONFIG['username'], CONFIG['password'])
        schema = root_schema.add(CONFIG['schema'], JdbcSchema.create(root_schema, CONFIG['schema'], data_source, None, None))
        config = Frameworks.newConfigBuilder().defaultSchema(schema).build()
        planner = Frameworks.getPlanner(config)
    print('planner configured')

    try:
        if dialect: pass
    except:
        dialect = PostgresqlSqlDialect.DEFAULT
    print('dialect configured')

    return planner, dialect

def main():

    argus = parse_cmd_args()

    # config java environment
    # base_dir = start_java_env()
    query_dir = os.path.join(base_dir, 'queries', argus['workload'])

    # initialize sql rewriter
    planner, dialect = initialize_rewriter()

    # rule type: RelOptRule
    # rule = jpype.JClass("org.apache.calcite.rel.rules.FilterMergeRule") # FilterMergeRule
    # rb = jpype.JClass(RelBuilderFactory)
    ruledir = jp.JPackage('org.apache.calcite.rel.rules')
    #rule = ruledir.AbstractJoinExtractFilterRule

    timestamp=int(time.time())

    with open(CONFIG['logdir']+'/'+str(timestamp),'a') as logf:
        logf.write(' rewriting results '.center(60, '-')+'\n')

    # rewrite all the queries within the ``./queries/input_sql/'' directory
    for file in sorted(os.listdir(query_dir)):
        if not file.endswith('.sql'):
            continue
        os.rename(query_dir+'/'+file, query_dir+'/'+file.replace('mysql','tpch'))
        file = file.replace('mysql','tpch')
        with open(os.path.join(query_dir, file), 'r') as sql_file:
            sqls = [sql.value for sql in sqlparse.parse(sql_file.read()) if sql.get_type() == 'SELECT']
            for sql in sqls:
                rewrite_sql = rewrite(sql, dialect, planner)

                res = execute_sql('explain analyze ' + str(sql))
                tim = fetch_execution_time(res)
                # print(tim)

                res = execute_sql('explain analyze ' + str(rewrite_sql))
                rewrite_tim = fetch_execution_time(res)
                if rewrite_tim != -1:       # successfully rewrite
                    with open(CONFIG['logdir']+'/' + str(timestamp), 'a') as logf:
                        logf.write(file+'\n')
                        logf.write(' origin sql '.center(60, '-')+'\n')
                        logf.write(str(sql)+'\n')
                        logf.write(tim+'\n')
                        logf.write(' rewritten sql '.center(60, '-')+'\n')
                        logf.write(str(rewrite_sql)+'\n')
                        logf.write(rewrite_tim+'\n')

                    print(file, tim, rewrite_tim)

                    # detailed rewrite information is logged

if __name__ == '__main__':
    main()

