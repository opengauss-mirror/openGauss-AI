/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openGauss-AI is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package hr.fer.zemris.calcite.sqlparser;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.avatica.util.Quoting;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.impl.SqlParserImpl;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.util.SourceStringReader;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlConformanceEnum;

/**
 * Hello world!
 *
 */
public class SQLParse 
{
    public static void main( String[] args ) throws SqlParseException
    {
	SqlParser sqlParser = SqlParser.create(new SourceStringReader(args[0]),
                                            SqlParser.configBuilder()
                                                     .setParserFactory(SqlParserImpl.FACTORY)
                                                     .setQuoting(Quoting.DOUBLE_QUOTE)
                                                     .setUnquotedCasing(Casing.TO_UPPER)
                                                     .setQuotedCasing(Casing.UNCHANGED)
                                                     .setConformance(SqlConformanceEnum.DEFAULT)
                                                     .build());
         SqlNode sqlNode = sqlParser.parseQuery();
         System.out.println(sqlNode.toString());
    }
}
