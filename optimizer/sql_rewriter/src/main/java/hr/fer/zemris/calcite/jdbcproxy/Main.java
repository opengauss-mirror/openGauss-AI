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

package hr.fer.zemris.calcite.jdbcproxy;

import java.sql.Connection;
import java.sql.DriverManager;
import java.util.Properties;

public class Main {

	/*
	 * Argument is a model.json file
	 */
	public static void main (String[] argv) throws Exception {

		Class.forName(org.apache.calcite.jdbc.Driver.class.getName());
		Properties info = new Properties();
		info.setProperty("lex", "JAVA");
		info.setProperty("model", argv[0]);
		Connection calConnection = DriverManager.getConnection("jdbc:calcite:", info);
	}
}
