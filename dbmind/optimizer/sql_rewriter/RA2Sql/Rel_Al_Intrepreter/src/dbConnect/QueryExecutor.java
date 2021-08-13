package dbConnect;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;

public class QueryExecutor {

	public static void execute(String query, Connection c) {
		try {
			/* create a Statement object to execute query */
			Statement stmt = c.createStatement();
			/* execute a query on the server */
			ResultSet rs = stmt.executeQuery(query);
			/* obtain information about the resulting relation */
			ResultSetMetaData rsm = rs.getMetaData();
			/* first, print the names of the attributes */
			int ncolumns = rsm.getColumnCount();
			for (int i = 1; i <= ncolumns; i++) {
				if (i > 1)
					System.out.print(",  ");
				System.out.print(rsm.getColumnName(i));
			}
			System.out.print("\n");
			/* print the result of the query */
			while (rs.next()) {
				for (int i = 1; i <= ncolumns; i++) {
					if (i > 1)
						System.out.print(",  ");
					String columnValue = rs.getString(i);
					System.out.print(columnValue);
				}
				System.out.println("");
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getClass().getName() + ": " + e.getMessage());
			System.exit(0);
		}
	}

}
