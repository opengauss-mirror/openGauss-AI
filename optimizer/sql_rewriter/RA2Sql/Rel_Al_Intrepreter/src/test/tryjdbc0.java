package test;

//
// tryjdbc0.java
//
//  Connects to the rating database and retrieves all tuples in the
//  movie relation. It is assumed that the rating database exists in
//  local server. You can use the accompanying rating.txt file to
//  create and populate the database.
//
//  Created by Narayanan Chatapuram Krishnan on 12/02/15.
//

/* import the sql package and not the postgresql/openGauss package. */
import java.sql.*;

public class tryjdbc0
{
    public static void main(String args[])
    {
        /* create a successful connection to the database server. */
        Connection c = null;
        try
        {
            Class.forName("org.postgresql.Driver");
            c = DriverManager.getConnection("jdbc:postgresql://localhost:5432/rating",
                           "jdbc","password");
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.err.println(e.getClass().getName()+": "+e.getMessage());
            System.exit(0);
        }
        System.out.println("Opened database successfully");
        
        try
        {
            Statement stmt = c.createStatement();
            /* execute a query on the server*/
            ResultSet rs = stmt.executeQuery("select * from movie;");
            /* obtain information about the resulting relation*/
            ResultSetMetaData rsm = rs.getMetaData();
            /* first, print the names of the attributes*/
            int ncolumns = rsm.getColumnCount();
            for (int i = 1; i <= ncolumns; i++)
            {
                if (i > 1) System.out.print(",  ");
                System.out.print(rsm.getColumnName(i));
            }
            System.out.print("\n");
            /* print the result of the qeury */
            while (rs.next())
            {
                for (int i = 1; i <= ncolumns; i++)
                {
                    if (i > 1) System.out.print(",  ");
                    String columnValue = rs.getString(i);
                    System.out.print(columnValue);
                }
                System.out.println("");
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.err.println(e.getClass().getName()+": "+e.getMessage());
            System.exit(0);
        }
    }
}