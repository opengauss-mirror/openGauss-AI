package dbConnect;

import java.io.IOException;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

public class DatabaseConnectionManager {

	Connection conn = null;

	public DatabaseConnectionManager() throws SQLException {
		// TODO Auto-generated constructor stub
		connect();
	}

	private void connect() throws SQLException {
		// TODO Auto-generated method stub
		

		//Defining properties object to read appropriate parameter
		//from the config.properties file
		Properties prop = new Properties();
		String filename = "config.properties";

		InputStream input = null;
		input = DatabaseConnectionManager.class.getClassLoader()
				.getResourceAsStream(filename);
		
		//Trying to establish connection with default settings
		//in case no file is found or input is null
		if (input == null) {
			System.out.println("Sorry, unable to find " + filename);
			System.out.println("Trying default settings...");
			System.out.println("Connecting to Database rating...");
			conn = DriverManager.getConnection(
					"jdbc:postgresql://localhost:5432/rating", "jdbc",
					"password");
			System.out.println("Connected to Database rating!");
		} else {
			//input from config.properties is valid
			//establishing connection with the database
			try {
				//loading the properties
				prop.load(input);
				//System.out.println(prop.getProperty("ipport"));
				//System.out.println(prop.getProperty("database"));
				//System.out.println(prop.getProperty("user"));
				//System.out.println(prop.getProperty("password"));
				
				System.out.println("Connecting to Database "+prop.getProperty("database")+"...");
				conn = DriverManager.getConnection(
						"jdbc:postgresql://"+prop.getProperty("ipport")+"/"
								+ prop.getProperty("database"),
						prop.getProperty("user"), prop.getProperty("password"));
				System.out.println("Connected to Database "+prop.getProperty("database") + "!");

			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
		
	}

	//getter method to get the connection object
	public Connection getConnection() {
		if (conn == null)
			try {
				connect();
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		return conn;
	}

}
