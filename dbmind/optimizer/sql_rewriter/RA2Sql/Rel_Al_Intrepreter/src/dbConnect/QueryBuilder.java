package dbConnect;

import interpreter.RelAlgebraInterpreter;
import interpreter.RelAlgebraLexer;
import interpreter.RelAlgebraParser;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;


public class QueryBuilder {

	public static String buildQuery(String relationalExp) throws IOException{
		// Defining the input stream with the relational algebra expression
		InputStream is = new ByteArrayInputStream(relationalExp.getBytes());
		// tokenizing of the input
		ANTLRInputStream input = new ANTLRInputStream(is);
		// Tokenizing the input using RelAlgebraLexer object
		RelAlgebraLexer lexer = new RelAlgebraLexer(input);
		// Creating a buffer of tokens from the lexer
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		// Creating a parser that feeds off the token buffer
		RelAlgebraParser parser = new RelAlgebraParser(tokens);
		// Beginning the parsing at the expression rule
		ParseTree tree = parser.expression();
		//Creating RelAlgebraInterpreter object to visit the parsed tree
		RelAlgebraInterpreter interpreter = new RelAlgebraInterpreter();
		
		//Creating the final query from the string returned by the visit function
		//of the RelAlgebraInterpreter object
		String query = interpreter.visit(tree) + ";";
		
		//System.out.println(tree.toStringTree(parser));

		System.out.println("################## SQL Statement after Rewrite ##################");
		System.out.println(query);
		
		//Return the SQL query built to be executed
		return query;
	}
}