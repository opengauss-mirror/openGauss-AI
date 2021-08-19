Relation Algebra Interpreter (using Antlr v4)
===============================================

Created By Gaurav Mittal & Kaushal Yagnik
-----------------------------------------

####CSL - 451 Lab 2

###Summary

A interpreter which takes relational algebra expression as input from the user, automatically converts them to SQL queries, executes the queries on the database server, and outputs the results of the query to the user.

###Contents

On unzipping, the folder obtained 2012CSB1013_2012CSB1039 contains the following:

1. **Rel_Al_Interpreter** - folder containing the associated java project

2. **RelAlgebraInterpreter.jar** - runnable .jar file to run the project

3. **External_Libraries** - folder containing external jar dependencies 

###How to Run

* The code is already compiled and a runnable .jar file by the name RelAlgebraInterpreter.jar has been created to run the project.

* Before running the project, some configuration properties like **database name, username, password and database server ip and port number** need to be specified to properly connect to the database server. 

* Open RelAlgebraInterpreter.jar using Archive Manager and open the **config.properties** present in the root folder.

* Modify config.properties by entering the appropriate configuration properties. Save the file and update the jar file.

* Now, run RelAlgebraInterpreter.jar by opening the terminal and executing the following command from the directory containing the jar file:  
**`$ java -jar RelAlgebraInterpreter.jar`**

* On getting the following output, you are ready to input relational algebra expressions and get their output:  
```
Connecting to the Database <database_name>...
Connected to Database <database_name>!
```

###Description

* The Relation Algebra Interpreter is developed in Java using ANTLR v4. 

* ANTLR (ANother Tool for Language Recognition) is a powerful parser generator for reading, processing, executing, or translating structured text. From a grammar, ANTLR generates a parser that can build and walk parse trees.

* The Interpreter first established a connection with the database server using the specified configuration and then accepts relation algebra expressions as input from the user.

* A grammar has been specified for the relational algebra expressions which is used to generate a parser.

* This parser build the parse tree and walk down the tree converting the relational algebra expression into appropriate SQL query.

* After the entire SQL query has been built, it's executed on the server and the output obtained is displayed to the user.

**Relational Algebra Expression Properties**

* All relational algebra expressions are accepted in a **case insensitive** manner. 

* They can be nested within each other to any extent in a valid manner.

* To take care of the 'as' clause in the query, the interpreter automatically assigns dummy names to the nested expression wherever necessary in a preorder fashion starting from rel0, rel1 and so on.

* No semicolon is needed to end the expression.

* Following is the syntax to be followed to input the various kinds of relational algebra expressions:
    * **`select [predicate] (relation)`**
    
    *  **`projection [attribute1, attribute2, ...] (relation)`** 
    
    *  **`natural_join (relation1, relation2)`**
    
    *  **`cartesian_product (relation1, relation2, ...)`**

###Sample Expression

Here's a list of sample relational algebra expressions to execute on the rating database:

* **`select [mid>105] (movie)`**    
  Outputs the tuples from movie relation having mid greater than 105 

* **`projection [mid,rid] (natural_join(movie,rating))`**  
  Output the columns mid and rid from the natural join of the relations movie and rating

* **`select [movie.mid = rating.mid and rating.rid = reviewer.rid] (cartesian_product (movie, rating, reviewer))`**  
Outputs the cartesian product of movie, rating and reviewer with mid of movie and rating being equal with rid of rating and reviewer being equal

* **`projection [rel0.mid,rel0.rid] (select [] (natural_join(movie,rating)))`**  
  Outputs the columns mid and rid of the natural join of the relations movie and rating








