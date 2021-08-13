grammar RelAlgebra;

//importing the lexer rules defined in RelLex.g4
import RelLex;

/*
 * This file defines the grammar rules required to parse
 * the relational algebra expressions from the input
 * accepted by the user
 */

// expression rule saying that expression can be either selection,
// projection, natural join or cartesian product
expression
:
	select
	| project
	| natural
	| cartesian
;

// select rule to parse selection expression consisting of
// select token, predicate and relation in parentheses
select
:
	SELECT PREDICATE '(' relation ')'
;

// project rule to parse projection expression consisting of
// project token, list of attribute names taken as a predicate
// and relation in parentheses
project
:
	PROJECT PREDICATE '(' relation ')'
;

// natural rule to parse natural join expression consisting of
// natural token, relation 1 and relation 2
natural
:
	NATURAL '(' relation ',' relation ')'
;

// cartesian rule to parse cartesian product expression consisting
// of cartesian token followed by list of relations
cartesian
:
	CARTESIAN '(' relation
	(
		',' relation
	)* ')'
;


// relation rule saying that relation can be a simple name of a relation
// or it can be an expression in itself
relation
:
	RELATION #simple 
	| expression #nested
	
;