lexer grammar RelLex;

/*
 * This file defines the valid lexicons which the relational
 * algebra interpreter will recognize and tokenize.
 * Care has been taken to make the lexicons case-insensitive
 * just the way tokens are treated by PostgreSQL/openGauss
 */
 
 //select token for the selection expression
SELECT
:
	S E L E C T
;

//project token for the projection expression
PROJECT
:
	P R O J E C T I O N
;


//natural token for the natural join expression
NATURAL
:
	N A T U R A L [_] J O I N
;


//cartesian token for the cartesian product expression
CARTESIAN
:
	C A R T E S I A N [_] P R O D U C T
;

//attribute token to recognize valid attribute names
ATTRIBUTE
:
	'\''.*?'\''
;

//predicate token to recognize valid predicates
PREDICATE
:
	'[' .*? ']'
;

//relation token to recognize valid relation names
RELATION
:
	[a-zA-Z] [a-zA-Z0-9_]*
;

//lexer rule to skip whitespaces, newlines and carriage returns
WS
:
	[ \t\r\n]+ -> skip
;


//fragments defined to make the input case-insensitive
fragment A:('a'|'A');
fragment B:('b'|'B');
fragment C:('c'|'C');
fragment D:('d'|'D');
fragment E:('e'|'E');
fragment F:('f'|'F');
fragment G:('g'|'G');
fragment H:('h'|'H');
fragment I:('i'|'I');
fragment J:('j'|'J');
fragment K:('k'|'K');
fragment L:('l'|'L');
fragment M:('m'|'M');
fragment N:('n'|'N');
fragment O:('o'|'O');
fragment P:('p'|'P');
fragment Q:('q'|'Q');
fragment R:('r'|'R');
fragment S:('s'|'S');
fragment T:('t'|'T');
fragment U:('u'|'U');
fragment V:('v'|'V');
fragment W:('w'|'W');
fragment X:('x'|'X');
fragment Y:('y'|'Y');
fragment Z:('z'|'Z');
