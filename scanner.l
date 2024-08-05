/*
    Flex scanner (also known as lexer/lexical analyzer or tokenizer) for the
    Citip grammar. The scanner transforms a byte input stream into a stream
    of tokens.

    For those who are new to flex/bison:

    This file is transpiled (using 'flex scanner.l') to a .cpp source file
    which implements a tokenizer function that returns on each call the
    integer tag of a token and stores additional information into its output
    parameters. The function declaration looks as follows:

        int yylex(parser::semantic_type* yylval_param,  // [out]
                  parser::location_type* yylloc_param,  // [in/out]
                  yyscan_t yyscanner)                   // scanner object
*/

 /*
    c++                 generate C++ parser class
    8bit                don't fail on 8-bit input characters
    warn                warn about inconsistencies
    nodefault           don't create default echo-all rule
    noyywrap            don't use yywrap() function
 */
%option c++
%option 8bit warn nodefault
%option noyywrap

 /* code that only goes into the implementation file */
%{
    #include <stdexcept>
    #include "parser.hxx"
    #include "scanner.hpp"

    // utility macros to simplify the actions
    #define YIELD_TOKEN(tok, val, type)     yylval->build<type>(val); \
                                            return yy::parser::token::T_##tok;

    #define YY_TXT(skip)            std::string(std::string(yytext + (skip), yyleng - (skip)))
    #define YY_NUM(skip)            std::stoi(YY_TXT(skip))
    #define YY_DBL(skip)            std::stod(YY_TXT(skip))

    #define INT_TOKEN(tok, val)     YIELD_TOKEN(tok, val, int)
    #define NUM_TOKEN(tok, skip)    YIELD_TOKEN(tok, YY_DBL(skip), double)
    #define STR_TOKEN(tok, skip)    YIELD_TOKEN(tok, YY_TXT(skip), std::string)
    #define LITERAL                 return yytext[0];

    #define YY_USER_ACTION          yylloc->columns(yyleng);
%}

%%

 /* code to be executed at every yylex() call */
%{
    yylloc->step();
%}

 /* Keywords */

approx                                 INT_TOKEN(APPROX, 0)
indist                                 INT_TOKEN(INDIST, 0)
implicit                               INT_TOKEN(IMPLICIT, 0)
prove                                  INT_TOKEN(PROVE, 0)

 /* Distinguish between entropies and random variables by checking for a following '('. */
I[[:alnum:]_{}\\]*/\(                  STR_TOKEN(MUTUAL_INFORMATION, 1)
H[[:alnum:]_{}\\]*/\(                  STR_TOKEN(ENTROPY, 1)

 /* Coefficient to optimize in the target bound. */
c[[:digit:]]+                          INT_TOKEN(OPT_COEFF, YY_NUM(1))

[[:alnum:]_{}\\]*[[:alpha:]{}\\][[:alnum:]_{}\\]*  STR_TOKEN(NAME, 0)

[[:digit:]]+                           STR_TOKEN(INT, 0) /* Delay parsing in case it's a scenario. */
[[:digit:]]*\.[[:digit:]]+             NUM_TOKEN(NUM, 0)

\<-                                    INT_TOKEN(FUNCTION_OF, 0)

==?                                    INT_TOKEN(REL, ast::REL_EQ)
\<=                                    INT_TOKEN(REL, ast::REL_LE)
\>=                                    INT_TOKEN(REL, ast::REL_GE)

\+                                     INT_TOKEN(SIGN, ast::SIGN_PLUS)
\-                                     INT_TOKEN(SIGN, ast::SIGN_MINUS)

#.*                                    {/* eat comments */}
[ \t]                                  {/* eat whitespace */}

\n                                     yylloc->lines(1); LITERAL

                                       /* forward everything else, even invalid
                                        * tokens - making use of bison's automatic
                                        * error messages */
.                                      LITERAL

%%

yy::scanner::scanner(std::istream* in, std::ostream* out)
    : yyFlexLexer(in, out)
{
}

int yyFlexLexer::yylex()
{
    throw std::logic_error(
        "The yylex() exists for technical reasons and must not be used.");
}
