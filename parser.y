 /*
    Bison parser definition.

    For those who are new to flex/bison:

    This file is transpiled (using 'bison parser.y') to a .cpp file that
    implements the class yy::parser. This class has the job to analyze the
    token stream resulting from repeated invocation of the yylex() function
    and create an AST (abstract syntax tree) of the given string expressions.

    Although the grammar is quite simple and does not require this level of
    parsing power, using bison was a nice getting-to-know exercise and makes
    language easier to extend and the code easier to maintain.
 */

/* C++ parser interface */
%skeleton "lalr1.cc"
%require  "3.0"

/* add parser members (scanner, cb) and yylex parameters (loc, scanner) */
%parse-param  {yy::scanner* scanner} {ParserCallback* cb}
%locations

/* increase usefulness of error messages */
%define parse.error verbose

/* assert correct cleanup of semantic value objects */
%define parse.assert

%define api.value.type variant
%define api.token.prefix {T_}

%token                  END     0   "end of file"

%token <std::string>    MUTUAL_INFORMATION
%token <std::string>    ENTROPY
%token <std::string>    NAME
%token <std::string>    INT
%token <double>         NUM
%token <int>            SIGN
                        REL
%token <int>            INDIST
%token <int>            IMPLICIT

%type <ast::Relation>                   inform_inequ
%type <ast::MutualIndependence>         mutual_indep
%type <std::vector<ast::VarList>>       mutual_indep_list
%type <ast::MarkovChain>                markov_chain
%type <std::vector<ast::VarList>>       markov_chain_list
%type <ast::FunctionOf>                 determ_depen
%type <bool>                            maybe_implicit
%type <ast::Expression>                 inform_expr
%type <ast::Term>                       inform_term
%type <ast::Quantity>                   inform_quant
%type <ast::EntropyQuantity>            entropy
%type <ast::EntropyQuantity>            mutual_inf
%type <ast::VarList>                    name_list
%type <std::vector<ast::VarList>>       mut_inf_core;
%type <std::string>                     name;
%type <ast::IndistinguishableScenarios> indist;
%type <ast::IndistinguishableScenarios> indist_list;
%type <ast::IndistinguishableScenario>  indist_item;

%start statement


%code requires {
    #include <stdexcept>
    #include <string>

    #include "ast.hpp"
    #include "location.hh"

    namespace yy {
        class scanner;
    };

    // results
    struct ParserCallback {
        virtual void relation(ast::Relation) = 0;
        virtual void markov_chain(ast::MarkovChain) = 0;
        virtual void mutual_independence(ast::MutualIndependence) = 0;
        virtual void function_of(ast::FunctionOf) = 0;
        virtual void indist(ast::IndistinguishableScenarios) = 0;
    };
}

%code {
    #include <iostream>     // cerr, endl
    #include <utility>      // move
    #include <string>

    #include "scanner.hpp"

    using std::move;

    #ifdef  yylex
    # undef yylex
    #endif
    #define yylex scanner->lex

    template <class T, class V>
    T&& enlist(T& t, V& v)
    {
        t.push_back(move(v));
        return move(t);
    }
}
%%

    /* deliver output */

statement    : %empty           { /* allow empty (or pure comment) lines */ }
             | inform_inequ     { cb->relation(move($1)); }
             | mutual_indep     { cb->mutual_independence(move($1)); }
             | markov_chain     { cb->markov_chain(move($1)); }
             | determ_depen     { cb->function_of(move($1)); }
             | indist           { cb->indist(move($1)); }
             ;

    /* statements */

inform_inequ      : inform_expr REL inform_expr                               { $$ = {$1, $2, $3}; }
                  ;

markov_chain      : markov_chain_list                                         { $$ = {{}, $1}; }
                  | name_list ':' markov_chain_list                           { $$ = {$1, $3}; }
                  ;

markov_chain_list : markov_chain_list '/' name_list                           { $$ = enlist($1, $3); }
                  | name_list '/' name_list '/' name_list                     { $$ = {$1, $3, $5}; }
                  ;

mutual_indep      : mutual_indep_list maybe_implicit                          { $$ = {{}, $1, $2}; }
                  | name_list ':' mutual_indep_list maybe_implicit            { $$ = {$1, $3, $4}; }
                  ;

mutual_indep_list : mutual_indep_list '.' name_list                           { $$ = enlist($1, $3); }
                  |         name_list '.' name_list                           { $$ = {$1, $3}; }
                  ;

determ_depen      : name_list '<' '-' name_list maybe_implicit                { $$ = {{}, $1, $4, $5}; }
                  | name_list ':' name_list '<' '-' name_list maybe_implicit  { $$ = {$1, $3, $6, $7}; }
                  ;

indist            : INDIST indist_list                                        { $$ = $2; }
                  | INDIST name_list                                          { $$ = {{{}, $2}}; }
                  | INDIST ':' name_list                                      { $$ = {{{}, $3}}; }
                  ;

indist_list       : indist_list ';' indist_item                               { $$ = enlist($1, $3); }
                  | indist_item                                               { $$ = {$1}; }
                  ;

indist_item       : name_list ':' name_list                                   { $$ = {$1, $3}; }
                  ;

    /* building blocks */

inform_expr    : inform_expr SIGN inform_term     { $$ = enlist($1, $3.flip_sign($2)); }
               |             SIGN inform_term     { $$ = {$2.flip_sign($1)}; }
               |                  inform_term     { $$ = {$1}; }
               ;

inform_term    : INT inform_quant                 { $$ = {std::stod($1), $2}; }
               | NUM inform_quant                 { $$ = {$1, $2}; }
               |     inform_quant                 { $$ = { 1, $1}; }
               | INT NAME                         { $$ = {std::stod($1), ast::VariableQuantity{$2}}; }
               | NUM NAME                         { $$ = {$1, ast::VariableQuantity{$2}}; }
               |     NAME                         { $$ = { 1, ast::VariableQuantity{$1}}; }
               | INT                              { $$ = {std::stod($1), ast::ConstantQuantity{}}; }
               | NUM                              { $$ = {$1, ast::ConstantQuantity{}}; }
               ;

inform_quant   : entropy                          { $$ = $1; }
               | mutual_inf                       { $$ = $1; }
               ;

entropy        : ENTROPY '(' name_list              ')'      { $$ = {{$1, {$3}}}; }
               | ENTROPY '(' name_list '|' name_list ')'      { $$ = {{$1, {$3}}, $5}; }
               ;

mutual_inf     : MUTUAL_INFORMATION '(' mut_inf_core              ')'  { $$ = {{$1, $3}}; }
               | MUTUAL_INFORMATION '(' mut_inf_core '|' name_list ')'  { $$ = {{$1, $3}, $5}; }
               ;

mut_inf_core   :  mut_inf_core ';' name_list     { $$ = enlist($1, $3); }
               |      name_list ';' name_list     { $$ = {$1, $3}; }
               ;

name_list      : name_list ',' name               { $$ = enlist($1, $3); }
               |               name               { $$ = {$1}; }
               ;

name           : NAME                             { $$ = $1; }
               | INT                              { $$ = $1; }
               ;

maybe_implicit : IMPLICIT                         { $$ = true; }
               | %empty                           { $$ = false; }
               ;

%%

void yy::parser::error(const parser::location_type& l, const std::string& m)
{
    throw yy::parser::syntax_error(l, m);
}
