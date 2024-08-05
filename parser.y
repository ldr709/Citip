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
%token <int>            REL
%token <int>            FUNCTION_OF
%token <int>            OPT_COEFF
%token <int>            APPROX
%token <int>            INDIST
%token <int>            IMPLICIT
%token <int>            PROVE

%type <ast::Relation>                       inequality
%type <ast::MutualIndependence>             mutual_indep
%type <std::vector<ast::VarList>>           mutual_indep_list
%type <ast::MarkovChain>                    markov_chain
%type <std::vector<ast::VarList>>           markov_chain_list
%type <ast::FunctionOf>                     determ_depen
%type <std::vector<ast::Expression>>        expr_list
%type <ast::Expression>                     expr
%type <ast::Term>                           term
%type <ast::Quantity>                       quant
%type <ast::TargetRelation>                 target
%type <ast::TargetExpression>               target_expr
%type <ast::TargetTerm>                     target_term
%type <ast::TargetCoeff>                    target_coeff
%type <ast::EntropyQuantity>                entropy
%type <ast::EntropyQuantity>                mutual_inf
%type <ast::VarList>                        name_list
%type <std::vector<ast::VarList>>           mut_inf_core;
%type <std::string>                         name;
%type <ast::IndistinguishableScenarios>     indist;
%type <ast::IndistinguishableScenariosList> indist_list;
%type <ast::IndistinguishableScenario>      indist_item;
%type <ast::BoundOrImplicit>                bound_or_implicit;
%type <ast::Expression>                     maybe_bound;
%type <std::vector<ast::Expression>>        indist_bound;

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
        virtual void target(ast::TargetRelation) = 0;
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
             | target           { cb->target(move($1)); }
             | inequality       { cb->relation(move($1)); }
             | mutual_indep     { cb->mutual_independence(move($1)); }
             | markov_chain     { cb->markov_chain(move($1)); }
             | determ_depen     { cb->function_of(move($1)); }
             | indist           { cb->indist(move($1)); }
             ;

    /* statements */

target            : PROVE target_expr REL target_expr                            { $$ = {$2, $3, $4}; }
                  ;

inequality        : expr REL expr                                                { $$ = {$1, $2, $3}; }
                  ;

markov_chain      : markov_chain_list maybe_bound                                { $$ = {{}, $1, $2}; }
                  | name_list ':' markov_chain_list maybe_bound                  { $$ = {$1, $3, $4}; }
                  ;

markov_chain_list : markov_chain_list '/' name_list                              { $$ = enlist($1, $3); }
                  | name_list '/' name_list '/' name_list                        { $$ = {$1, $3, $5}; }
                  ;

mutual_indep      : mutual_indep_list bound_or_implicit                          { $$ = {{}, $1, $2}; }
                  | name_list ':' mutual_indep_list bound_or_implicit            { $$ = {$1, $3, $4}; }
                  ;

mutual_indep_list : mutual_indep_list '.' name_list                              { $$ = enlist($1, $3); }
                  |         name_list '.' name_list                              { $$ = {$1, $3}; }
                  ;

determ_depen      : name_list FUNCTION_OF name_list bound_or_implicit            { $$ = {{}, $1, $3, $4}; }
                  | name_list ':' name_list '<' '-' name_list bound_or_implicit  { $$ = {$1, $3, $6, $7}; }
                  ;

indist            : INDIST indist_list indist_bound                              { $$ = {$2, $3}; }
                  | INDIST name_list indist_bound                                { $$ = {{{{}, $2}}, $3}; }
                  | INDIST ':' name_list indist_bound                            { $$ = {{{{}, $3}}, $4}; }
                  ;

indist_list       : indist_list ';' indist_item                                  { $$ = enlist($1, $3); }
                  | indist_item                                                  { $$ = {$1}; }
                  ;

indist_item       : name_list ':' name_list                                      { $$ = {$1, $3}; }
                  ;

    /* building blocks */

expr_list      : expr_list ',' expr               { $$ = enlist($1, $3); }
               | expr                             { $$ = {$1}; }
               ;

expr           : expr SIGN term                   { $$ = enlist($1, $3.flip_sign($2)); }
               |      SIGN term                   { $$ = {$2.flip_sign($1)}; }
               |           term                   { $$ = {$1}; }
               ;

target_expr    : target_expr SIGN target_term     { $$ = enlist($1, $3.flip_sign($2)); }
               |             SIGN target_term     { $$ = {$2.flip_sign($1)}; }
               |                  target_term     { $$ = {$1}; }
               ;

term           : INT quant                        { $$ = {std::stod($1), $2}; }
               | NUM quant                        { $$ = {$1, $2}; }
               |     quant                        { $$ = { 1, $1}; }
               | INT                              { $$ = {std::stod($1), ast::ConstantQuantity{}}; }
               | NUM                              { $$ = {$1, ast::ConstantQuantity{}}; }
               ;

target_term    : target_coeff quant               { $$ = {$1, $2}; }
               |              quant               { $$ = {{1, {}}, $1}; }
               | target_coeff                     { $$ = {$1, ast::ConstantQuantity{}}; }
               ;

target_coeff   : INT OPT_COEFF                    { $$ = {std::stod($1), $2}; }
               | NUM OPT_COEFF                    { $$ = {$1, $2}; }
               |     OPT_COEFF                    { $$ = { 1, $1}; }
               | INT                              { $$ = {std::stod($1), {}}; }
               | NUM                              { $$ = {$1, {}}; }
               ;

quant          : entropy                          { $$ = $1; }
               | mutual_inf                       { $$ = $1; }
               | NAME                             { $$ = ast::VariableQuantity{$1}; }
               ;

entropy        : ENTROPY '(' name_list              ')'      { $$ = {{$1, {$3}}}; }
               | ENTROPY '(' name_list '|' name_list ')'      { $$ = {{$1, {$3}}, $5}; }
               ;

mutual_inf     : MUTUAL_INFORMATION '(' mut_inf_core              ')'  { $$ = {{$1, $3}}; }
               | MUTUAL_INFORMATION '(' mut_inf_core '|' name_list ')'  { $$ = {{$1, $3}, $5}; }
               ;

mut_inf_core   :  mut_inf_core ';' name_list      { $$ = enlist($1, $3); }
               |      name_list ';' name_list     { $$ = {$1, $3}; }
               ;

name_list      : name_list ',' name               { $$ = enlist($1, $3); }
               |               name               { $$ = {$1}; }
               ;

name           : NAME                             { $$ = $1; }
               | INT                              { $$ = $1; }
               ;

bound_or_implicit : IMPLICIT                      { $$ = {}; }
                  | maybe_bound                   { $$ = {$1}; }
                  ;

maybe_bound    : APPROX expr                      { $$ = $2; }
               | %empty                           { $$ = {}; }
               ;

indist_bound   : APPROX expr_list                 { $$ = $2; }
               | %empty                           { $$ = {}; }
               ;

%%

void yy::parser::error(const parser::location_type& l, const std::string& m)
{
    throw yy::parser::syntax_error(l, m);
}
