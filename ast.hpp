#ifndef __AST_HPP__INCLUDED__
#define __AST_HPP__INCLUDED__

// This file defines the AST (abstract syntax tree) for the Citip grammar,
// i.e. the data structures that hold all the information from the parsed
// input statements.

# include <string>
# include <vector>


namespace ast
{

    enum {
        SIGN_PLUS,
        SIGN_MINUS
    };

    enum {
        REL_EQ,
        REL_LE,
        REL_GE
    };

    typedef std::vector<std::string>    VarList;

    struct VarCore
    {
        std::string scenario;
        std::vector<VarList> lists;
    };

    struct Quantity
    {
        VarCore parts;
        VarList cond;
    };

    struct Term
    {
        double coefficient;
        Quantity quantity;

        inline Term& flip_sign(int s)
        {
            if (s == SIGN_MINUS) {
                coefficient = -coefficient;
            }
            return *this;
        }
    };

    typedef std::vector<Term> Expression;

    struct Relation {
        Expression left;
        int relation;
        Expression right;
    };

    typedef VarCore MutualIndependence;
    typedef VarCore MarkovChain;

    struct FunctionOf {
        std::string scenario;
        VarList function, of;
    };

    // Specifies that each view (a collection of random variables) must be indistinguishable across
    // all scenarios in the list. If scenarios is empty, this instead counts as every scenario.
    struct IndistinguishableScenario {
        std::vector<std::string> scenarios;
        VarList view;
    };
    typedef std::vector<IndistinguishableScenario> IndistinguishableScenarios;
}

#endif // include-guard
