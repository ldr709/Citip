#ifndef __AST_HPP__INCLUDED__
#define __AST_HPP__INCLUDED__

// This file defines the AST (abstract syntax tree) for the Citip grammar,
// i.e. the data structures that hold all the information from the parsed
// input statements.

# include <string>
# include <variant>
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

    struct EntropyQuantity
    {
        VarCore parts;
        VarList cond;
    };

    struct VariableQuantity
    {
        std::string name;
    };

    struct ConstantQuantity {};

    typedef std::variant<EntropyQuantity, VariableQuantity, ConstantQuantity> Quantity;

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

    struct MutualIndependence
    {
        VarList scenarios;
        std::vector<VarList> lists;
        bool implicit;
    };

    struct MarkovChain {
        VarList scenarios;
        std::vector<VarList> lists;
    };

    struct FunctionOf {
        VarList scenarios;
        VarList function, of;
        bool implicit;
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
