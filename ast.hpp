#ifndef __AST_HPP__INCLUDED__
#define __AST_HPP__INCLUDED__

// This file defines the AST (abstract syntax tree) for the Citip grammar,
// i.e. the data structures that hold all the information from the parsed
// input statements.

# include <string>
# include <variant>
# include <vector>
# include <optional>


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
            if (s == SIGN_MINUS)
                coefficient = -coefficient;
            return *this;
        }
    };

    struct TargetCoeff
    {
        double scalar;
        std::optional<int> optimize_coeff_var;

        inline TargetCoeff& flip_sign(int s)
        {
            if (s == SIGN_MINUS)
                scalar = -scalar;
            return *this;
        }
    };

    struct TargetTerm
    {
        TargetCoeff coefficient;
        Quantity quantity;

        inline TargetTerm& flip_sign(int s)
        {
            coefficient.flip_sign(s);
            return *this;
        }
    };

    typedef std::vector<Term> Expression;
    typedef std::vector<TargetTerm> TargetExpression;

    typedef std::optional<ast::Expression> BoundOrImplicit;

    struct Relation {
        Expression left;
        int relation;
        Expression right;
    };

    struct TargetRelation {
        TargetExpression left;
        int relation;
        TargetExpression right;
    };

    struct MutualIndependence
    {
        VarList scenarios;
        std::vector<VarList> lists;
        BoundOrImplicit bound_or_implicit;

        bool implicit() const { return !bound_or_implicit; }
    };

    struct MarkovChain {
        VarList scenarios;
        std::vector<VarList> lists;

        Expression bound;
    };

    struct FunctionOf {
        VarList scenarios;
        VarList function, of;
        BoundOrImplicit bound_or_implicit;

        bool implicit() const { return !bound_or_implicit; }
    };

    // Specifies that each view (a collection of random variables) must be indistinguishable across
    // all scenarios in the list. If scenarios is empty, this instead counts as every scenario.
    struct IndistinguishableScenario {
        std::vector<std::string> scenarios;
        VarList view;
    };

    typedef std::vector<IndistinguishableScenario> IndistinguishableScenariosList;

    struct ApproxGroup;

    struct IndistinguishableScenarios {
        IndistinguishableScenariosList indist_scenarios;
        std::vector<ApproxGroup> bounds;
    };

    struct ApproxGroup
    {
        Expression bound;
        unsigned int count;

        void inc_group() { ++count; }
    };
}

#endif // include-guard
