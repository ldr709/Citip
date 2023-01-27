#ifndef __CITIP_HPP__INCLUDED__
#define __CITIP_HPP__INCLUDED__

# include <map>
# include <string>
# include <vector>
# include <variant>
# include <array>
# include <algorithm>
# include <functional>

# include "parser.hxx"


struct glp_prob;                    // defined in <glpk.h>


template<typename T>
struct SparseVectorT
{
    std::map<T, double> entries;
    bool is_equality = false;

    // get component i
    double get(const T& i) const
    {
        auto&& it = entries.find(i);
        if (it != entries.end())
            return it->second;
        return 0;
    }

    // increase/decrease component
    void inc(const T& i, double v) { entries[i] += v; }
};

template<typename T>
using MatrixT = std::vector<SparseVectorT<T>>;

typedef SparseVectorT<int> SparseVector;
typedef MatrixT<int> Matrix;


// Generic variable -- just an variable number.
struct LinearVariable {
    int id;

    friend std::ostream& operator<<(std::ostream&, const LinearVariable&);
};

// Generic rule -- just a variable that has to be nonnegative.
struct NonNegativityRule {
    int v;

    template<typename Var>
    void print(std::ostream& out, const Var* vars) const
    {
        out << vars[v] << " >= 0";
    }
};

// Either a NonNegativityRule, or some other kind of rule.
template<typename Rule>
struct NonNegativityOrOtherRule : public std::variant<NonNegativityRule, Rule> {
    typedef std::variant<NonNegativityRule, Rule> Parent;

    template<typename Var>
    void print(std::ostream& out, const Var* vars) const
    {
        return std::visit([&](auto&& rule) { return rule.print(out, vars); }, *this);
    }
};

template<typename Var=LinearVariable, typename Rule=NonNegativityRule>
struct LinearProof;

template<typename Var, typename Rule>
inline std::ostream& operator<<(std::ostream&, const LinearProof<Var, Rule>&);

template<typename Var, typename Rule>
struct LinearProof
{
    // Index 0: constant offset.
    // Indices 1 to regular_constraints.size(): regular (e.g. non-negativity) constraints.
    // Indices regular_constraints.size() + 1 to
    // regular_constraints.size() + custom_constraints.size(): custom constraints.
    SparseVector dual_solution;

    // The variables the constraints are defined over.
    std::vector<Var> variables;

    // User-defined constraints, which can't be specified using the Rule type. Each constraint is a
    // sparse vector of coefficients, representing the constraint <coefficients, variables> >= 0. Or
    // == 0, if is_equality. Index 0 in the coefficients is the constant offset.
    Matrix custom_constraints;

    // Regular constraints, which can be encoded in the Rule type.
    std::vector<Rule> regular_constraints;

    // What we ended up proving. I.e., the sum of all the constraints. Again, index 0 is constant
    // offset.
    SparseVector objective;

    bool initialized = false;

    LinearProof() = default;
    LinearProof(const LinearProof&) = default;
    LinearProof(LinearProof&&) = default;

    // Copy everything except for Vars and Rules, which are transformed by provided maps.
    template<typename Var2, typename Rule2, typename Func1, typename Func2>
    LinearProof(LinearProof<Var2, Rule2> other, Func1&& map_vars, Func2&& map_rules) :
        initialized(other.initialized),
        dual_solution(std::move(other.dual_solution)),
        custom_constraints(std::move(other.custom_constraints)),
        objective(std::move(other.objective))
    {
        if (!initialized)
            return;

        std::transform(other.variables.begin(), other.variables.end(),
                       std::back_inserter(variables), std::forward<Func1>(map_vars));
        std::transform(other.regular_constraints.begin(), other.regular_constraints.end(),
                       std::back_inserter(regular_constraints), std::forward<Func2>(map_rules));
        other.initialized = false;
    }

    inline void print_custom_constraint(std::ostream& out, const SparseVector& constraint) const;

    operator bool() const { return initialized; }
    bool operator!() const { return !(bool) *this; }

    friend std::ostream& operator<< <Var, Rule>(std::ostream&, const LinearProof<Var, Rule>&);
};

void print_coeff(std::ostream& out, double c, bool first);

template<typename Var, typename Rule>
void LinearProof<Var, Rule>::print_custom_constraint(
    std::ostream& out, const SparseVector& constraint) const
{
    double constant_offset = 0.0;
    bool first = true;
    for (const auto& [j, coeff] : constraint.entries)
    {
        if (j == 0)
        {
            constant_offset = coeff;
            continue;
        }

        print_coeff(out, coeff, first);
        first = false;
        out << variables[j - 1];
    }

    if (constraint.is_equality)
        out << " == ";
    else
        out << " >= ";
    if (constant_offset == 0.0)
        constant_offset = -0.0; // Print "0" instead of "-0"
    out << -constant_offset;
}

template<typename Var, typename Rule>
std::ostream& operator<<(std::ostream& out, const LinearProof<Var, Rule>& proof)
{
    if (!proof)
    {
        out << "FALSE";
        return out;
    }

    for (const auto& [i, dual] : proof.dual_solution.entries)
    {
        print_coeff(out, dual, false);
        if (i == 0)
            out << "(1 == 1)\n";
        else if (i <= proof.regular_constraints.size())
        {
            out << "(";
            proof.regular_constraints[i - 1].print(out, proof.variables.data() - 1);
            out << ")\n";
        }
        else
        {
            out << "(";
            int j = i - proof.regular_constraints.size() - 1;
            proof.print_custom_constraint(out, proof.custom_constraints[j]);
            out << ")\n";
        }
    }

    out << "\n => ";
    proof.print_custom_constraint(out, proof.objective);
    out << '\n';

    return out;
}

// Lightweight C++ wrapper for a GLPK problem (glp_prob*). This manages a
// problem of the form "Is I>=0 valid, subject to the constraints C>=0, and
// X>=0 for all column variables X".
class LinearProblem
{
public:
    LinearProblem();
    explicit LinearProblem(int num_cols);
    ~LinearProblem();

    void add_columns(int num_cols);

    LinearProblem(const LinearProblem&) = delete;
    LinearProblem(LinearProblem&& other) : lp(other.lp) { other.lp = NULL; }
    LinearProblem& operator = (const LinearProblem&) = delete;

    // add a constraint C>=0
    void add(const SparseVector&);

    template<typename Rule>
    inline LinearProof<LinearVariable, NonNegativityOrOtherRule<Rule>>
    prove(const SparseVector& I, const std::vector<Rule>& rules)
    {
        auto orig_proof = prove_impl(I, rules.size(), true);
        typedef LinearProof<LinearVariable, NonNegativityOrOtherRule<Rule>> OutputProof;
        if (!orig_proof)
            return OutputProof();

        OutputProof proof(
            orig_proof, std::identity(),
            [] (const NonNegativityRule& r) -> NonNegativityOrOtherRule<Rule> {
                return NonNegativityOrOtherRule<Rule>(
                    typename NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<0>(), r));
            });
        std::transform(
            rules.begin(), rules.end(), std::back_inserter(proof.regular_constraints),
            [] (const Rule& r) {
                return NonNegativityOrOtherRule<Rule>(
                    typename NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<1>(), r));
            });
        return proof;
    }

    // check if I>=0 is redundant
    inline bool check(const SparseVector& I)
    {
        return prove_impl(I, 0, false);
    }

protected:
    LinearProof<> prove_impl(const SparseVector& I, int num_regular_rules, bool want_proof);

    glp_prob* lp;
};

typedef std::array<int, 3> CmiTriplet;

struct ShannonVar {
    const std::vector<std::string>& random_var_names;

    struct PrintVarsOut {
        const ShannonVar& parent;

        friend std::ostream& operator<<(std::ostream&, const PrintVarsOut& out);
    };

    PrintVarsOut print_vars() const { return PrintVarsOut{*this}; }

    int v;
    friend std::ostream& operator<<(std::ostream&, const ShannonVar&);
};

struct ShannonRule : public CmiTriplet {
    void print(std::ostream&, const ShannonVar* vars) const;
};

struct ExtendedShannonVar : public CmiTriplet {
    const std::vector<std::string>* random_var_names = NULL;

    friend std::ostream& operator<<(std::ostream&, const ExtendedShannonVar&);
};

// Linear (in)equalities, other than I(a;b|z) >= 0.
struct ExtendedShannonRule
{
    enum type_enum
    {
        CMI_DEF_I,
        MI_DEF_I,
        MI_DEF_CI,

        CHAIN,
        MUTUAL_CHAIN,

        MONOTONE,
    };

    type_enum type;
    std::array<int, 4> subsets;

    friend auto operator<=>(const ExtendedShannonRule& a, const ExtendedShannonRule& b) = default;

    bool is_equality() const
    {
        return type != MONOTONE;
    }

    double complexity_cost() const;

    void print(std::ostream& out, const ExtendedShannonVar* vars) const;
};

typedef LinearProof<ExtendedShannonVar, NonNegativityOrOtherRule<ExtendedShannonRule>>
    SimplifiedShannonProof;
struct ShannonTypeProof : public LinearProof<ShannonVar, ShannonRule>
{
    typedef LinearProof<ShannonVar, ShannonRule> Parent;
    using Parent::Parent;

    SimplifiedShannonProof simplify() const;

    // Save these in case simplify() is run.
    MatrixT<CmiTriplet> cmi_constraints;
    SparseVectorT<CmiTriplet> cmi_objective;
};

class ParserOutput;

// Manage a linear programming problem in the context of random variables.
// The system has 2**num_vars-1 random variables which correspond to joint
// entropies of the non-empty subsets of random variables. These quantities
// are indexed in a canonical way, such that the bit-representation of the
// index is in one-to-one correspondence with the subset.
class ShannonTypeProblem
    : public LinearProblem
{
public:
    explicit ShannonTypeProblem(std::vector<std::string> random_var_names_);
    ShannonTypeProblem(const ParserOutput&);

    void add(const SparseVector&, SparseVectorT<CmiTriplet>);
    ShannonTypeProof prove(const SparseVector& I, SparseVectorT<CmiTriplet> cmi_I);

protected:
    void add_elemental_inequalities(glp_prob* lp, int num_vars);

    MatrixT<CmiTriplet> cmi_constraints;

    const std::vector<std::string> random_var_names;
    std::vector<CmiTriplet> row_to_cmi;
};


class ParserOutput : public ParserCallback
{
    int get_var_index(const std::string&);
    int get_set_index(const ast::VarList&);     // as in 'set of variables'
    void add_term(SparseVector&, SparseVectorT<CmiTriplet>&, const ast::Term&, double scale=1);

    std::map<std::string, int> vars;

    void add_relation(SparseVector, SparseVectorT<CmiTriplet>, bool is_inquiry);
public:
    // consider this read-only
    std::vector<std::string> var_names;

    Matrix inquiries;
    Matrix constraints;

    MatrixT<CmiTriplet> cmi_constraints;
    MatrixT<CmiTriplet> cmi_inquiries;

    // parser callback
    void relation(ast::Relation);
    void markov_chain(ast::MarkovChain);
    void mutual_independence(ast::MutualIndependence);
    void function_of(ast::FunctionOf);
};


ParserOutput parse(const std::vector<std::string>&);


#endif // include guard
