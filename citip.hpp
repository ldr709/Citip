#ifndef __CITIP_HPP__INCLUDED__
#define __CITIP_HPP__INCLUDED__

# include <map>
# include <string>
# include <vector>
# include <variant>
# include <array>
# include <algorithm>
# include <functional>
# include <memory>

# include <coin/CoinPackedMatrix.hpp>

# include "parser.hxx"

// https://www.cppstories.com/2019/02/2lines3featuresoverload.html/
template<class... Ts> struct overload : Ts... { using Ts::operator()...; };

class OsiSolverInterface;

// Coin osi problem, before it's given to a solver.
struct CoinOsiProblem {
    CoinPackedMatrix constraints;
    std::vector<double> collb, colub;
    std::vector<double> rowlb, rowub;
    std::vector<double> obj;
    double infinity = 0.0;

    int num_rows = 0;
    int num_cols = 0;

    CoinOsiProblem(bool colmajor = true) : constraints(colmajor, 2.0, 2.0) {}
    CoinOsiProblem(OsiSolverInterface& solver, bool colmajor = true) :
        CoinOsiProblem(colmajor) { setup(solver); }
    void setup(OsiSolverInterface& solver);

    void load_problem_into_solver(OsiSolverInterface& solver);

    int add_row_lb(double lb, int count = 0, int* indices = nullptr,
                   double* values = nullptr)
    {
        return add_row(lb, infinity, count, indices, values);
    }
    int add_row_ub(double ub, int count = 0, int* indices = nullptr,
                   double* values = nullptr)
    {
        return add_row(-infinity, ub, count, indices, values);
    }
    int add_row_fixed(double rhs, int count = 0, int* indices = nullptr,
                      double* values = nullptr)
    {
        return add_row(rhs, rhs, count, indices, values);
    }
    int add_col_lb(double lb, double obj_ = 0.0, int count = 0, int* indices = nullptr,
                   double* values = nullptr)
    {
        return add_col(lb, infinity, obj_, count, indices, values);
    }
    int add_col_ub(double ub, double obj_ = 0.0, int count = 0, int* indices = nullptr,
                   double* values = nullptr)
    {
        return add_col(-infinity, ub, obj_, count, indices, values);
    }
    int add_col_free(double obj_ = 0.0, int count = 0, int* indices = nullptr,
                     double* values = nullptr)
    {
        return add_col(-infinity, infinity, obj_, count, indices, values);
    }

    int add_row(double lb, double ub,
                int count = 0, int* indices = nullptr, double* values = nullptr);
    int add_col(double lb, double ub, double obj_ = 0.0,
                int count = 0, int* indices = nullptr, double* values = nullptr);

    operator bool() const { return infinity != 0.0; }

private:
    static void append_with_default(std::vector<double>& vec, int size, double val, double def);
};

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

void print_coeff(std::ostream& out, double c, bool first);

// Generic rule -- just a variable that has to be nonnegative.
struct NonNegativityRule {
    int v;

    template<typename Var>
    void print(std::ostream& out, const Var* vars, double scale = 1.0) const
    {
        print_coeff(out, scale, true);
        out << vars[v] << " >= 0";
    }
};

// Either a NonNegativityRule, or some other kind of rule.
template<typename Rule>
struct NonNegativityOrOtherRule : public std::variant<NonNegativityRule, Rule> {
    typedef std::variant<NonNegativityRule, Rule> Parent;

    template<typename Var>
    void print(std::ostream& out, const Var* vars, double scale = 1.0) const
    {
        return std::visit([&](auto&& rule) { return rule.print(out, vars, scale); }, *this);
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

    inline void print_custom_constraint(std::ostream& out, const SparseVector& constraint,
                                        double scale = 1.0) const;

    operator bool() const { return initialized; }
    bool operator!() const { return !(bool) *this; }

    inline void print_step(std::ostream& out, int step, double dual_coeff) const;

    friend std::ostream& operator<< <Var, Rule>(std::ostream&, const LinearProof<Var, Rule>&);
};

template<typename Var, typename Rule>
void LinearProof<Var, Rule>::print_custom_constraint(
    std::ostream& out, const SparseVector& constraint, double scale) const
{
    double constant_offset = 0.0;
    bool first = true;
    for (const auto& [j, coeff] : constraint.entries)
    {
        if (j == 0)
        {
            constant_offset = coeff * scale;
            continue;
        }

        print_coeff(out, coeff * scale, first);
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
        proof.print_step(out, i, dual);

    out << "\n => ";
    proof.print_custom_constraint(out, proof.objective);
    out << '\n';

    return out;
}

template<typename Var, typename Rule>
void LinearProof<Var, Rule>::print_step(std::ostream& out, int step, double dual_coeff) const
{
    if (step == 0)
        out << "0 >= " << -dual_coeff << "\n";
    else if (step <= regular_constraints.size())
    {
        regular_constraints[step - 1].print(out, variables.data() - 1, dual_coeff);
        out << '\n';
    }
    else
    {
        int j = step - regular_constraints.size() - 1;
        print_custom_constraint(out, custom_constraints[j], dual_coeff);
        out << '\n';
    }
}

// Lightweight C++ wrapper for a GLPK problem (glp_prob*). This manages a
// problem of the form "Is I>=0 valid, subject to the constraints C>=0, and
// X>=0 for all column variables X".
class LinearProblem
{
public:
    LinearProblem();
    ~LinearProblem();
    explicit LinearProblem(int num_cols);

    void add_columns(int num_cols);

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

    std::unique_ptr<OsiSolverInterface> si;
    CoinOsiProblem coin;
};

struct CmiTriplet :
    public std::array<int, 3>
{
    int scenario;

    CmiTriplet() = default;
    CmiTriplet(int a, int b, int c, int scenario_) :
        std::array<int, 3>{a, b, c},
        scenario(scenario_)
    {
        auto& t = *this;
        t[0] &= ~t[2];
        t[1] &= ~t[2];

        if (t[0] > t[1])
            std::swap(t[0], t[1]);

        if ((t[0] | t[1]) == t[1])
            t[1] = t[0];
    }

    friend auto operator<=>(const CmiTriplet& a, const CmiTriplet& b) = default;
};

namespace std
{
    template<>
    struct tuple_size<CmiTriplet> : public std::integral_constant<std::size_t, 3> {};

    template<std::size_t I>
    struct tuple_element<I, CmiTriplet>
    {
        static_assert(I < 3);
        typedef int type;
    };
}

struct ShannonVar {
    const std::vector<std::string>& random_var_names;
    const std::vector<std::string>& scenario_names;

    struct PrintVarsOut {
        const ShannonVar& parent;

        friend std::ostream& operator<<(std::ostream&, const PrintVarsOut& out);
    };

    PrintVarsOut print_vars() const { return PrintVarsOut{*this}; }
    const std::string& scenario() const;

    int v;
    friend std::ostream& operator<<(std::ostream&, const ShannonVar&);
};

struct ShannonRule : public CmiTriplet {
    void print(std::ostream&, const ShannonVar* vars, double scale = 1.0) const;
};

struct ExtendedShannonVar : public CmiTriplet {
    const std::vector<std::string>* random_var_names = nullptr;
    const std::vector<std::string>* scenario_names = nullptr;

    friend std::ostream& operator<<(std::ostream&, ExtendedShannonVar);
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
    int scenario;

    friend auto operator<=>(const ExtendedShannonRule& a, const ExtendedShannonRule& b) = default;

    bool is_equality() const
    {
        return type != MONOTONE;
    }

    int get_constraint(CmiTriplet indices[], double values[]) const;
    double complexity_cost() const;

    void print(std::ostream& out, const ExtendedShannonVar* vars, double scale = 1.0) const;
};

struct OrderedSimplifiedShannonProof;

struct SimplifiedShannonProof :
    public LinearProof<ExtendedShannonVar, NonNegativityOrOtherRule<ExtendedShannonRule>>
{
    typedef LinearProof<ExtendedShannonVar, NonNegativityOrOtherRule<ExtendedShannonRule>> Parent;
    using Parent::Parent;

    OrderedSimplifiedShannonProof order() const;
};

struct ShannonTypeProof : public LinearProof<ShannonVar, ShannonRule>
{
    typedef LinearProof<ShannonVar, ShannonRule> Parent;
    using Parent::Parent;

    SimplifiedShannonProof simplify() const;

    // Save these in case simplify() is run.
    MatrixT<CmiTriplet> cmi_constraints;
    SparseVectorT<CmiTriplet> cmi_objective;
};

struct OrderedSimplifiedShannonProof : public SimplifiedShannonProof
{
    std::vector<int> order;

    friend std::ostream& operator<<(std::ostream&, const OrderedSimplifiedShannonProof&);
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
    ShannonTypeProblem(std::vector<std::string> random_var_names_,
                       std::vector<std::string> scenario_names_);
    ShannonTypeProblem(const ParserOutput&);

    void add(const SparseVector&, SparseVectorT<CmiTriplet>);
    ShannonTypeProof prove(const SparseVector& I, SparseVectorT<CmiTriplet> cmi_I);

protected:
    void add_elemental_inequalities(int num_vars, int num_scenarios);

    MatrixT<CmiTriplet> cmi_constraints;

    const std::vector<std::string> random_var_names;
    const std::vector<std::string> scenario_names;
    std::vector<CmiTriplet> row_to_cmi;
};


class ParserOutput : public ParserCallback
{
    int get_var_index(const std::string&);
    int get_set_index(const ast::VarList&);     // as in 'set of variables'
    void add_term(SparseVector&, SparseVectorT<CmiTriplet>&, const ast::Term&,
                  int scenario_wildcard, double scale);

    std::map<std::string, int> vars;
    std::map<std::string, int> scenarios;

    enum statement_type
    {
        RELATION = 0,
        MARKOV_CHAIN,
        MUTUAL_INDEPENDENCE,
        FUNCTION_OF,
        INDISTINGUISHABLE_SCENARIOS,
    };
    typedef std::variant<ast::Relation, ast::MarkovChain, ast::MutualIndependence,
                         ast::FunctionOf, ast::IndistinguishableScenarios> statement;
    std::vector<statement> statement_list;

    void add_relation(SparseVector, SparseVectorT<CmiTriplet>, bool is_inquiry);

    void add_scenario(const std::string&);
    void add_symbols(const ast::VarList&);

    void process_statement(const statement& s);
    void process_relation(const ast::Relation&);
    void process_markov_chain(const ast::MarkovChain&);
    void process_mutual_independence(const ast::MutualIndependence&);
    void process_function_of(const ast::FunctionOf&);
    void process_indist(const ast::IndistinguishableScenarios&);

public:
    // consider this read-only
    std::vector<std::string> var_names;
    std::vector<std::string> scenario_names;

    Matrix inquiries;
    Matrix constraints;

    MatrixT<CmiTriplet> cmi_constraints;
    MatrixT<CmiTriplet> cmi_inquiries;

    void process();

    // parser callback
    virtual void relation(ast::Relation) override;
    virtual void markov_chain(ast::MarkovChain) override;
    virtual void mutual_independence(ast::MutualIndependence) override;
    virtual void function_of(ast::FunctionOf) override;
    virtual void indist(ast::IndistinguishableScenarios) override;
};


ParserOutput parse(const std::vector<std::string>&);


#endif // include guard
