#ifndef __CITIP_HPP__INCLUDED__
#define __CITIP_HPP__INCLUDED__

# include <map>
# include <string>
# include <vector>
# include <variant>
# include <array>
# include <algorithm>
# include <functional>
# include <fstream>
# include <memory>

# include <coin/CoinPackedMatrix.hpp>

# include "parser.hxx"

constexpr double eps = 3e-4;

// https://www.cppstories.com/2019/02/2lines3featuresoverload.html/
template<class... Ts> struct overload : Ts... { using Ts::operator()...; };

class OsiSolverInterface;
class OsiClpSolverInterface;

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
    CoinOsiProblem(const OsiSolverInterface& solver, bool colmajor = true) :
        CoinOsiProblem(colmajor) { setup(solver); }
    void setup(const OsiSolverInterface& solver);

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
    void inc(const T& i, double v)
    {
        auto [it, inserted] = entries.insert({i, v});
        if (inserted)
            return;

        it->second += v;
        if (it->second == 0.0)
            entries.erase(it);
    }
};

template<typename T>
using MatrixT = std::vector<SparseVectorT<T>>;

typedef SparseVectorT<int> SparseVector;
typedef MatrixT<int> Matrix;


struct ImplicitFunctionOf
{
    int func;
    int of;
};

struct ImplicitIndependence
{
    int set;
    int indep_from;
};

struct ImplicitRules {
    std::vector<ImplicitFunctionOf> funcs;
    std::vector<ImplicitIndependence> indeps;
};

// Generic variable -- just an variable number.
struct LinearVariable {
    int id;

    bool is_zero() const { return false; }

    friend std::ostream& operator<<(std::ostream&, const LinearVariable&);
};

void print_coeff(std::ostream& out, double c, bool first);

// Generic rule -- just a variable that has to be nonnegative.
struct NonNegativityRule {
    int v;

    template<typename Var>
    bool print(std::ostream& out, const Var* vars, double scale = 1.0) const
    {
        if (scale == 0.0 || vars[v].is_zero())
            return false;

        print_coeff(out, scale, true);
        out << vars[v] << " >= 0";
        return true;
    }
};

// Either a NonNegativityRule, or some other kind of rule.
template<typename Rule>
struct NonNegativityOrOtherRule : public std::variant<NonNegativityRule, Rule> {
    typedef std::variant<NonNegativityRule, Rule> Parent;

    template<typename Var>
    bool print(std::ostream& out, const Var* vars, double scale = 1.0) const
    {
        return std::visit([&](const auto& rule) { return rule.print(out, vars, scale); }, *this);
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

    // Values of the variables for the optimal solution. Might be helpful for proof simplification.
    std::vector<double> primal_solution;

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
        primal_solution(std::move(other.primal_solution)),
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

        if (false)
        {
            std::ofstream solution_file("primal_solution.txt");
            for (int i = 0; i < variables.size(); ++i)
                solution_file << variables[i] << ": " << primal_solution[i] << '\n';
        }
    }

    inline bool print_custom_constraint(std::ostream& out, const SparseVector& constraint,
                                        double scale = 1.0) const;

    operator bool() const { return initialized; }
    bool operator!() const { return !(bool) *this; }

    inline void print_step(std::ostream& out, int step, double dual_coeff) const;

    friend std::ostream& operator<< <Var, Rule>(std::ostream&, const LinearProof<Var, Rule>&);
};

template<typename Var, typename Rule>
bool LinearProof<Var, Rule>::print_custom_constraint(
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

        const auto& v = variables[j - 1];
        if (scale == 0.0 || coeff == 0.0 || v.is_zero())
            continue;

        print_coeff(out, coeff * scale, first);
        first = false;
        out << v;
    }

    if (first)
        return false;

    if (constraint.is_equality)
        out << " == ";
    else
        out << " >= ";
    if (constant_offset == 0.0)
        constant_offset = -0.0; // Print "0" instead of "-0"
    out << -constant_offset;
    return true;
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
        if (std::abs(dual) > eps)
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
        if (regular_constraints[step - 1].print(out, variables.data(), dual_coeff))
            out << '\n';
    }
    else
    {
        int j = step - regular_constraints.size() - 1;
        if (print_custom_constraint(out, custom_constraints[j], dual_coeff))
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

    // Find the smallest that I can be
    inline std::optional<double> optimize(const SparseVector& I)
    {
        auto proof = prove_impl(I, 0, false, false);
        if (!proof)
            return {};
        return proof.dual_solution.get(0);
    }

protected:
    LinearProof<> prove_impl(const SparseVector& I, int num_regular_rules,
                             bool want_proof, bool check_bound = true);

    std::unique_ptr<OsiClpSolverInterface> si;
    CoinOsiProblem coin;
};

struct CmiTriplet :
    public std::array<int, 3>
{
    int scenario;

    CmiTriplet() = default;
    CmiTriplet(int a, int b, int c, int scenario_) :
        CmiTriplet(ImplicitRules(), a, b, c, scenario_) {}

    CmiTriplet(const ImplicitRules& implicits,
               int a, int b, int c, int scenario_);

    bool is_zero() const;
    double complexity_cost() const;

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
    const std::vector<std::vector<std::string>>& var_names_by_scenario;
    const std::vector<std::string>& scenario_names;
    const std::vector<std::string>& real_var_names;
    const std::map<int, int>& column_map;
    const std::vector<ImplicitRules>& implicits_by_scenario;

    struct PrintVarsOut {
        const ShannonVar& parent;

        friend std::ostream& operator<<(std::ostream&, const PrintVarsOut& out);
    };

    PrintVarsOut print_vars() const { return PrintVarsOut{*this}; }
    const std::string& scenario() const;

    bool is_zero() const { return v == 0; }

    int v;
    friend std::ostream& operator<<(std::ostream&, const ShannonVar&);
};

struct ShannonRule : public CmiTriplet {
    bool print(std::ostream&, const ShannonVar* vars, double scale = 1.0) const;
};

struct ExtendedShannonVar : public CmiTriplet {
    const std::vector<std::vector<std::string>>* var_names_by_scenario = nullptr;
    const std::vector<std::string>* scenario_names = nullptr;
    const std::vector<std::string>* real_var_names = nullptr;
    const ImplicitRules* implicits = nullptr;

    // For if this is a real variable instead of a CMI of random variables.
    int real_var = -1;

    operator std::variant<CmiTriplet, int>() const;

    bool is_zero() const;
    friend std::ostream& operator<<(std::ostream&, ExtendedShannonVar);
};

// Linear (in)equalities, other than I(a;b|z) >= 0.
struct ExtendedShannonRule
{
    enum type_enum
    {
        CMI_DEF_I,

        CHAIN,
        MUTUAL_CHAIN,

        MONOTONE_COND,
        MONOTONE_MUT,
    };

    ExtendedShannonRule() = default;
    ExtendedShannonRule(type_enum type, int z, int a, int b, int c, int scenario_) :
        ExtendedShannonRule(ImplicitRules(), type, z, a, b, c, scenario_) {}

    ExtendedShannonRule(const ImplicitRules& implicits, type_enum type_,
                        int z, int a, int b, int c, int scenario_);

    type_enum type;
    std::array<int, 4> subsets;
    int scenario;

    friend auto operator<=>(const ExtendedShannonRule& a, const ExtendedShannonRule& b) = default;

    bool is_equality() const
    {
        return type != MONOTONE_COND && type != MONOTONE_MUT;
    }

    bool is_trivial(const SparseVectorT<CmiTriplet>& c) const;
    bool is_trivial(const ImplicitRules& implicits) const;

    SparseVectorT<CmiTriplet> get_constraint(const ImplicitRules& implicits) const;
    double complexity_cost(const ImplicitRules& implicits) const;

    bool print(std::ostream& out, const ExtendedShannonVar* vars, double scale = 1.0) const;
};

struct OrderedSimplifiedShannonProof;

struct SimplifiedShannonProof :
    public LinearProof<ExtendedShannonVar, NonNegativityOrOtherRule<ExtendedShannonRule>>
{
    typedef LinearProof<ExtendedShannonVar, NonNegativityOrOtherRule<ExtendedShannonRule>> Parent;
    using Parent::Parent;

    typedef std::variant<CmiTriplet, int> Symbol;

    OrderedSimplifiedShannonProof order() const;

    const std::vector<ImplicitRules>* implicits_by_scenario = nullptr;
};

struct ShannonTypeProof : public LinearProof<ShannonVar, ShannonRule>
{
    typedef LinearProof<ShannonVar, ShannonRule> Parent;
    using Parent::Parent;

    typedef SimplifiedShannonProof::Symbol Symbol;

    SimplifiedShannonProof simplify(int depth) const;

    // Save these in case simplify() is run.
    MatrixT<Symbol> cmi_constraints;
    MatrixT<Symbol> cmi_constraints_redundant;
    SparseVectorT<Symbol> cmi_objective;
};

struct OrderedSimplifiedShannonProof : public SimplifiedShannonProof
{
    std::vector<int> order;

    friend std::ostream& operator<<(std::ostream&, const OrderedSimplifiedShannonProof&);
};

class ParserOutput;

// Manage a linear programming problem in the context of random variables.
class ShannonTypeProblem
    : public LinearProblem
{
public:
    typedef ShannonTypeProof::Symbol Symbol;

    ShannonTypeProblem(std::vector<std::vector<std::string>> var_names_by_scenario_,
                       std::vector<std::string> scenario_names_,
                       std::vector<std::string> real_var_names_,
                       std::vector<int> opt_coeff_var_names_,
                       std::vector<ImplicitRules> implicits_by_scenario,
                       MatrixT<Symbol> cmi_constraints_redundant_);
    ShannonTypeProblem(const ParserOutput&);

    void add(const SparseVector&);
    void add(const SparseVector&, SparseVectorT<Symbol>);
    ShannonTypeProof prove(Matrix I, MatrixT<Symbol> cmi_I);

protected:
    void add_columns();
    using LinearProblem::add_columns;
    using LinearProblem::prove;
    using LinearProblem::check;
    using LinearProblem::add;
    void add_elemental_inequalities();

    MatrixT<Symbol> cmi_constraints;
    MatrixT<Symbol> cmi_constraints_redundant;
    std::vector<ImplicitRules> implicits_by_scenario;

    std::map<int, int> column_map;
    std::vector<int> inv_column_map;
    int one_var;

    const std::vector<std::string> scenario_names;
    const std::vector<std::vector<std::string>> var_names_by_scenario;

    // Real valued variables used in the linear inequalities, not the random variables above.
    const std::vector<std::string> real_var_names;

    std::vector<int> opt_coeff_var_names;

    std::vector<CmiTriplet> row_to_cmi;
};


class ParserOutput : public ParserCallback
{
    typedef ShannonTypeProof::Symbol Symbol;

    int get_var_index(int scenario, const std::string&);
    int get_real_var_index(const std::string&);
    int get_set_index(int scenario, const ast::VarList&);     // as in 'set of variables'
    void add_term(SparseVector&, SparseVectorT<Symbol>&, const ast::Term&,
                  int scenario_wildcard, double scale);
    void add_quantity(SparseVector&, SparseVectorT<Symbol>&, const ast::Quantity&,
                      int scenario_wildcard, double scale);

    std::map<std::tuple<int, std::string>, int> vars_by_scenario;
    std::map<std::string, int> scenarios;

    // Real valued variables used in the linear inequalities, not the random variables above.
    std::map<std::string, int> real_vars;

    std::map<int, int> opt_coeff_vars;

    std::tuple<int, int> scenario_range(const std::string& scenario) const;
    const std::vector<std::string>& scenario_list(const ast::VarList& scenarios) const;

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

    std::optional<ast::TargetRelation> target_ast;

    void add_scenario(const std::string&);
    void add_scenarios(const ast::VarList&);
    void add_vars(int scenario, const ast::VarList&);
    void add_real_var(const std::string&);
    void add_cmi(SparseVector& v, SparseVectorT<Symbol>& cmi_v,
                 CmiTriplet t, double coeff) const;

    void process_statement(const statement& s);
    void process_relation(const ast::Relation&);
    void process_markov_chain(const ast::MarkovChain&);
    void process_mutual_independence(const ast::MutualIndependence&);
    void process_function_of(const ast::FunctionOf&);
    void process_indist(const ast::IndistinguishableScenarios&);

public:
    // consider this read-only
    std::vector<std::string> scenario_names;
    std::vector<std::vector<std::string>> var_names_by_scenario;
    std::vector<std::string> real_var_names;
    std::vector<int> opt_coeff_var_names;

    std::vector<ImplicitRules> implicits_by_scenario;

    // First row: constant terms (independent of opt_coeff_vars) for the proof target. Subsequent
    // rows get multiplied by the corresponding opt_coeff_vars.
    Matrix target_mat;
    MatrixT<Symbol> cmi_target_mat;

    Matrix constraints;
    MatrixT<Symbol> cmi_constraints;

    // Redundant constraints that can be used for proof simplification.
    MatrixT<Symbol> cmi_constraints_redundant;

    void process();

    // parser callback
    virtual void target(ast::TargetRelation) override;
    virtual void relation(ast::Relation) override;
    virtual void markov_chain(ast::MarkovChain) override;
    virtual void mutual_independence(ast::MutualIndependence) override;
    virtual void function_of(ast::FunctionOf) override;
    virtual void indist(ast::IndistinguishableScenarios) override;
};


ParserOutput parse(const std::vector<std::string>&);


#endif // include guard
