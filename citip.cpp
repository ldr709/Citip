#include <math.h>       // NAN
#include <utility>      // move
#include <random>
#include <sstream>      // istringstream
#include <stdexcept>    // runtime_error
#include <filesystem>
#include <iostream>
#include <thread>

#include <coin/CoinPackedMatrix.hpp>
#include <coin/OsiClpSolverInterface.hpp>
#include <coin/CbcSolver.hpp>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "citip.hpp"
#include "parser.hxx"
#include "scanner.hpp"
#include "common.hpp"

using std::move;
using util::sprint_all;


void CoinOsiProblem::setup(const OsiSolverInterface& solver)
{
    infinity = solver.getInfinity();
}

void CoinOsiProblem::load_problem_into_solver(OsiSolverInterface& solver)
{
    const double* p_collb = collb.empty() ? nullptr : collb.data();
    const double* p_colub = colub.empty() ? nullptr : colub.data();
    const double* p_rowlb = rowlb.empty() ? nullptr : rowlb.data();
    const double* p_rowub = rowub.empty() ? nullptr : rowub.data();
    const double* p_obj   = obj.empty() ? nullptr : obj.data();

    solver.loadProblem(constraints, p_collb, p_colub, p_obj, p_rowlb, p_rowub);
}

void CoinOsiProblem::append_with_default(std::vector<double>& vec, int size, double val, double def)
{
    if (val != def)
    {
        vec.resize(size, def);
        vec.back() = val;
    }
    else if (!vec.empty())
        vec.resize(size, def);
}

int CoinOsiProblem::add_row(double lb, double ub, int count, int* indices, double* values)
{
    int idx = num_rows++;

    constraints.appendRow(count, indices, values);
    append_with_default(rowlb, num_rows, lb, -infinity);
    append_with_default(rowub, num_rows, ub, infinity);

    return idx;
}

int CoinOsiProblem::add_col(double lb, double ub, double obj_, int count, int* indices, double* values)
{
    int idx = num_cols++;

    constraints.appendCol(count, indices, values);
    append_with_default(collb, num_cols, lb, -infinity);
    append_with_default(colub, num_cols, ub, infinity);
    append_with_default(obj, num_cols, obj_, 0.0);

    return idx;
}


static void check_num_vars(int num_vars)
{
    // The index type (int) must allow to represent column numbers up to
    // 2**num_vars. For signed int MAXINT = 2**(8*sizeof(int)-1)-1,
    // therefore the following is the best we can do (and 30 or so random
    // variables are probably already too much to handle anyway):
    int max_vars = 8*sizeof(int) - 2;
    if (num_vars > max_vars) {
        // Note that the base class destructor ~LinearProblem will still be
        // executed, thus freeing the allocated resource.
        throw std::runtime_error(sprint_all(
                    "Too many variables! At most ", max_vars,
                    " are allowed."));
    }
}


// Shift bits such that the given bit is free.
static int skip_bit(int pool, int bit_index)
{
    int bit = 1 << bit_index;
    int left = (pool & ~(bit-1)) << 1;
    int right = pool & (bit-1);
    return left | right;
}

static std::array<int, 2> var_to_set_and_scenario(
    int v, const std::vector<std::vector<std::string>>& var_names_by_scenario)
{
    for (int i = 0; i < var_names_by_scenario.size(); ++i)
    {
        int num_vars = (1 << var_names_by_scenario[i].size()) - 1;

        if (v <= num_vars)
            return {v, i};

        v -= num_vars;
    }

    throw std::runtime_error("Could not find variable's scenario");
}

static int apply_implicits(const ImplicitRules& implicits, int set)
{
    bool changed;
    do
    {
        changed = false;
        for (auto f : implicits.funcs)
        {
            if ((set & f.of) == f.of && (set | f.func) != set)
            {
                set |= f.func;
                changed = true;
            }
        }
    } while (changed);

    return set;
}

CmiTriplet::CmiTriplet(const ImplicitRules& implicits,
                       int a, int b, int c, int scenario_) :
    std::array<int, 3>{a, b, c},
    scenario(scenario_)
{
    auto& t = *this;

    // Apply implicit rules.

    auto apply_funcs_and_meta_rules = [&]()
    {
        t[2] = apply_implicits(implicits, t[2]);
        t[0] = apply_implicits(implicits, t[0] | t[2]) & ~t[2];
        t[1] = apply_implicits(implicits, t[1] | t[2]) & ~t[2];

        if (t[0] > t[1])
            std::swap(t[0], t[1]);

        if ((t[0] | t[1]) == t[1])
            t[1] = t[0];
    };

    apply_funcs_and_meta_rules();

    std::array<int, 3> old;
    do
    {
        old = t;

        for (auto indep : implicits.indeps)
        {
            // If independent from the other side and the premise, can move into the premise (MI
            // chain rule).
            for (int i = 0; i < 2; ++i)
            {
                auto to_move = indep.set & t[i];
                if (to_move && ((t[i ^ 1] | t[2]) | indep.indep_from) == indep.indep_from)
                {
                    t[i] &= ~to_move;
                    t[2] |= to_move;
                }
            }

            // If a premise is independent of everything else, it can be dropped.
            auto to_drop = indep.set & t[2];
            if (to_drop && ((t[0] | t[1] | t[2] & ~to_drop) | indep.indep_from) == indep.indep_from)
                t[2] &= ~to_drop;
        }

        if (old == t)
            break;

        apply_funcs_and_meta_rules();
    } while (old != t);
}

bool CmiTriplet::is_zero() const
{
    const auto& t = *this;
    return t[0] == 0 || t[1] == 0;
}

bool ExtendedShannonVar::is_zero() const
{
    return real_var < 0 && CmiTriplet::is_zero();
}

ExtendedShannonVar::operator std::variant<CmiTriplet, int>() const
{
    if (real_var < 0)
        return *(CmiTriplet*) this;
    else
        return real_var;
}

static inline int scenario_var(
    const ImplicitRules& implicits, const std::vector<std::vector<std::string>>& var_names_by_scenario,
    int scenario, int a)
{
    int prev_vars = 0;
    for (int i = 0; i < scenario; ++i)
        prev_vars += (1 << var_names_by_scenario[i].size()) - 1;
    return prev_vars + apply_implicits(implicits, a);
}


void ShannonTypeProblem::add_columns()
{
    for (int v = 0; v < real_var_names.size(); ++v)
    {
        int c = column_map.size();
        column_map[-v - 1] = c;
        inv_column_map.push_back(-v - 1);
    }

    for (int scenario = 0; scenario < scenario_names.size(); ++scenario)
    {
        for (int i = 1; i < (1<<var_names_by_scenario[scenario].size()); ++i)
        {
            int v = scenario_var(implicits_by_scenario[scenario], var_names_by_scenario, scenario, i);
            auto [it, inserted] = column_map.insert({v, column_map.size()});
            if (inserted)
                inv_column_map.push_back(v);
        }
    }

    add_columns(column_map.size());

    one_var = coin.add_col(0, 0); // Hack to make sure that the column index actually gets allocated.
    coin.colub[one_var] = coin.infinity;
}

void ShannonTypeProblem::add_elemental_inequalities()
{
    int num_scenarios = var_names_by_scenario.size();

    // Identify each variable with its index i from I = {0, 1, ..., N-1}.
    // Then entropy is a real valued set function from the power set of
    // indices P = 2**I. The value for the empty set can be defined to be
    // zero and is irrelevant. Therefore the dimensionality of the problem
    // is 2**N-1, summed over the different scenarios num_scenarios).

    int num_rows = 0;
    for (int scenario = 0; scenario < num_scenarios; ++scenario)
    {
        int num_vars = var_names_by_scenario[scenario].size();

        // After choosing 2 variables there are 2**(N-2) possible subsets of
        // the remaining N-2 variables.
        int sub_dim = 1 << (num_vars-2);
        num_rows += num_vars + (num_vars * (num_vars - 1) / 2) * sub_dim + 1;
    }
    row_to_cmi.reserve(num_rows);

    for (int scenario = 0; scenario < num_scenarios; ++scenario)
    {
        int num_vars = var_names_by_scenario[scenario].size();
        int sub_dim = 1 << (num_vars-2);
        const auto& implicits = implicits_by_scenario[scenario];

        // NOTE: We use 1-based indices for variable numbers, because 0 would correspond to H() = 0
        // and so be useless. However, Coin expects 0-based indices, so translation is needed.
        int indices[4];
        double values[4];
        int i, a, b;

        // index of the entropy component corresponding to the joint entropy of
        // all variables. NOTE: since the left-most column is not used, the
        // variables involved in a joint entropy correspond exactly to the bit
        // representation of its index.
        size_t all = (1<<num_vars) - 1;

        // Add all elemental conditional entropy positivities, i.e. those of
        // the form H(X_i|X_c)>=0 where c = ~ {i}:
        for (i = 0; i < num_vars; ++i) {
            int c = all ^ (1 << i);
            if (all == apply_implicits(implicits, c) || c == 0)
                continue;

            indices[0] = column_map.at(scenario_var(implicits, var_names_by_scenario, scenario, all));
            indices[1] = column_map.at(scenario_var(implicits, var_names_by_scenario, scenario, c));
            values[0] = +1;
            values[1] = -1;
            int row = coin.add_row_lb(0.0, 2, indices, values);
            row_to_cmi.emplace_back(CmiTriplet(implicits, 1 << i, 1 << i, c, scenario));
        }

        // Add all elemental conditional mutual information positivities, i.e.
        // those of the form I(X_a:X_b|X_K)>=0 where a,b not in K
        for (a = 0; a < num_vars-1; ++a) {
            for (b = a+1; b < num_vars; ++b) {
                int A = 1 << a;
                int B = 1 << b;
                for (i = 0; i < sub_dim; ++i) {
                    int K = skip_bit(skip_bit(i, a), b);
                    if (K != apply_implicits(implicits, K))
                        continue;

                    indices[0] = column_map.at(scenario_var(implicits, var_names_by_scenario, scenario, A|K));
                    indices[1] = column_map.at(scenario_var(implicits, var_names_by_scenario, scenario, B|K));
                    indices[2] = column_map.at(scenario_var(implicits, var_names_by_scenario, scenario, A|B|K));
                    values[0] = +1;
                    values[1] = +1;
                    values[2] = -1;
                    if (K)
                    {
                        indices[3] = column_map.at(scenario_var(implicits, var_names_by_scenario, scenario, K));
                        values[3] = -1;
                    }
                    int row = coin.add_row_lb(0.0, K ? 4 : 3, indices, values);
                    row_to_cmi.emplace_back(CmiTriplet(implicits, A,B,K, scenario));
                }
            }
        }
    }
}


//----------------------------------------
// ParserOutput
//----------------------------------------

void ParserOutput::add_term(SparseVector& v, SparseVectorT<Symbol>& cmi_v,
                            const ast::Term& t, int scenario_wildcard, double scale)
{
    add_quantity(v, cmi_v, t.quantity, scenario_wildcard, scale * t.coefficient);
}

void ParserOutput::add_quantity(SparseVector& v, SparseVectorT<Symbol>& cmi_v,
                                const ast::Quantity& quantity, int scenario_wildcard, double scale)
{
    if (std::holds_alternative<ast::ConstantQuantity>(quantity)) {
        v.inc(0, scale);
        return;
    }

    if (const auto* var = std::get_if<ast::VariableQuantity>(&quantity))
    {
        // A real valued variables, rather than an entropy.
        int var_i = get_real_var_index(var->name);
        v.inc(-var_i - 1, scale);
        cmi_v.inc(var_i, scale);
        return;
    }

    const auto& q = std::get<ast::EntropyQuantity>(quantity);
    int num_parts = q.parts.lists.size();

    const int scenario = (q.parts.scenario == "" && scenario_wildcard >= 0)
                         ? scenario_wildcard : scenarios.at(q.parts.scenario);

    const auto& implicits = implicits_by_scenario[scenario];

    // Need to index 2**num_parts subsets. For more detailed reasoning see
    // the check_num_vars() function.
    int max_parts = 8*sizeof(int) - 2;
    if (num_parts > max_parts) {
        throw std::runtime_error(sprint_all(
                    "Too many parts in multivariate mutual information! ",
                    "At most ", max_parts, " are allowed."));
    }

    // Multivariate mutual information is recursively defined by
    //
    //          I(a:…:y:z) = I(a:…:y) - I(a:…:y|z)
    //
    // These rules were rewritten from the alternating sum of (conditional)
    // entropies of all subsets of the parts [Jakulin & Bratko (2003)].
    //
    //      I(X₁:…:Xₙ|Y) = - Σ (-1)^|T| H(T|Y)
    //
    // where the sum is over all T ⊆ {X₁, …, Xₙ}.
    //
    // See: http://en.wikipedia.org/wiki/Multivariate_mutual_information

    std::vector<int> set_indices(num_parts);
    for (int i = 0; i < num_parts; ++i)
        set_indices[i] = get_set_index(scenario, q.parts.lists[i]);
    int c = get_set_index(scenario, q.cond);

    if (num_parts == 1)
    {
        add_cmi(v, cmi_v, CmiTriplet(implicits, set_indices[0], set_indices[0], c, scenario), scale);
    }
    else
    {
        // Use conditional mutual informations. That is, don't go all the way down to individual
        // entropies.
        // I(X₁:…:Xₙ|Y) = - Σ (-1)^|T| H(T|Y)
        // = - Σ (-1)^|T'| (H(T'|Y) - H(T',X_n|Y)) = Σ (-1)^|T'| H(X_n|T',Y)
        // = Σ (-1)^|T''| (H(X_n|T'',Y) - H(X_n|T'',X_{n-1},Y))
        // = Σ (-1)^|T''| I(X_{n-1},X_n | T'',Y)
        // where T' excludes X_n, and T'' excludes X_{n-1} as well.

        int num_subsets = 1 << (num_parts - 2);
        for (int set = 0; set < num_subsets; ++set)
        {
            int s = 1;
            int z = c;
            for (int i = 0; i < num_parts - 2; ++i) {
                if (set & 1<<i) {
                    c |= set_indices[i];
                    s = -s;
                }
            }
            add_cmi(v, cmi_v, CmiTriplet(implicits, set_indices[num_parts-2], set_indices[num_parts-1], z, scenario), s*scale);
        }
    }
}


int ParserOutput::get_var_index(int scenario, const std::string& s)
{
    std::tuple<int, std::string> key = {scenario, s};
    auto&& it = vars_by_scenario.find(key);
    if (it != vars_by_scenario.end())
        return it->second;
    int next_index = var_names_by_scenario[scenario].size();
    check_num_vars(next_index + 1);
    vars_by_scenario[key] = next_index;
    var_names_by_scenario[scenario].push_back(s);
    return next_index;
}

int ParserOutput::get_real_var_index(const std::string& name)
{
    auto&& it = real_vars.find(name);
    if (it != real_vars.end())
        return it->second;
    int next_index = real_var_names.size();
    real_vars[name] = next_index;
    real_var_names.push_back(name);
    return next_index;
}

int ParserOutput::get_set_index(int scenario, const ast::VarList& l)
{
    int idx = 0;
    for (auto&& v : l)
        idx |= 1 << get_var_index(scenario, v);
    return apply_implicits(implicits_by_scenario[scenario], idx);
}

void ParserOutput::target(ast::TargetRelation t)
{
    if (target_ast)
        throw std::runtime_error("Can only prove one single target bound at a time.");
    if (t.relation == ast::REL_EQ)
        throw std::runtime_error("Cannot prove equality bounds, as they are two bounds at once.");

    target_ast = t;
}

// Unfortunately the parser now needs two passes, so first save everything in a statement list.
void ParserOutput::relation(ast::Relation re)
{
    statement_list.emplace_back(std::in_place_index_t<RELATION>(), move(re));
}
void ParserOutput::mutual_independence(ast::MutualIndependence mi)
{
    statement_list.emplace_back(std::in_place_index_t<MUTUAL_INDEPENDENCE>(), move(mi));
}
void ParserOutput::markov_chain(ast::MarkovChain mc)
{
    statement_list.emplace_back(std::in_place_index_t<MARKOV_CHAIN>(), move(mc));
}
void ParserOutput::function_of(ast::FunctionOf fo)
{
    statement_list.emplace_back(std::in_place_index_t<FUNCTION_OF>(), move(fo));
}
void ParserOutput::indist(ast::IndistinguishableScenarios is)
{
    statement_list.emplace_back(std::in_place_index_t<INDISTINGUISHABLE_SCENARIOS>(), move(is));
}

void ParserOutput::process()
{
    if (!target_ast)
        throw std::runtime_error("Must provide a bound to prove.");

    // Add the target coefficients to optimize
    std::set<int> opt_coeff_vars_sorted;
    for (const ast::TargetExpression& side : {target_ast->left, target_ast->right})
        for (const auto& term : side)
            if (term.coefficient.optimize_coeff_var)
                opt_coeff_vars_sorted.insert(term.coefficient.optimize_coeff_var.value());

    for (int opt_coeff_var : opt_coeff_vars_sorted)
    {
        opt_coeff_vars[opt_coeff_var] = opt_coeff_var_names.size();
        opt_coeff_var_names.push_back(opt_coeff_var);
    }

    // Add the scenarios

    auto add_scenarios_from_quantity =
        [&](const ast::Quantity& quantity)
        {
            if (const auto* q = std::get_if<ast::EntropyQuantity>(&quantity))
                if (q->parts.scenario != "")
                    add_scenario(q->parts.scenario);
        };

    auto add_scenarios_from_expr =
        [&](const ast::Expression& e)
        {
            for (const auto& term : e)
                add_scenarios_from_quantity(term.quantity);
        };

    for (const ast::TargetExpression& side : {target_ast->left, target_ast->right})
        for (const auto& term : side)
            add_scenarios_from_quantity(term.quantity);

    for (const auto& s : statement_list)
        std::visit(overload {
            [&](const ast::Relation& r)
            {
                for (const ast::Expression& side : {r.left, r.right})
                    add_scenarios_from_expr(side);
            },
            [&](const ast::MarkovChain& mc)
            {
                add_scenarios(mc.scenarios);
                add_scenarios_from_expr(mc.bound);
            },
            [&](const ast::MutualIndependence& mi)
            {
                add_scenarios(mi.scenarios);
                if (mi.bound_or_implicit)
                    add_scenarios_from_expr(mi.bound_or_implicit.value());
            },
            [&](const ast::FunctionOf& f)
            {
                add_scenarios(f.scenarios);
                if (f.bound_or_implicit)
                    add_scenarios_from_expr(f.bound_or_implicit.value());
            },
            [&](const ast::IndistinguishableScenarios& is)
            {
                for (const auto& group: is.indist_scenarios)
                    add_scenarios(group.scenarios);
                for (const auto& expr: is.bound)
                    add_scenarios_from_expr(expr);
            }
        }, s);

    if (scenario_names.empty())
        add_scenario("");

    implicits_by_scenario.resize(scenario_names.size());
    var_names_by_scenario.resize(scenario_names.size());

    // Add the variables.

    auto add_vars_from_quantity =
        [&](const ast::Quantity& quantity)
        {
            std::visit(overload {
                [&](const ast::EntropyQuantity& q)
                {
                    for (auto [scenario, last] = scenario_range(q.parts.scenario);
                         scenario < last; ++scenario)
                    {
                        for (const auto& vl : q.parts.lists)
                            add_vars(scenario, vl);
                        add_vars(scenario, q.cond);
                    }
                },
                [&](const ast::VariableQuantity& var)
                {
                    add_real_var(var.name);
                },
                [&](const ast::ConstantQuantity& mi) {}
            }, quantity);
        };

    auto add_vars_from_expr =
        [&](const ast::Expression& e)
        {
            for (const auto& term : e)
                add_vars_from_quantity(term.quantity);
        };

    for (const ast::TargetExpression& side : {target_ast->left, target_ast->right})
        for (const auto& term : side)
            add_vars_from_quantity(term.quantity);

    for (const auto& s : statement_list)
        std::visit(overload {
            [&](const ast::Relation& r)
            {
                for (const ast::Expression& side : {r.left, r.right})
                    add_vars_from_expr(side);
            },
            [&](const ast::MarkovChain& mc)
            {
                for (const auto& sc: scenario_list(mc.scenarios))
                {
                    int scenario = scenarios.at(sc);
                    for (const auto& vl : mc.lists)
                        add_vars(scenario, vl);
                    add_vars_from_expr(mc.bound);
                }
            },
            [&](const ast::MutualIndependence& mi)
            {
                for (const auto& sc: scenario_list(mi.scenarios))
                {
                    int scenario = scenarios.at(sc);
                    for (const auto& vl : mi.lists)
                        add_vars(scenario, vl);
                    if (mi.bound_or_implicit)
                        add_vars_from_expr(mi.bound_or_implicit.value());
                }
            },
            [&](const ast::FunctionOf& f)
            {
                for (const auto& sc: scenario_list(f.scenarios))
                {
                    int scenario = scenarios.at(sc);
                    add_vars(scenario, f.function);
                    add_vars(scenario, f.of);
                    if (f.bound_or_implicit)
                        add_vars_from_expr(f.bound_or_implicit.value());
                }
            },
            [&](const ast::IndistinguishableScenarios& is)
            {
                for (const auto& group: is.indist_scenarios)
                    for (const auto& sc: scenario_list(group.scenarios))
                        add_vars(scenarios.at(sc), group.view);
                for (const auto& expr: is.bound)
                    add_vars_from_expr(expr);
            }
        }, s);

    // Retrieve the implicit function_of statements.
    for (const auto& s : statement_list)
    {
        if (s.index() != FUNCTION_OF)
            continue;
        const auto& fo = std::get<FUNCTION_OF>(s);
        if (!fo.implicit())
            continue;

        for (const auto& sc: scenario_list(fo.scenarios))
        {
            int scenario = scenarios.at(sc);
            int func = get_set_index(scenario, fo.function);
            int of = get_set_index(scenario, fo.of);
            implicits_by_scenario[scenario].funcs.push_back({func, of});
        }
    }

    // Apply the functions_of statements to each other, so that later they can be resolved faster.
    for (auto& impls : implicits_by_scenario)
    {
        bool changed;
        do
        {
            changed = false;
            for (auto& f : impls.funcs)
            {
                int new_func = f.func | (apply_implicits(impls, f.func | f.of) & ~f.of);
                if (new_func != f.func)
                {
                    f.func = new_func;
                    changed = true;
                }
            }
        } while (changed);
    }

    // Retrieve the implicit independence statements.
    for (const auto& s : statement_list)
    {
        if (s.index() != MUTUAL_INDEPENDENCE)
            continue;
        const auto& mi = std::get<MUTUAL_INDEPENDENCE>(s);
        if (!mi.implicit())
            continue;

        for (const auto& sc: scenario_list(mi.scenarios))
        {
            int scenario = scenarios.at(sc);
            std::vector<int> sets(mi.lists.size());
            for (int i = 0; i < sets.size(); ++i)
                // Implicit function rules are applied to the independence sets here.
                sets[i] = get_set_index(scenario, mi.lists[i]);

            std::vector<int> set_forward_unions(sets.size(), 0);
            for (int i = 1; i < sets.size(); ++i)
                set_forward_unions[i] = set_forward_unions[i - 1] | sets[i - 1];

            std::vector<int> set_backward_unions(sets.size(), 0);
            for (int i = sets.size() - 1; i > 0; --i)
                set_backward_unions[i - 1] = set_backward_unions[i] | sets[i];

            for (int i = 0; i < sets.size(); ++i)
                implicits_by_scenario[scenario].indeps.push_back(
                    ImplicitIndependence {sets[i], set_forward_unions[i] | set_backward_unions[i]});
        }
    }

    // Retrieve the target bound.
    {
        target_mat.resize(opt_coeff_var_names.size() + 1);
        cmi_target_mat.resize(opt_coeff_var_names.size() + 1);
        int sign = target_ast->relation == ast::REL_LE ? -1 : 1;
        for (const ast::TargetExpression& side : {target_ast->left, target_ast->right})
        {
            for (auto&& term : side)
            {
                int row = term.coefficient.optimize_coeff_var.has_value()
                        ? opt_coeff_vars.at(*term.coefficient.optimize_coeff_var) + 1 : 0;
                add_quantity(target_mat[row], cmi_target_mat[row], term.quantity, -1, sign * term.coefficient.scalar);
            }

            sign = -sign;
        }
    }

    for (const auto& s : statement_list)
        process_statement(s);
}

void ParserOutput::add_scenario(const std::string& scenario)
{
    auto [it, inserted] = scenarios.insert({scenario, scenario_names.size()});
    if (inserted)
        scenario_names.push_back(scenario);
}

void ParserOutput::add_scenarios(const ast::VarList& scenarios)
{
    for (const auto& sc: scenarios)
        add_scenario(sc);
}

void ParserOutput::add_vars(int scenario, const ast::VarList& vl)
{
    for (auto&& v : vl)
        get_var_index(scenario, v);
}

void ParserOutput::add_real_var(const std::string& name)
{
    get_real_var_index(name);
}

void ParserOutput::add_cmi(SparseVector& v, SparseVectorT<Symbol>& cmi_v,
                           CmiTriplet t, double coeff) const
{
    const auto& implicits = implicits_by_scenario[t.scenario];
    t = CmiTriplet(implicits, t[0], t[1], t[2], t.scenario);
    if (t.is_zero())
        return;

    auto [a, b, z] = t;
    v.inc(scenario_var(implicits, var_names_by_scenario, t.scenario, a|z), coeff);
    v.inc(scenario_var(implicits, var_names_by_scenario, t.scenario, b|z), coeff);
    v.inc(scenario_var(implicits, var_names_by_scenario, t.scenario, a|b|z), -coeff);
    if (z)
        v.inc(scenario_var(implicits, var_names_by_scenario, t.scenario, z), -coeff);

    cmi_v.inc(t, coeff);
}

std::tuple<int, int> ParserOutput::scenario_range(const std::string& scenario) const
{
    const int first_scenario = scenario == "" ? 0 : scenarios.at(scenario);
    const int last_scenario = scenario == "" ? scenario_names.size() : first_scenario + 1;

    return {first_scenario, last_scenario};
}

const std::vector<std::string>& ParserOutput::scenario_list(const ast::VarList& scenarios) const
{
    if (scenarios.empty() || (scenarios.size() == 1 && scenarios[0] == ""))
        return scenario_names;
    else
        return scenarios;
}

void ParserOutput::process_statement(const statement& s)
{
    switch (s.index())
    {
    case RELATION:
        process_relation(std::get<RELATION>(s));
        break;
    case MARKOV_CHAIN:
        process_markov_chain(std::get<MARKOV_CHAIN>(s));
        break;
    case MUTUAL_INDEPENDENCE:
        process_mutual_independence(std::get<MUTUAL_INDEPENDENCE>(s));
        break;
    case FUNCTION_OF:
        process_function_of(std::get<FUNCTION_OF>(s));
        break;
    case INDISTINGUISHABLE_SCENARIOS:
        process_indist(std::get<INDISTINGUISHABLE_SCENARIOS>(s));
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
        return;
    }
}

void ParserOutput::process_relation(const ast::Relation& re)
{
    bool has_wildcard_scenario = false;
    for (const ast::Expression& side : {re.left, re.right})
    {
        for (const auto& term : side)
        {
            if (const auto* q = std::get_if<ast::EntropyQuantity>(&term.quantity))
            {
                if (!q->parts.lists.empty() && q->parts.scenario == "")
                {
                    has_wildcard_scenario = true;
                    goto done;
                }
            }
        }
    }
done:

    for (int wildcard_scenario = 0;
         wildcard_scenario < (has_wildcard_scenario ? scenario_names.size() : 1);
         ++wildcard_scenario)
    {
        // create a SparseVector of standard '>=' form. For that the relation
        // needs to be transformed such that:
        //
        //      l <= r      =>      -l + r >= 0
        //      l >= r      =>       l - r >= 0
        //      l  = r      =>       l - r  = 0
        int l_sign = re.relation == ast::REL_LE ? -1 : 1;
        int r_sign = -l_sign;
        SparseVector v;
        SparseVectorT<Symbol> cmi_v;
        cmi_v.is_equality = v.is_equality = (re.relation == ast::REL_EQ);
        for (auto&& term : re.left)
            add_term(v, cmi_v, term, wildcard_scenario, l_sign);
        for (auto&& term : re.right)
            add_term(v, cmi_v, term, wildcard_scenario, r_sign);
        constraints.push_back(move(v));
        cmi_constraints.push_back(move(cmi_v));
    }
}

void ParserOutput::process_mutual_independence(const ast::MutualIndependence& mi)
{
    bool implicit = mi.implicit();
    bool approx = implicit ? false : !mi.bound_or_implicit.value().empty();

    for (const auto& sc: scenario_list(mi.scenarios))
    {
        int scenario = scenarios.at(sc);
        const auto& implicits = implicits_by_scenario[scenario];

        std::vector<int> set_indices(mi.lists.size());
        for (int i = 0; i < set_indices.size(); ++i)
            set_indices[i] = get_set_index(scenario, mi.lists[i]);

        auto convert_set = [&](int set)
        {
            int out_set = 0;
            for (int i = 0; i < set_indices.size(); ++i)
                if ((set >> i) & 1)
                    out_set |= set_indices[i];
            return out_set;
        };

        // Add many redundant rules for many formulations of independence. Hopefully some help to
        // simplify the proof.

        // 0 = H(a) + H(b) + H(c) + … - H(a,b,c,…) , and all subsets thereof.
        auto partition_constraint = [&](const std::vector<unsigned int>& subsets)
        {
            if (subsets.size() <= 2)
                return;

            //std::cout << "{";
            //for (auto subset : subsets)
            //    std::cout << subset << ", ";
            //std::cout << "}\n";

            int all = 0;
            SparseVector v;
            SparseVectorT<Symbol> cmi_v;
            cmi_v.is_equality = v.is_equality = !approx;
            for (auto subset : subsets) {
                // Check if subset contains the sentinal element saying that this subset is left out
                // of the partition.
                if ((subset >> set_indices.size()) & 1)
                    continue;

                int idx = convert_set(subset);
                all |= idx;
                add_cmi(v, cmi_v, CmiTriplet(implicits, idx,idx,0, scenario), -1);
                //std::cout << ExtendedShannonVar {CmiTriplet(implicits, idx,idx,0, scenario), &var_names, &scenario_names, &implicits} << " + ";
            }
            add_cmi(v, cmi_v, CmiTriplet(implicits, all,all,0, scenario), 1);
            //std::cout << "0 = " << ExtendedShannonVar {CmiTriplet(implicits, all,all,0, scenario), &var_names, &scenario_names, &implicits} << '\n';
            if (approx)
                for (auto&& term : mi.bound_or_implicit.value())
                    add_term(v, cmi_v, term, scenario, 1);
            constraints.push_back(move(v));
            cmi_constraints.push_back(move(cmi_v));
        };

        // Skip the redundant constraints entirely and just do the full subset if there's way too many.
        // Also, if the independence is approximate only include the one rule, otherwise it seems
        // less clear how the approximation bound is defined. (Though it still implies the others.)
        if (set_indices.size() > 13 || approx)
        {
            partition_constraint(*util::partitions(set_indices.size() + 1).begin());
            continue;
        }

        if (set_indices.size() <= 8)
        {
            // Also add rules for all partitions of subsets of a,b,c...
            for (const auto& partition : util::partitions(set_indices.size() + 1))
                partition_constraint(partition);
        }
        else
        {
            // Only add rules for all subsets of a,b,c...
            std::vector<unsigned int> partition;
            for (const auto& subset : util::disjoint_subsets(0, (1 << set_indices.size()) - 1))
            {
                for (int i = 0; i < set_indices.size(); ++i)
                    if (subset & (1 << i))
                        partition.push_back(1 << i);
                partition_constraint(partition);
            }
        }

        // CMI(a ; b | z) = 0 among the independent variables, for b and z both disjoint from a.

        // Already handled for implicit rules, so skip in that case.
        if (implicit)
            continue;

        // Avoid adding duplicate rules.
        std::set<CmiTriplet> added_rules;

        int full_set = (1 << set_indices.size()) - 1;

        // Skip z > 0 when this would make the set of constraints way too big.
        int max_z = (set_indices.size() > 5 ? 0 : full_set);

        for (int z = 0; z <= max_z; ++z)
        {
            for (int b : util::skip_n(util::disjoint_subsets(0, full_set), 1))
            {
                for (int a : util::skip_n(util::disjoint_subsets(z | b, full_set), 1))
                {
                    if (a > b && (b & z) == 0)
                        break;

                    CmiTriplet cmi(implicits, convert_set(a), convert_set(b), convert_set(z), scenario);

                    if (cmi.is_zero())
                        continue;
                    if (!added_rules.insert(cmi).second)
                        continue;

                    SparseVectorT<Symbol> cmi_v;
                    cmi_v.is_equality = false;
                    cmi_v.inc(cmi, -1.0);
                    cmi_constraints_redundant.push_back(move(cmi_v));
                }
            }
        }
    }
}

void ParserOutput::process_markov_chain(const ast::MarkovChain& mc)
{
    for (const auto& sc: scenario_list(mc.scenarios))
    {
        int scenario = scenarios.at(sc);
        const auto& implicits = implicits_by_scenario[scenario];

        int a = 0;
        for (int i = 0; i+2 < mc.lists.size(); ++i) {
            int b, c;
            a |= get_set_index(scenario, mc.lists[i+0]);
            b = get_set_index(scenario, mc.lists[i+1]);
            c = get_set_index(scenario, mc.lists[i+2]);
            // 0 >= I(a:c|b) = H(a|b) + H(c|b) - H(a,c|b)

            SparseVector v;
            SparseVectorT<Symbol> cmi_v;
            cmi_v.is_equality = v.is_equality = false;

            add_cmi(v, cmi_v, CmiTriplet(implicits, a,c,b, scenario), -1);
            for (auto&& term : mc.bound)
                add_term(v, cmi_v, term, scenario, 1);
            constraints.push_back(move(v));
            cmi_constraints.push_back(move(cmi_v));
        }
    }
}

void ParserOutput::process_function_of(const ast::FunctionOf& fo)
{
    // Already handled.
    if (fo.implicit())
        return;

    for (const auto& sc: scenario_list(fo.scenarios))
    {
        int scenario = scenarios.at(sc);
        const auto& implicits = implicits_by_scenario[scenario];

        int func = get_set_index(scenario, fo.function);
        int of = get_set_index(scenario, fo.of);

        // 0 = H(func|of) = H(func,of) - H(of)
        SparseVector v;
        SparseVectorT<Symbol> cmi_v;
        cmi_v.is_equality = v.is_equality = false;
        add_cmi(v, cmi_v, CmiTriplet(implicits, func,func,of,scenario), -1);
        for (auto&& term : fo.bound_or_implicit.value())
            add_term(v, cmi_v, term, scenario, 1);
        constraints.push_back(move(v));
        cmi_constraints.push_back(move(cmi_v));
    }
}

void ParserOutput::process_indist(const ast::IndistinguishableScenarios& is)
{
    if (is.indist_scenarios.empty())
        return;

    size_t view_size = is.indist_scenarios.front().view.size();
    if (!view_size)
        return;

    auto bound = is.bound;
    bool approx = !bound.empty();
    if (bound.size() == 1)
        bound.resize(view_size + 1);
    if (approx && bound.size() != view_size + 1)
        throw std::runtime_error("indist expects view_size + 1 (or 1) approximation bounds.");

    auto add_approx_bound = [&](SparseVector& v, SparseVectorT<Symbol>& cmi_v,
                                int a, bool is_mutual, double scale)
    {
        for (auto&& term : bound[0])
            add_term(v, cmi_v, term, -1, scale * (1 + is_mutual));
        for (int i = 0; i < view_size; ++i)
            if ((a >> i) & 1)
                for (auto&& term : bound[i + 1])
                    add_term(v, cmi_v, term, -1, scale);
    };

    // Require that all entropies defined by the view match between the scenarios.
    auto indist_views = [&](int scenario0, const ast::VarList& view0,
                            int scenario1, const ast::VarList& view1,
                            bool all_redundant)
    {
        const auto& implicits0 = implicits_by_scenario[scenario0];
        const auto& implicits1 = implicits_by_scenario[scenario1];

        if (view0.size() != view_size || view1.size() != view_size)
            throw std::runtime_error("Supposed indistinguishable views have differing sizes.");

        int full_set = (1 << view0.size()) - 1;

        auto convert_set = [&](int scenario, const ast::VarList& view, int set)
        {
            int out_set = 0;
            for (int i = 0; i < view.size(); ++i)
                if ((set >> i) & 1)
                    out_set |= (1 << get_var_index(scenario, view[i]));
            return out_set;
        };

        // Require that all entropies defined by the view match between the scenarios. This is
        // skipped when all_redundant (i.e., when i > 1 && !approx), as those are implied by the two
        // equalities going through i = 1.
        if (!all_redundant)
            for (int a : util::skip_n(util::disjoint_subsets(0, full_set), 1))
            {
                int a0 = convert_set(scenario0, view0, a);
                int a1 = convert_set(scenario1, view1, a);

                CmiTriplet cmi0(implicits0, a0, a0, 0, scenario0);
                CmiTriplet cmi1(implicits1, a1, a1, 0, scenario1);
                if (cmi0.is_zero() && cmi1.is_zero())
                    continue;

                for (int neg = 0; neg <= approx; ++neg)
                {
                    SparseVector v;
                    SparseVectorT<Symbol> cmi_v;
                    cmi_v.is_equality = v.is_equality = !approx;
                    add_cmi(v, cmi_v, cmi0, 1 - 2*neg);
                    add_cmi(v, cmi_v, cmi1, 2*neg - 1);
                    if (approx)
                        add_approx_bound(v, cmi_v, a, false, 1.0);
                    constraints.push_back(move(v));
                    cmi_constraints.push_back(move(cmi_v));
                }
            }

        // Avoid adding duplicate rules.
        std::set<std::array<int, 6>> added_rules;

        // Skip z > 0 when this would make the set of constraints way too big.
        int max_z = (view0.size() > 6 && !approx ? 0 : full_set);

        // Skip the redundant constraints entirely if there's way too many.
        if (view0.size() > 14 && !approx)
            return;

        // Redundantly include all pairs of CMIs, rather than just the base entropies, so that the
        // simplifier can pick the most useful equalities. Again, these are not so redundant when
        // there's only an approximate bound on the entropy difference.
        for (int z = 0; z <= max_z; ++z)
        {
            for (int b : util::skip_n(util::disjoint_subsets(z, full_set), 1))
            {
                for (int a : util::skip_n(util::disjoint_subsets(z, full_set), 1))
                {
                    if (a > b)
                        break;
                    if (a == b && z == 0) // Already handled above.
                        continue;
                    if (a != b && ((a | b) == b || (a | b) == a))
                        continue;

                    CmiTriplet cmi0(implicits0,
                                    convert_set(scenario0, view0, a),
                                    convert_set(scenario0, view0, b),
                                    convert_set(scenario0, view0, z), scenario0);
                    CmiTriplet cmi1(implicits1,
                                    convert_set(scenario1, view1, a),
                                    convert_set(scenario1, view1, b),
                                    convert_set(scenario1, view1, z), scenario1);

                    if (cmi0.is_zero() && cmi1.is_zero())
                        continue;
                    if (!added_rules.insert({cmi0[0], cmi0[1], cmi0[2], cmi1[0], cmi1[1], cmi1[2]}).second)
                        continue;

                    if (!approx)
                    {
                        SparseVectorT<Symbol> cmi_v;
                        cmi_v.is_equality = true;
                        cmi_v.inc(cmi0, 1.0);
                        cmi_v.inc(cmi1, -1.0);
                        cmi_constraints_redundant.push_back(move(cmi_v));
                    }
                    else
                    {
                        for (int use_b_size = 0; use_b_size <= (a != b); ++use_b_size)
                        {
                            int set_for_size = use_b_size ? b : a;
                            for (int neg = 0; neg <= approx; ++neg)
                            {
                                SparseVector v;
                                SparseVectorT<Symbol> cmi_v;
                                cmi_v.is_equality = v.is_equality = false;
                                add_cmi(v, cmi_v, cmi0, 1 - 2*neg);
                                add_cmi(v, cmi_v, cmi1, 2*neg - 1);
                                add_approx_bound(v, cmi_v, set_for_size, a != b, 1.0);
                                constraints.push_back(move(v));
                                cmi_constraints.push_back(move(cmi_v));

                                //std::cout << ExtendedShannonVar {cmi0, &var_names_by_scenario, &scenario_names, nullptr, &implicits_by_scenario[scenario0]} << ", ";
                                //std::cout << ExtendedShannonVar {cmi1, &var_names_by_scenario, &scenario_names, nullptr, &implicits_by_scenario[scenario1]} << ", " << set_for_size << '\n';
                            }
                        }
                    }
                }
            }
        }
    };

    // Includes redundant pairs of scenarios so that the simplifier can pick the most useful pairs.
    // These are not so redundant when there's an approximate bound, as different ways through the
    // constraints add in different extra approximation values.
    int i = 0;
    for (const auto& group0 : is.indist_scenarios)
    {
        const auto& scenario_list0 = group0.scenarios.empty() ? scenario_names : group0.scenarios;
        for (const auto& scenario_name0 : scenario_list0)
        {
            ++i;
            int scenario0 = scenarios.at(scenario_name0);

            int j = 0;
            for (const auto& group1 : is.indist_scenarios)
            {
                const auto& scenario_list1 = group1.scenarios.empty() ? scenario_names : group1.scenarios;
                for (const auto& scenario_name1 : scenario_list1)
                {
                    ++j;
                    if (i >= j)
                        continue;
                    int scenario1 = scenarios.at(scenario_name1);

                    indist_views(scenario0, group0.view, scenario1, group1.view, i > 1 && !approx);
                }
            }
        }
    }
}


//----------------------------------------
// LinearProblem
//----------------------------------------

LinearProblem::LinearProblem() :
    si(new OsiClpSolverInterface()),
    coin(*si, false)
{}

LinearProblem::LinearProblem(int num_cols)
    : LinearProblem()
{
    add_columns(num_cols);
}

LinearProblem::~LinearProblem() {}

void LinearProblem::add_columns(int num_cols)
{
    for (int i = 1; i <= num_cols; ++i)
        coin.add_col_lb(0.0);
}

void LinearProblem::add(const SparseVector& v)
{
    std::vector<int> indices;
    std::vector<double> values;
    indices.reserve(v.entries.size());
    values.reserve(v.entries.size());
    for (auto&& ent : v.entries) {
        if (ent.first == 0)
            continue;
        indices.push_back(ent.first - 1);
        values.push_back(ent.second);
    }

    double lb = -v.get(0);
    double ub = v.is_equality ? lb : coin.infinity;
    coin.add_row(lb, ub, indices.size(), indices.data(), values.data());
}

void print_coeff(std::ostream& out, double c, bool first)
{
    if (!first)
    {
        if (c >= 0)
            out << " + ";
        else
        {
            out << " - ";
            c = -c;
        }
    }

    if (first && std::abs(c + 1.0) <= eps)
        out << '-';
    else if (std::abs(c - 1.0) > eps)
        out << c << ' ';
}

std::ostream& operator<<(std::ostream& out, const LinearVariable& v)
{
    out << 'v' << v.id;
    return out;
}

LinearProof<> LinearProblem::prove_impl(const SparseVector& I, int num_regular_rules,
                                        bool want_proof, bool check_bound)
{
    if (I.is_equality)
        throw std::runtime_error("Checking for equalities is not supported.");

    coin.obj.resize(coin.num_cols);
    for (int i = 1; i <= coin.num_cols; ++i)
        coin.obj[i - 1] = I.get(i);

    coin.load_problem_into_solver(*si);
    //si->writeLp("debug");

    si->setLogLevel(3);
    si->getModelPtr()->setPerturbation(50);

    if (false)
    {
        ClpSolve options;
        // Use the Idiot presolve with 2000 iterations, or something like that. Not sure which of
        // these 1200s is needed.
        options.setPresolveType(ClpSolve::presolveOn, 2000);
        options.setSpecialOption(1,2,2000);
        si->setSolveOptions(options);
    }

    si->initialSolve();

    if (!si->isProvenOptimal()) {
        throw std::runtime_error("LinearProblem: Failed to solve LP.");
    }

    // the original check was for the solution (primal variable values)
    // rather than objective value, but let's do it simpler for now (if
    // an optimum is found, it should be zero anyway):
    if (!check_bound || si->getObjValue() + I.get(0) + eps >= 0.0)
    {
        LinearProof proof;
        proof.initialized = true;

        double const_term = si->getObjValue() + I.get(0);
        proof.dual_solution.entries[0] = const_term;

        if (!want_proof)
            return proof;

        const double* si_solution = si->getColSolution();
        proof.primal_solution = std::vector<double>(si_solution, si_solution + coin.num_cols);

        proof.objective = I;
        proof.objective.is_equality = false;

        const double* row_price = si->getRowPrice();
        std::vector<double> col_price(coin.num_cols, 0.0);
        coin.constraints.transposeTimes(row_price, col_price.data());
        for (int j = 0; j < coin.num_cols; ++j)
            col_price[j] = coin.obj[j] - col_price[j];

        for (int j = 0; j < coin.num_cols; ++j)
        {
            proof.variables.emplace_back(LinearVariable{j});
            proof.regular_constraints.emplace_back(NonNegativityRule{j});

            double coeff = col_price[j];
            if (std::abs(coeff) > eps)
                proof.dual_solution.entries[j + 1] = coeff;
        }

        for (int i = 0; i < coin.num_rows; ++i)
        {
            double coeff = row_price[i];
            if (std::abs(coeff) > eps)
                proof.dual_solution.entries[i + coin.num_cols + 1] = coeff;
        }

        for (int i = num_regular_rules; i < coin.num_rows; ++i)
        {
            auto coin_constraint = coin.constraints.getVector(i);
            bool no_lb = coin.rowlb.empty() || coin.rowlb[i] == -coin.infinity;
            bool no_ub = coin.rowub.empty() || coin.rowub[i] == coin.infinity;

            if (no_lb && no_ub) // Free row -- not actually a constraint.
                continue;

            int row_type = 0;
            if (no_lb)
                row_type = -1;
            if (no_ub)
                row_type = 1;

            proof.custom_constraints.emplace_back();
            auto& constraint = proof.custom_constraints.back();
            for (int j = 0; j < coin_constraint.getNumElements(); ++j)
            {
                double coeff = coin_constraint.getElements()[j];
                constraint.inc(coin_constraint.getIndices()[j] + 1, row_type == -1 ? -coeff : coeff);
            }

            double const_offset = 0.0;
            if (row_type == -1)
                const_offset = coin.rowub[i];
            else
                const_offset = -coin.rowlb[i];
            if (std::abs(const_offset) > eps)
                constraint.entries[0] = const_offset;

            constraint.is_equality = (row_type == 0);
        }

        return proof;
    }
    else
    {
        // TODO: Counterexample.
        return LinearProof();
    }
}


ShannonTypeProblem::ShannonTypeProblem(std::vector<std::vector<std::string>> var_names_by_scenario_,
                                       std::vector<std::string> scenario_names_,
                                       std::vector<std::string> real_var_names_,
                                       std::vector<int> opt_coeff_var_names_,
                                       std::vector<ImplicitRules> implicits_by_scenario_,
                                       MatrixT<Symbol> cmi_constraints_redundant_) :
    LinearProblem(),
    var_names_by_scenario(move(var_names_by_scenario_)),
    scenario_names(move(scenario_names_)),
    real_var_names(move(real_var_names_)),
    opt_coeff_var_names(move(opt_coeff_var_names_)),
    implicits_by_scenario(move(implicits_by_scenario_)),
    cmi_constraints_redundant(cmi_constraints_redundant_)
{
    for (int i = 0; i < scenario_names_.size(); ++i)
        check_num_vars(var_names_by_scenario[i].size());

    add_columns();
    add_elemental_inequalities();
}

ShannonTypeProblem::ShannonTypeProblem(const ParserOutput& out) :
    ShannonTypeProblem(out.var_names_by_scenario, out.scenario_names, out.real_var_names,
                       out.opt_coeff_var_names, out.implicits_by_scenario,
                       out.cmi_constraints_redundant)
{
    for (int i = 0; i < out.constraints.size(); ++i)
        add(out.constraints[i], out.cmi_constraints[i]);
}

void ShannonTypeProblem::add(const SparseVector& v)
{
    SparseVector realV;
    realV.is_equality = v.is_equality;
    for (const auto& [x, y] : v.entries)
        realV.inc(x ? column_map.at(x) + 1 : one_var + 1, y);
    LinearProblem::add(realV);
}

void ShannonTypeProblem::add(const SparseVector& v, SparseVectorT<Symbol> cmi_v)
{
    add(v);
    cmi_constraints.push_back(move(cmi_v));
}

static void print_var_subset(std::ostream& out, int v,
                             const std::vector<std::string>& random_var_names)
{
    for (int i = 0; v; ++i)
    {
        if (v & 1)
        {
            out << random_var_names[i];
            if (v >>= 1)
                out << ',';
        }
        else
            v >>= 1;
    }
}

const std::string& ShannonVar::scenario() const
{
    return scenario_names[var_to_set_and_scenario(v, var_names_by_scenario)[1]];
}

std::ostream& operator<<(std::ostream& out, const ShannonVar::PrintVarsOut& pvo)
{
    auto [set, scenario] = var_to_set_and_scenario(pvo.parent.v, pvo.parent.var_names_by_scenario);
    print_var_subset(out, set, pvo.parent.var_names_by_scenario[scenario]);
    return out;
}

std::ostream& operator<<(std::ostream& out, const ShannonVar& sv)
{
    if (sv.v >= 0) // Entropy variable
        return out << 'H' << sv.scenario() << '(' << sv.print_vars() << ')';
    else
        return out << sv.real_var_names[-sv.v - 1];
}

bool ShannonRule::print(std::ostream& out, const ShannonVar* vars, double scale) const
{
    const std::vector<std::vector<std::string>>* var_names_by_scenario = &vars[0].var_names_by_scenario;
    const std::vector<std::string>* scenario_names = &vars[0].scenario_names;
    const ImplicitRules* implicits = &vars[0].implicits_by_scenario[scenario];

    if (scale == 0.0 || is_zero())
        return false;

    print_coeff(out, scale, true);
    if ((*this)[0] >= 0)
    	out << ExtendedShannonVar {*this, var_names_by_scenario, scenario_names, nullptr, implicits};
    else
    {
    	assert((*this)[1] == (*this)[0]);
    	assert((*this)[2] == 0);
    	out << ExtendedShannonVar {*this, var_names_by_scenario, scenario_names, &vars[0].real_var_names, implicits, -(*this)[0] - 1};
    }
    out << " >= 0";
    return true;
}

ShannonTypeProof ShannonTypeProblem::prove(Matrix I, MatrixT<Symbol> cmi_I)
{
    // Optimize the coefficients of the bound.

    int num_c_vars = opt_coeff_var_names.size();
    assert(I.size() == num_c_vars + 1);
    assert(cmi_I.size() == num_c_vars + 1);

    std::vector<double> opt_coeff_vals(num_c_vars);

    SparseVector real_obj;
    for (const auto& [x, y] : I[0].entries)
        real_obj.inc(x ? column_map.at(x) + 1 : one_var + 1, y);

    size_t c_var_rows_start = coin.num_rows;
    for (int c_var = 0; c_var < num_c_vars; ++c_var)
    {
        I[c_var + 1].is_equality = true;
        add(I[c_var + 1]);
    }

    for (int c_var = 0; c_var < num_c_vars; ++c_var)
    {
        // See how much the objective changes if expression c_var is multiplied by decreases by 1.
        coin.rowlb[c_var_rows_start + c_var] -= 1.0;
        coin.rowub[c_var_rows_start + c_var] -= 1.0;

        auto result = optimize(real_obj);
        if (!result.has_value())
            return ShannonTypeProof();

        opt_coeff_vals[c_var] = result.value();
        std::cout << "Optimized: c" << opt_coeff_var_names[c_var] << " = " << opt_coeff_vals[c_var] << '\n';

        // Set the coefficient and merge it into the main objective.
        for (const auto& [x, y] : I[c_var + 1].entries)
            real_obj.inc(x ? column_map.at(x) + 1 : one_var + 1, opt_coeff_vals[c_var] * y);
        for (const auto& [x, y] : cmi_I[c_var + 1].entries)
            cmi_I[0].inc(x, opt_coeff_vals[c_var] * y);
        coin.rowlb[c_var_rows_start + c_var] = -coin.infinity;
        coin.rowub[c_var_rows_start + c_var] =  coin.infinity;
    }

	// Add a pseudorandom cost to using each rule, to try to avoid any degeneracy. Degeneracy can
	// result in some arbitrary mix of different proofs, which will be more complicated than any one
	// of the proofs.
    std::lognormal_distribution rand_cost_dist(0.0, 1.0);
    std::minstd_rand rand_source(5847171071UL);

	for (int row = 0; row < c_var_rows_start; ++row)
	{
		double row_cost = rand_cost_dist(rand_source);
		if (row < coin.rowlb.size() && coin.rowlb[row] > -coin.infinity)
			coin.rowlb[row] -= row_cost;
		if (row < coin.rowub.size() && coin.rowub[row] < coin.infinity)
			coin.rowub[row] += row_cost;
	}

	for (int col = 0; col < coin.num_cols; ++col)
	{
		double col_cost = rand_cost_dist(rand_source);
		if (col < coin.collb.size() && coin.collb[col] > -coin.infinity)
			coin.collb[col] -= col_cost;
		if (col < coin.colub.size() && coin.colub[col] < coin.infinity)
			coin.colub[col] += col_cost;
	}

    LinearProof lproof = LinearProblem::prove(real_obj, row_to_cmi, false);

    // Remove the ficticious cost from the proof.
    std::cout << "Pseudorandom proof cost: " << -lproof.dual_solution.get(0) << '\n';
	lproof.dual_solution.entries.erase(0);

    // Remove one_var from lproof and replace it with the actual value 1, as it will just confuse
    // everything downstream.
    for (auto it = lproof.dual_solution.entries.cbegin(); it != lproof.dual_solution.entries.cend();)
    {
        if (it->first >= one_var + 1)
        {
        	if (it->first == one_var + 1)
            	lproof.dual_solution.inc(0, it->second);
        	else
            	lproof.dual_solution.entries[it->first - 1] = it->second;
            auto old_it = it++;
            lproof.dual_solution.entries.erase(old_it);
        }
        else
            ++it;
    }

    for (auto& constraint : lproof.custom_constraints)
    {
        constraint.entries[0] = constraint.get(one_var + 1);
        constraint.entries.erase(one_var + 1);
    }
    lproof.objective.entries[0] = lproof.objective.get(one_var + 1);
    lproof.objective.entries.erase(one_var + 1);

    lproof.primal_solution.pop_back();
    lproof.variables.pop_back();
    lproof.regular_constraints.erase(lproof.regular_constraints.begin() + one_var);

    ShannonTypeProof proof(
        lproof,
        [&] (const LinearVariable& v)
        {
            return ShannonVar{
                var_names_by_scenario, scenario_names, real_var_names,
                column_map, implicits_by_scenario, inv_column_map[v.id]
            };
        },
        [&] (const NonNegativityOrOtherRule<CmiTriplet>& r) -> ShannonRule {
            if (r.index() == 0)
            {
                int var = inv_column_map[lproof.variables[std::get<0>(r).v].id];
                auto [set, scenario] = var_to_set_and_scenario(var, var_names_by_scenario);
                return ShannonRule({set, set, 0, scenario});
            }
            else
                return ShannonRule((std::get<1>(r)));
        });
    proof.cmi_constraints = cmi_constraints;
    proof.cmi_constraints_redundant = cmi_constraints_redundant;
    proof.cmi_objective = move(cmi_I[0]);

    return proof;
}


// Simplify the Shannon bounds in a proof by combining them into conditional mutual informations.
struct ShannonProofSimplifier
{
    friend struct ExtendedShannonRule;
    typedef ExtendedShannonRule Rule;
    typedef ShannonTypeProof::Symbol Symbol;

    ShannonProofSimplifier() = delete;
    ShannonProofSimplifier(const ShannonTypeProof&);

    bool simplify(int depth);

    operator bool() const { return orig_proof; }
    operator SimplifiedShannonProof();

private:
    void add_all_rules();
    void add_adjacent_rules(CmiTriplet);
    void add_adjacent_rules_quadratic(CmiTriplet);

    static double custom_rule_complexity_cost(const SparseVectorT<CmiTriplet>&);
    static double custom_rule_complexity_cost(const SparseVectorT<Symbol>&);

    // How much to use each type of (in)equality:
    std::map<CmiTriplet, double> cmi_coefficients;
    std::map<Rule, double> rule_coefficients;
    std::map<int, double> custom_rule_coefficients;

    double cost;

    const ShannonTypeProof& orig_proof;
    MatrixT<Symbol> cmi_constraints;

    std::map<Symbol, int> cmi_indices; // rows represent conditional mutual informations.
    int get_row_index(Symbol t, bool allow_equality = true);

    std::map<Rule, int> rule_indices;
    int add_rule(const Rule& r);
    int add_constraint(const SparseVectorT<Symbol>& c, double cost = 0.0);

    CoinOsiProblem coin;

    const std::vector<ImplicitRules>& implicits_by_scenario;
    const std::vector<std::vector<std::string>>& var_names_by_scenario;
    const std::vector<std::string>& real_var_names;
    const std::vector<std::string>& scenario_names;
};

SimplifiedShannonProof ShannonTypeProof::simplify(int depth) const
{
    ShannonProofSimplifier simplifier(*this);

    if (depth == 0)
        return simplifier;
    else if (depth == -1)
    {
        simplifier.simplify(depth);
        return simplifier;
    }

    // Iteratively deepen the simplification.
    bool changed;
    do
    {
        changed = false;
        for (int d = 1; d <= depth; ++d)
        {
            changed = simplifier.simplify(d);
            if (changed)
                break;
        }
    } while (changed);

    return simplifier;
}

// Divide the rules being used into inequalities and equalities. Each rule gets its own cost of use.
// First, the inequalities consist mainly of the non-negativity of conditional mutual information.
// Equality rules connect the different kinds of information together:
//
// Inequality:
// CMI nonneg.                    I(a;b|z) >= 0
// Learning reduces entropy       I(a|z) >= I(a|b,z) (CMI nonneg I(a;b|z) and CMI defn. 3)
// More variables more entropy    I(a,b|z) >= I(a|z) (CMI nonneg I(b|a,z) and chain rule)
// (Combined into monotone:)      I(a,b|z) >= I(a|c,z)
// Monotone mutual information    I(a;b,c|z) >= I(a;b|z) (CMI nonneg I(a;c|b,z) and MI chain rule)
//
// Equality:
// CMI defn. 1                    I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
// CMI defn. 2                    I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z) (Same as defn. 1 with c=0)
// CMI defn. 3                    I(a;b|z) = I(a|z) - I(a|b,z) (Same as MI chain with c->b and b->a)
// chain rule:                    I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
// mutual information chain rule: I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
//
// CMI defn. 2 is the same as the two-set inclusion-exclusion principle.
//
// Implicitly used meta-rules:
// I(a;b|z) = I(b;a|z) (symmetry)
// I(a,a,b;c|z) = I(a,b;c|z) and I(a;b|c,c,z)=I(a;b|c,z) (redundancy)
// I(a;|z) = 0 (trivial rule)
// I(a;b|c,z) = I(a;b,c|c,z) (redundancy and trivial rule, combined with MI chain rule)
// I(a,b;b|z) = I(b;b|z) (mutual information is a subset of all information (special case of CMI defn. 2)

ExtendedShannonRule::ExtendedShannonRule(const ImplicitRules& implicits, type_enum type_,
                                         int z, int a, int b, int c, int scenario_) :
    type(type_),
    scenario(scenario_)
{
    // Attempt to reduce rule duplication by applying implicit rules, where possible.
    switch (type)
    {
    case CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
    case CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        {
            CmiTriplet cmi(implicits, a, b, c | z, scenario);
            z = apply_implicits(implicits, z);
            c = cmi[2] & ~z;
            a = cmi[0];
            b = cmi[1];
        }
        break;

    case MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        z = apply_implicits(implicits, z);
        a = apply_implicits(implicits, a | z) & ~z;
        c = apply_implicits(implicits, c | z) & ~z;
        b = apply_implicits(implicits, b | c | z) & ~(c | z);

        if ((a | b | c) == (b | c))
            b = a;
        break;

    case MONOTONE_COND:
        // I(a,b|z) >= I(a|c,z)
        {
            z      = apply_implicits(implicits, z);
            int ab = apply_implicits(implicits, a | b | z) & ~z;
            c      = apply_implicits(implicits, c | z) & ~z;
            a      = apply_implicits(implicits, a | c | z) & ~(c | z);
            b      = ab & ~a;
        }
        break;

    case MONOTONE_MUT:
        // I(a;b,c|z) >= I(a;b|z)
        z = apply_implicits(implicits, z);
        a = apply_implicits(implicits, a | z) & ~z;
        b = apply_implicits(implicits, b | z) & ~z;
        c = apply_implicits(implicits, c | b | z) & ~(b | z);

        if ((a | b) == b)
            // Trivial case.
            b = a;
        if ((a | b | c) == (b | c))
            c = a & ~b;

        if ((a | b | c) == a)
            a = b | c;
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
    }

    subsets = {z, a, b, c};
}

const double rule_cost                           = 1.0;
const double information_cost                    = 1.0;
const double conditional_information_cost        = 1.1;
const double mutual_information_cost             = 1.5;
const double conditional_mutual_information_cost = 1.6;
const double real_var_cost                       = 0.5;

// Cost of using the bound I(a;b|c) >= 0 where t == (a, b, c).
double CmiTriplet::complexity_cost() const
{
    if (is_zero())
        // Give a reason to avoid using rules with trivial terms, if other alternatives exist.
        return 0.01;

    const CmiTriplet& t = *this;
    if (t[0] == t[1])
        if (t[2] == 0)
            return information_cost;
        else
            return conditional_information_cost;
    else
        if (t[2] == 0)
            return mutual_information_cost;
        else
            return conditional_mutual_information_cost;
}

double ExtendedShannonRule::complexity_cost(const ImplicitRules& implicits) const
{
    return ShannonProofSimplifier::custom_rule_complexity_cost(get_constraint(implicits));
}

static double complexity_cost(std::variant<CmiTriplet, int> s)
{
    return std::visit(overload {
        [&](const CmiTriplet& cmi)
        {
            return cmi.complexity_cost();
        },
        [&](const int& v)
        {
            return real_var_cost;
        }
    }, s);
}

double ShannonProofSimplifier::custom_rule_complexity_cost(const SparseVectorT<CmiTriplet>& c)
{
    SparseVectorT<Symbol> constraint;
    constraint.is_equality = c.is_equality;
    for (const auto& [cmi, v] : c.entries)
        constraint.entries[cmi] = v;
    return custom_rule_complexity_cost(constraint);
}

double ShannonProofSimplifier::custom_rule_complexity_cost(const SparseVectorT<Symbol>& c)
{
    double cost = 0.0;
    for (const auto& [cmi, v] : c.entries)
        cost += complexity_cost(cmi);

    // Cost for adding more rules, even if they are simple.
    return cost + rule_cost;
}

bool ExtendedShannonRule::is_trivial(const SparseVectorT<CmiTriplet>& c) const
{
    for (const auto& [cmi, v] : c.entries)
        if (v != 0.0 && !cmi.is_zero())
            return false;
    return true;
}

bool ExtendedShannonRule::is_trivial(const ImplicitRules& implicits) const
{
    return is_trivial(get_constraint(implicits));
}

SparseVectorT<CmiTriplet> ExtendedShannonRule::get_constraint(
    const ImplicitRules& implicits) const
{
    auto [z, a, b, c] = subsets;

    SparseVectorT<CmiTriplet> constraint;
    constraint.is_equality = is_equality();
    switch (type)
    {
    case CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
        constraint.inc(CmiTriplet(implicits, a, b, c|z, scenario),        1.0);
        constraint.inc(CmiTriplet(implicits, a|c, a|c, z, scenario),     -1.0);
        constraint.inc(CmiTriplet(implicits, b|c, b|c, z, scenario),     -1.0);
        constraint.inc(CmiTriplet(implicits, a|b|c, a|b|c, z, scenario),  1.0);
        constraint.inc(CmiTriplet(implicits, c, c, z, scenario),          1.0);
        break;

    case CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        constraint.inc(CmiTriplet(implicits, c, c, z, scenario),          1.0);
        constraint.inc(CmiTriplet(implicits, a, b, c|z, scenario),        1.0);
        constraint.inc(CmiTriplet(implicits, a|c, b|c, z, scenario),     -1.0);
        break;

    case MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        constraint.inc(CmiTriplet(implicits, a, c, z, scenario),          1.0);
        constraint.inc(CmiTriplet(implicits, a, b, c|z, scenario),        1.0);
        constraint.inc(CmiTriplet(implicits, a, b|c, z, scenario),       -1.0);
        break;

    case MONOTONE_COND:
        // I(a,b|z) >= I(a|c,z)
        constraint.inc(CmiTriplet(implicits, a|b, a|b, z, scenario),      1.0);
        constraint.inc(CmiTriplet(implicits, a, a, c|z, scenario),       -1.0);
        break;

    case MONOTONE_MUT:
        // I(a;b,c|z) >= I(a;b|z)
        constraint.inc(CmiTriplet(implicits, a, b|c, z, scenario),        1.0);
        constraint.inc(CmiTriplet(implicits, a, b, z, scenario),         -1.0);
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
    }

    return constraint;
}

int ShannonProofSimplifier::add_rule(const Rule& r)
{
    auto [z, a, b, c] = r.subsets;

    auto [it, inserted] = rule_indices.insert(std::make_pair(r, coin.num_cols));
    int idx = it->second;
    if (!inserted)
        return idx;

    auto constraint_cmi = r.get_constraint(implicits_by_scenario[r.scenario]);

    // Don't add trivial rules. Also, don't add rules that are identical to just CMI >= 0 rules.
    bool skip = true;
    for (const auto& [cmi, v] : constraint_cmi.entries)
        if (!cmi.is_zero() && (v < 0.0 || constraint_cmi.is_equality))
        {
            skip = false;
            break;
        }
    skip = (skip || r.is_trivial(constraint_cmi));

    if (skip)
    {
        rule_indices.erase(it);
        return -1;
    }

    SparseVectorT<Symbol> constraint;
    constraint.is_equality = constraint_cmi.is_equality;
    for (const auto& [cmi, v] : constraint_cmi.entries)
        constraint.entries[cmi] = v;
    add_constraint(constraint);
    return idx;
}

int ShannonProofSimplifier::add_constraint(const SparseVectorT<Symbol>& c, double cost)
{
    bool eq = c.is_equality;

    std::vector<int> indices(c.entries.size());
    std::vector<double> values(c.entries.size());
    int count = 0;
    for (const auto& [cmi, v] : c.entries) {
        indices[count] = get_row_index(cmi);
        values[count++] = v;
    }

    int idx = coin.add_col_lb(0.0, cost, count, indices.data(), values.data());
    if (eq)
        coin.add_col_ub(0.0, -cost, count, indices.data(), values.data());

    return idx;
}

int ShannonProofSimplifier::get_row_index(Symbol t, bool allow_equality)
{
    auto [it, inserted] = cmi_indices.insert(std::make_pair(t, coin.num_rows));
    int idx = it->second;
    if (!inserted)
        return idx;

    coin.add_row_ub(0.0);
    if (allow_equality && std::holds_alternative<int>(t))
        coin.add_row_lb(0.0);
    return idx;
}

ShannonProofSimplifier::ShannonProofSimplifier(const ShannonTypeProof& orig_proof_) :
    orig_proof(orig_proof_),
    var_names_by_scenario(orig_proof.variables[0].var_names_by_scenario),
    scenario_names(orig_proof.variables[0].scenario_names),
    real_var_names(orig_proof.variables[0].real_var_names),
    implicits_by_scenario(orig_proof.variables[0].implicits_by_scenario),
    cmi_constraints(orig_proof.cmi_constraints)
{
    if (!orig_proof)
        return;

    cmi_constraints.insert(cmi_constraints.end(), orig_proof.cmi_constraints_redundant.begin(),
                           orig_proof.cmi_constraints_redundant.end());

    cost = 0.0;

    // Translate the old proof into CMI notation.
    SparseVectorT<CmiTriplet> cmi_usage;
    for (auto [i, coeff] : orig_proof.dual_solution.entries)
    {
        if (i == 0 || coeff == 0.0)
            continue;

        if (i <= orig_proof.regular_constraints.size())
        {
            const auto& c = orig_proof.regular_constraints[i - 1];
            cmi_coefficients[c] = coeff;
            cmi_usage.inc(c, coeff);

            cost += coeff * (c.complexity_cost() + rule_cost);
        }
        else
        {
            // Use the original CMI representation of the custom rules.
            int j = i - orig_proof.regular_constraints.size() - 1;
            custom_rule_coefficients[j] = coeff;

            const auto& c = cmi_constraints[j];
            for (const auto& [s, v] : c.entries)
                if (const auto* cmi = std::get_if<CmiTriplet>(&s))
                    cmi_usage.inc(*cmi, coeff * v);
            cost += std::abs(coeff) * custom_rule_complexity_cost(c);
        }
    }

    // Also use the original CMI representation of the objective.
    for (const auto& [s, v] : orig_proof.cmi_objective.entries)
        if (const auto* cmi = std::get_if<CmiTriplet>(&s))
            cmi_usage.inc(*cmi, v);

    // Add rules to convert every CMI into individual entropies.
    for (const auto& [t, v] : cmi_usage.entries)
    {
        Rule r;
        if (t[0] == t[1] && t[2] == 0)
            // Already a single entropy;
            continue;

        const auto& implicits = implicits_by_scenario[t.scenario];

        // I(a;b|c) = I(a,c) + I(b,c) - I(a,b,c) - I(c)
        r = Rule(implicits, Rule::CMI_DEF_I, 0, t[0], t[1], t[2], t.scenario);

        rule_coefficients[r] = -v;
        cost += std::abs(v) * r.complexity_cost(implicits);
    }

    std::cout << "Simplifying from cost " << cost << '\n';
}

void ShannonProofSimplifier::add_all_rules()
{
    // Add rules (other than CMI non negativity, which is implicit.)
    for (int scenario = 0; scenario < scenario_names.size(); ++scenario)
    {
        const auto& implicits = implicits_by_scenario[scenario];
        int num_vars = var_names_by_scenario[scenario].size();
        int full_set = (1 << num_vars) - 1;

        for (int z = 0; z < full_set; ++z)
        {
            for (int a : util::skip_n(util::disjoint_subsets(z, full_set), 1))
            {
                for (int b : util::skip_n(util::disjoint_subsets(z, full_set), 1))
                {
                    for (int c : util::disjoint_subsets(z|a|b, full_set))
                    {
                        if (a <= b)
                        {
                            add_rule(Rule(implicits, Rule::CMI_DEF_I, z, a, b, c, scenario));
                            add_rule(Rule(implicits, Rule::CHAIN, z, a, b, c, scenario));
                        }

                        add_rule(Rule(implicits, Rule::MUTUAL_CHAIN, z, a, b, c, scenario));
                    }

                    if ((a & b) == 0)
                        for (int c : util::disjoint_subsets(z|a, full_set))
                            add_rule(Rule(implicits, Rule::MONOTONE_COND, z, a, b, c, scenario));

                    for (int c : util::disjoint_subsets(z|b, full_set))
                        add_rule(Rule(implicits, Rule::MONOTONE_MUT, z, a, b, c, scenario));
                }

                for (int c : util::skip_n(util::disjoint_subsets(z|a, full_set), 1))
                    add_rule(Rule(implicits, Rule::MONOTONE_COND, z, a, 0, c, scenario));
            }
        }
    }
}

void ShannonProofSimplifier::add_adjacent_rules(CmiTriplet t)
{
    const auto& implicits = implicits_by_scenario[t.scenario];
    int num_vars = var_names_by_scenario[t.scenario].size();
    int full_set = (1 << num_vars) - 1;

    // Handle symmetry of CMI through brute force. (I.e., flip the CMI and try again).
    for (int flip = 0; flip < 2; ++flip)
    {
        // CMI_DEF_I:

        // I(a;b|c,z)
        for (int z : util::all_subsets(t[2], full_set))
            add_rule(Rule(implicits, Rule::CMI_DEF_I, z, t[0], t[1], t[2] & ~z, t.scenario));

        if (t[0] == t[1])
        {
            // I(a,c|z)
            for (int b : util::disjoint_subsets(t[2], full_set))
                for (int c : util::all_subsets(t[0] & ~b, full_set))
                    add_rule(Rule(implicits, Rule::CMI_DEF_I, t[2], t[0] & ~c, b, c, t.scenario));

            // I(b,c|z)
            for (int a : util::disjoint_subsets(t[2], full_set))
                for (int c : util::all_subsets(t[0] & ~a, full_set))
                    add_rule(Rule(implicits, Rule::CMI_DEF_I, t[2], a, t[0] & ~c, c, t.scenario));

            // I(a,b,c|z)
            for (int a : util::all_subsets(t[0], full_set))
                for (int b : util::all_subsets(t[0], full_set))
                    add_rule(Rule(implicits, Rule::CMI_DEF_I, t[2], a, b, t[0] & ~(a|b), t.scenario));
        }

        // CHAIN:

        // I(a;b|c,z)
        for (int z : util::all_subsets(t[2], full_set))
            add_rule(Rule(implicits, Rule::CHAIN, z, t[0], t[1], t[2] & ~z, t.scenario));

        // I(a,c;b,c|z)
        for (int c : util::all_subsets(t[0] & t[1], full_set))
            add_rule(Rule(implicits, Rule::CHAIN, t[2], t[0] & ~c, t[1] & ~c, c, t.scenario));

        // MUTUAL_CHAIN:

        // I(a;c|z)
        for (int b : util::disjoint_subsets(t[1] | t[2], full_set))
            add_rule(Rule(implicits, Rule::MUTUAL_CHAIN, t[2], t[0], b, t[1], t.scenario));

        // I(a;b|c,z)
        for (int z : util::all_subsets(t[2], full_set))
            add_rule(Rule(implicits, Rule::MUTUAL_CHAIN, z, t[0], t[1], t[2] & ~z, t.scenario));

        // I(a;b,c|z)
        for (int c : util::all_subsets(t[1], full_set))
            add_rule(Rule(implicits, Rule::MUTUAL_CHAIN, t[2], t[0], t[1] & ~c, c, t.scenario));

        // MONOTONE_COND:

        if (t[0] == t[1])
        {
            // I(a,b|z)
            for (int a : util::all_subsets(t[0], full_set))
                for (int c : util::disjoint_subsets(t[2] | a, full_set))
                    add_rule(Rule(implicits, Rule::MONOTONE_COND, t[2], a, t[0] & ~a, c, t.scenario));

            // I(a|c,z)
            for (int z : util::all_subsets(t[2], full_set))
                for (int b : util::disjoint_subsets(t[0] | z, full_set))
                    add_rule(Rule(implicits, Rule::MONOTONE_COND, z, t[0], b, t[2] & ~z, t.scenario));
        }

        // MONOTONE_MUT:

        // I(a;b,c|z)
        for (int c : util::all_subsets(t[1], full_set))
            add_rule(Rule(implicits, Rule::MONOTONE_MUT, t[2], t[0], t[1] & ~c, c, t.scenario));

        // I(a;b|z)
        for (int c : util::disjoint_subsets(t[1] | t[2], full_set))
            add_rule(Rule(implicits, Rule::MONOTONE_MUT, t[2], t[0], t[1], c, t.scenario));

        // Repeat with symmetrical CMI, if necessary.
        if (t[0] == t[1])
            break;
        std::swap(t[0], t[1]);
    }
}

// Applies the rules that were excluded from add_adjacent_rules because they result in a quadratic
// (in full_set) branching factor.
void ShannonProofSimplifier::add_adjacent_rules_quadratic(CmiTriplet t)
{
    const auto& implicits = implicits_by_scenario[t.scenario];
    int num_vars = var_names_by_scenario[t.scenario].size();
    int full_set = (1 << num_vars) - 1;

    if (t[0] != t[1])
        return;

    // CMI_DEF_I:
    // I(c|z)
    for (int a : util::disjoint_subsets(t[0] | t[2], full_set))
        for (int b : util::disjoint_subsets(t[0] | t[2], full_set))
            add_rule(Rule(implicits, Rule::CMI_DEF_I, t[2], a, b, t[0], t.scenario));

    // CHAIN:
    // I(c|z)
    for (int a : util::disjoint_subsets(t[0] | t[2], full_set))
        for (int b : util::disjoint_subsets(t[0] | t[2], full_set))
            add_rule(Rule(implicits, Rule::CHAIN, t[2], a, b, t[0], t.scenario));
}

// Optimize complexity of proof. Note that here L0 norm (weight if rule used) is approximated by
// L1 norm (weight proportional to use).
bool ShannonProofSimplifier::simplify(int depth)
{
    if (!*this)
        return false;

    std::unique_ptr<OsiClpSolverInterface> si(new OsiClpSolverInterface());
    coin = CoinOsiProblem(*si);
    cmi_indices.clear();
    rule_indices.clear();

    if (depth == -1)
        add_all_rules();
    else
    {
        // Add all existing rules.
        for (const auto& [cmi, v] : cmi_coefficients)
            get_row_index(cmi);
        for (const auto& [r, v] : rule_coefficients)
            add_rule(r);

        // Include all custom constraints, not just those that were used.
        for (int i = 0; i < cmi_constraints.size(); ++i)
        {
            const auto& c = cmi_constraints[i];
            for (const auto& [cmi, v] : c.entries)
                get_row_index(cmi);
        }

        // And include the objective of course.
        for (auto [cmi, v] : orig_proof.cmi_objective.entries)
            get_row_index(cmi);

        // TODO: can probably reduce the problem size using some kind of MITM, by only adding rules
        // that can map back to two different original CMI triplets.
        std::map<Symbol, int> prev_prev_cmi;
        for (int i = 0; i < depth; ++i)
        {
            std::map<Symbol, int> prev_cmi = cmi_indices;

            for (auto [s, v] : prev_cmi)
                if (const auto* cmi = std::get_if<CmiTriplet>(&s))
                    add_adjacent_rules(*cmi);
            if (i > 0)
                for (auto [s, v] : prev_prev_cmi)
                    if (const auto* cmi = std::get_if<CmiTriplet>(&s))
                        add_adjacent_rules_quadratic(*cmi);

            prev_prev_cmi = std::move(prev_cmi);
        }
    }

    // Constraint to force the constant term to match the original proof.
    std::vector<int> const_indices;
    std::vector<double> const_values;

    // Include columns for the custom constraints.
    std::vector<int> custom_constraint_indices;
    for (int i = 0; i < cmi_constraints.size(); ++i)
    {
        const auto& c = cmi_constraints[i];

        double v_const = 0.0;
        if (i < orig_proof.custom_constraints.size())
            v_const = orig_proof.custom_constraints[i].get(0);

        custom_constraint_indices.push_back(add_constraint(c, custom_rule_complexity_cost(c)));

        if (v_const != 0.0)
        {
            if (c.is_equality)
            {
                const_indices.push_back(coin.num_cols - 2);
                const_values.push_back(v_const);
            }
            const_indices.push_back(coin.num_cols - 1);
            const_values.push_back(v_const);
        }
    }

    double const_coeff_value = orig_proof.objective.get(0) - orig_proof.dual_solution.get(0);
    int const_row = coin.add_row_fixed(const_coeff_value, const_indices.size(), const_indices.data(), const_values.data());

    // Use the original CMI representation of the objective.
    for (auto [cmi, v] : orig_proof.cmi_objective.entries)
    {
        int row = get_row_index(cmi);
        coin.rowub[row] += v;
    }

    coin.obj.resize(coin.num_cols, 0.0);
    double obj_offset = 0.0;

    // cols are easy:
    for (auto [r, col] : rule_indices)
    {
        double cost = r.complexity_cost(implicits_by_scenario[r.scenario]);
        coin.obj[col] += cost;
        if (r.is_equality())
            coin.obj[col + 1] -= cost;
    }

    // rows have to be sent through the constraint map to get the objective.

    std::vector<double> row_obj(coin.num_rows, 0.0);
    for (auto [t, row] : cmi_indices)
    {
        double cost = complexity_cost(t) + rule_cost;
        row_obj[row] -= cost;
        obj_offset += cost * coin.rowub[row];
    }

    std::vector<double> col_obj_from_row_obj(coin.num_cols, 0.0);
    coin.constraints.transposeTimes(row_obj.data(), col_obj_from_row_obj.data());
    for (int i = 0; i < coin.num_cols; ++i)
        coin.obj[i] += col_obj_from_row_obj[i];

    coin.load_problem_into_solver(*si);
    //si->writeLp("simplify_debug");

    // Set the initial solution status based on the old proof.

    auto cstat = std::make_unique<int[]>(coin.num_cols);
    auto rstat = std::make_unique<int[]>(coin.num_rows);
    for (auto [r, i] : rule_indices)
    {
        double v = rule_coefficients.contains(r) ? rule_coefficients.at(r) : 0.0;
        cstat[i] = v > 0.0 ? 1 : 3;
        if (r.is_equality())
            cstat[i + 1] = v < 0.0 ? 1 : 2;
    }

    for (int i = 0; i < cmi_constraints.size(); ++i)
    {
        const auto& orig_constraint = cmi_constraints[i];
        int j = custom_constraint_indices.at(i);
        double v = custom_rule_coefficients.contains(i) ? custom_rule_coefficients.at(i) : 0.0;

        cstat[j] = v > 0.0 ? 1 : 3;
        if (orig_constraint.is_equality)
            cstat[j + 1] = v < 0.0 ? 1 : 2;
    }

    for (auto [t, i] : cmi_indices)
    {
        if (const auto* cmi = std::get_if<CmiTriplet>(&t))
        {
            double v = cmi_coefficients.contains(*cmi) ? cmi_coefficients.at(*cmi) : 0.0;
            rstat[i] = v > 0.0 ? 1 : 3; // At upper bound, but 2 and 3 are swapped for rows.
        }
        else
            rstat[const_row] = 2; // Could be either 2 or 3, because this is a fixed variable.
    }

    rstat[const_row] = 2; // Could be either 2 or 3, because this is a fixed variable.

    si->setBasisStatus(cstat.get(), rstat.get());

    si->setLogLevel(3);
    si->getModelPtr()->setPerturbation(50);
    //std::cout << si->getModelPtr()->perturbation() << '\n';

    std::cout << "Setting OsiDoDualInInitial: " << si->setHintParam(OsiDoDualInInitial, false, OsiHintDo) << '\n';
    std::cout << "Setting OsiPrimalTolerance: " << si->setDblParam(OsiPrimalTolerance, 1e-9) << '\n';
    //std::cout << "Setting OsiDualTolerance: " << si->setDblParam(OsiDualTolerance, 1e-9) << '\n';
    si->initialSolve();

    if (!si->isProvenOptimal()) {
        throw std::runtime_error("ShannonProofSimplifier: Failed to solve LP.");
    }

    double new_cost = si->getObjValue() + obj_offset;
    std::cout << "Simplified to cost " << new_cost << '\n';

    const double* col_sol = si->getColSolution();
    std::vector<double> row_sol(coin.num_rows, 0.0);
    coin.constraints.times(col_sol, row_sol.data());

    auto old_cmi_coefficients         = std::move(cmi_coefficients);
    auto old_rule_coefficients        = std::move(rule_coefficients);
    auto old_custom_rule_coefficients = std::move(custom_rule_coefficients);
    cmi_coefficients.clear();
    rule_coefficients.clear();
    custom_rule_coefficients.clear();

    for (auto [t, row] : cmi_indices)
    {
        if (const auto* cmi = std::get_if<CmiTriplet>(&t))
        {
            double coeff = coin.rowub[row] - row_sol[row];
            if (std::abs(coeff) > eps)
                cmi_coefficients[*cmi] = coeff;
        }
    }

    for (auto [r, col] : rule_indices)
    {
        double coeff = col_sol[col];
        if (r.is_equality())
            coeff += col_sol[col + 1];

        if (std::abs(coeff) > eps)
            rule_coefficients[r] = coeff;
    }

    for (int i = 0; i < cmi_constraints.size(); ++i)
    {
        const auto& orig_constraint = cmi_constraints[i];
        double coeff = col_sol[custom_constraint_indices.at(i)];
        if (orig_constraint.is_equality)
            coeff += col_sol[custom_constraint_indices.at(i) + 1];

        if (std::abs(coeff) > eps)
            custom_rule_coefficients[i] = coeff;
    }

    bool changed = (cost - new_cost > eps);
    cost = new_cost;

    return changed;
}

ShannonProofSimplifier::operator SimplifiedShannonProof()
{
    if (!*this)
        return SimplifiedShannonProof();

    SimplifiedShannonProof proof;
    proof.implicits_by_scenario = &implicits_by_scenario;
    proof.initialized = true;

    coin = CoinOsiProblem(OsiClpSolverInterface());
    cmi_indices.clear();
    rule_indices.clear();

    if (orig_proof.dual_solution.get(0) != 0.0)
        proof.dual_solution.inc(0, orig_proof.dual_solution.get(0));

    for (auto [t, v] : cmi_coefficients)
    {
        int idx = get_row_index(t, false);
        proof.regular_constraints.emplace_back(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<0>(), idx));
        proof.dual_solution.inc(idx + 1, v);
    }
    int n = coin.num_rows;

    for (auto [r, v] : rule_coefficients)
    {
        proof.regular_constraints.emplace_back(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<1>(), r));

        auto c = r.get_constraint(implicits_by_scenario[r.scenario]);
        for (const auto& [cmi, v] : c.entries)
            get_row_index(cmi, false);

        proof.dual_solution.inc(++n, v);
    }

    for (auto [i, v] : custom_rule_coefficients)
    {
        const auto& orig_constraint = cmi_constraints[i];

        proof.custom_constraints.emplace_back();
        auto& constraint = proof.custom_constraints.back();
        constraint.is_equality = orig_constraint.is_equality;

        if (i < orig_proof.custom_constraints.size())
            constraint.inc(0, orig_proof.custom_constraints[i].get(0));
        for (const auto& [cmi, v2] : orig_constraint.entries)
            constraint.inc(get_row_index(cmi, false) + 1, v2);

        proof.dual_solution.inc(++n, v);
    }

    proof.variables.resize(coin.num_rows);
    for (auto [t, i] : cmi_indices)
    {
        proof.variables[i] = std::visit(overload {
            [&](const CmiTriplet& cmi)
            {
                return ExtendedShannonVar{
                    cmi, &var_names_by_scenario, &scenario_names,
                    &real_var_names, &implicits_by_scenario[cmi.scenario]
                };
            },
            [&](const int& v)
            {
                return ExtendedShannonVar{
                    CmiTriplet{}, &var_names_by_scenario, &scenario_names,
                    &real_var_names, nullptr, v
                };
            }
        }, t);
    }

    if (orig_proof.objective.get(0) != 0.0)
        proof.objective.inc(0, orig_proof.objective.get(0));
    for (auto [cmi, v] : orig_proof.cmi_objective.entries)
        proof.objective.inc(cmi_indices.at(cmi) + 1, v);

    return proof;
}

std::ostream& operator<<(std::ostream& out, ExtendedShannonVar t)
{
    if (t.real_var >= 0)
        return out << (*t.real_var_names)[t.real_var];

    if (t[0] > t[1])
        std::swap(t[0], t[1]);

    const auto& var_names = (*t.var_names_by_scenario)[t.scenario];
    out << (t[0] == t[1] ? 'H' : 'I') << (*t.scenario_names)[t.scenario] << '(';
    print_var_subset(out, t[0], var_names);

    if (t[1] != t[0])
    {
        out << "; ";
        print_var_subset(out, t[1], var_names);
    }

    if (t[2])
    {
        out << " | ";
        print_var_subset(out, t[2], var_names);
    }

    out << ')';
    return out;
}

bool ExtendedShannonRule::print(std::ostream& out, const ExtendedShannonVar* vars, double scale) const
{
    const std::vector<std::vector<std::string>>* var_names_by_scenario = vars[0].var_names_by_scenario;
    const std::vector<std::string>* scenario_names = vars[0].scenario_names;
    const ImplicitRules* implicits = nullptr;
    for (size_t i = 0; !implicits; ++i)
        if (vars[i].scenario == scenario)
            implicits = vars[i].implicits;

    auto c = get_constraint(*implicits);

    bool first = true;
    for (const auto& [cmi, v] : c.entries)
    {
        if (scale == 0.0 || v == 0.0 || cmi.is_zero())
        {
            std::cout << "Skipped: " << v << ", " << cmi[0] << ", " << cmi[1] << ", " << cmi[2] << '\n';
            continue;
        }
        print_coeff(out, scale * v, first);
        out << ExtendedShannonVar{cmi, var_names_by_scenario, scenario_names, nullptr, implicits};
        first = false;
    }

    if (first)
    {
        std::cout << "Trivial rule! " << scale << ", " << type << ", " << subsets[0] << ", " <<
            subsets[1] << ", " << subsets[2] << ", " << subsets[3] << ", " << c.entries.size() << '\n';
        return false;
    }

    if (is_equality())
        out << " == 0";
    else
        out << " >= 0";
    return true;
}

OrderedSimplifiedShannonProof SimplifiedShannonProof::order() const
{
    if (!*this)
        return OrderedSimplifiedShannonProof();

    OsiClpSolverInterface si;
    CoinOsiProblem coin(si, true);

    std::vector<int> constraint_map;
    MatrixT<Symbol> used_constraints;

    for (auto [i, v] : dual_solution.entries)
    {
        assert(v != 0.0);

        // Ignore constant terms.
        if (i == 0)
            continue;

        constraint_map.push_back(i);
        used_constraints.emplace_back();

        if (i <= regular_constraints.size())
        {
            used_constraints.back() = std::visit(overload {
                [&](const NonNegativityRule& r)
                {
                    SparseVectorT<Symbol> c;
                    c.inc(variables[r.v], 1.0);
                    return c;
                },
                [&](const ExtendedShannonRule& r)
                {
                    SparseVectorT<CmiTriplet> c = r.get_constraint((*implicits_by_scenario)[r.scenario]);
                    SparseVectorT<Symbol> constraint;
                    constraint.is_equality = c.is_equality;
                    for (const auto& [cmi, v] : c.entries)
                        constraint.entries[cmi] = v;
                    return constraint;
                }
            }, regular_constraints[i - 1]);

            for (auto& [cmi, v2] : used_constraints.back().entries)
                v2 *= v;
        }
        else
        {
            for (auto [j, coeff] : custom_constraints[i - regular_constraints.size() - 1].entries)
            {
                if (j == 0 || coeff == 0.0)
                    // Ignore constant and zero terms.
                    continue;

                used_constraints.back().inc(variables[j - 1], v * coeff);
            }
        }
    }

    // How big can each partial sum get.
    std::vector<double> min_partial_sums, max_partial_sums;

    std::map<Symbol, int> used_terms;
    for (const auto& constraint : used_constraints)
    {
        for (const auto& [t, val] : constraint.entries)
        {
            auto [it, inserted] = used_terms.insert({t, used_terms.size()});

            double min_v = std::min(val, 0.0);
            double max_v = std::max(val, 0.0);

            if (inserted)
            {
                min_partial_sums.push_back(min_v);
                max_partial_sums.push_back(max_v);
            }
            else
            {
                int idx = it->second;
                min_partial_sums[idx] += min_v;
                max_partial_sums[idx] += max_v;
            }
        }
    }

    const int steps = used_constraints.size();
    const int terms = used_terms.size();

    std::vector<int> integer_vars;
    std::vector<int> indices;
    std::vector<double> values;

    // Requirements that the partial sums (defined below) are be correct.
    const int partial_sum_correctness_start = coin.num_rows;
    for (int i = 0; i < steps - 1; ++i)
        for (int j = 0; j < terms; ++j)
            coin.add_row_fixed(0.0);

    // Requirements for the partial sum nonzero flags (defined below) to be correct.
    const int nonzero_partial_sum_correctness_start = coin.num_rows;
    for (int i = 0; i < steps - 1; ++i)
        for (int j = 0; j < terms; ++j)
        {
            coin.add_row_lb(0.0);
            coin.add_row_ub(0.0);
        }

    // Variables for the partial sums after each step (except for the last).
    const int partial_sums_start = coin.num_cols;
    for (int i = 0; i < steps - 1; ++i)
    {
        for (int j = 0; j < terms; ++j)
        {
            indices.clear(); values.clear();
            indices.push_back(partial_sum_correctness_start + i * terms + j);
            values.push_back(-1.0);
            if (i < steps - 2)
            {
                indices.push_back(partial_sum_correctness_start + (i + 1) * terms + j);
                values.push_back(1.0);
            }

            indices.push_back(nonzero_partial_sum_correctness_start + (i * terms + j) * 2);
            values.push_back(-1.0);
            indices.push_back(nonzero_partial_sum_correctness_start + (i * terms + j) * 2 + 1);
            values.push_back(-1.0);

            coin.add_col_free(0.0, indices.size(), indices.data(), values.data());
        }
    }

    // Variables for each step and constraint, for whether the constraint is placed at that step.
    // Don't bother with the last step, as it can be inferred from the rest.
    const int step_to_constraint_start = coin.num_cols;
    for (int i = 0; i < steps - 1; ++i)
    {
        for (int j = 0; j < steps; ++j)
        {
            indices.clear(); values.clear();
            for (const auto& [t, v] : used_constraints[j].entries)
            {
                indices.push_back(partial_sum_correctness_start + i * terms + used_terms.at(t));
                values.push_back(v);
            }

            int col_idx = coin.add_col(0.0, 1.0, 0.0, indices.size(), indices.data(), values.data());
            integer_vars.push_back(col_idx);
        }
    }

    // TODO: Maybe better to allow variations in the scale at which each rule is used, as long as
    // only 1 rule is used at a time. That way this optimization could simplify cases where two
    // different proofs got mixed together.

    // Variables for each step and term (except for the last), for whether the term is present in
    // the partial sum after this step. The objective is to minimize these variables.
    const int nonzero_partial_sums_start = coin.num_cols;
    for (int i = 0; i < steps - 1; ++i)
    {
        for (int j = 0; j < terms; ++j)
        {
            indices.clear(); values.clear();
            indices.push_back(nonzero_partial_sum_correctness_start + (i * terms + j) * 2);
            values.push_back(max_partial_sums[j]);
            indices.push_back(nonzero_partial_sum_correctness_start + (i * terms + j) * 2 + 1);
            values.push_back(min_partial_sums[j]);

            int col_idx = coin.add_col(0.0, 1.0, 1.0, indices.size(), indices.data(), values.data());
            integer_vars.push_back(col_idx);
        }
    }

    // Require that the variables for each step and constraint form a permutation.
    indices.resize(steps);
    values.clear();
    values.resize(steps, 1.0);
    for (int i = 0; i < steps; ++i)
    {
        // Sums within each step
        if (i < steps - 1)
        {
            for (int j = 0; j < steps; ++j)
                indices[j] = step_to_constraint_start + i * steps + j;
            coin.add_row_fixed(1.0, steps, indices.data(), values.data());
        }

        // Sums between steps
        for (int j = 0; j < steps - 1; ++j)
            indices[j] = step_to_constraint_start + j * steps + i;
        coin.add_row_ub(1.0, steps - 1, indices.data(), values.data());
    }

    coin.load_problem_into_solver(si);
    si.setInteger(integer_vars.data(), integer_vars.size());

    // Make sure that the integer variables are counted as binary.
    for (auto v : integer_vars)
        assert(si.isBinary(v));

    std::string temp_dir = "/tmp/Citip.XXXXXXXX";
    assert(mkdtemp(temp_dir.data()));

    auto problem_filename = temp_dir + "/order_problem";

    //si.writeLp("order_debug");
    si.writeMps(problem_filename.c_str());
    problem_filename += ".mps.gz";

    si.setHintParam(OsiDoReducePrint);

    int threads = std::thread::hardware_concurrency();
    std::string threads_str = std::to_string(threads);

    bool succeeded = false;
    double obj;
    const  double* sol = nullptr;
    std::unique_ptr<double[]> sol_storage;
    if (false)
    {
        // CLP
        si.branchAndBound();
        succeeded = si.isProvenOptimal();
        obj = si.getObjValue();
        sol = si.getColSolution();
    }
    else if (false)
    {
        // CBC

        CbcModel model(si);
        CbcSolverUsefulData solverData;
        CbcMain0(model, solverData);

        // For some reason it's faster to save & load the model.
        const char* argv[] = {"", "-threads", threads_str.c_str(), "-import", problem_filename.c_str(), "-solve"};
        CbcMain1(6, argv, model, [](CbcModel *currentSolver, int whereFrom) -> int { return 0; },
                 solverData);

        succeeded = model.isProvenOptimal();
        obj = model.getObjValue();
        sol = model.bestSolution();
    }
    else if (true)
    {
        // SCIP
        SCIP* scip_;
        SCIPcreate(&scip_);
        auto scip_deleter = [&](SCIP* ptr) { SCIPfree(&ptr); };
        std::unique_ptr<SCIP, decltype(scip_deleter)> scip(scip_, scip_deleter);

        SCIPincludeDefaultPlugins(scip.get());

        SCIPreadProb(scip.get(), problem_filename.c_str(), nullptr);
        if (SCIPgetStage(scip.get()) == SCIP_STAGE_PROBLEM)
            succeeded = true;

        if (succeeded)
        {
            SCIPsolve(scip.get());
            succeeded = (SCIPgetStage(scip.get()) == SCIP_STAGE_SOLVED);
        }

        if (succeeded)
        {
            SCIP_SOL* scip_sol = SCIPgetBestSol(scip.get());
            obj = SCIPgetSolOrigObj(scip.get(), scip_sol);

            SCIP_VAR** scip_vars = SCIPgetOrigVars(scip.get());
            int n_scip_vars = SCIPgetNOrigVars(scip.get());
            assert(n_scip_vars == coin.num_cols);

            // SCIP mangles the variable order, but sorting should fix this.
            std::sort(scip_vars, scip_vars + n_scip_vars, [&](SCIP_VAR* a, SCIP_VAR* b) {
                return strcmp(SCIPvarGetName(a), SCIPvarGetName(b)) < 0;
            });

            for (int i = 0; i < n_scip_vars; ++i)
                assert(std::atol(SCIPvarGetName(scip_vars[i]) + 1) == i);

            sol_storage.reset(new double[n_scip_vars]);
            SCIPgetSolVals(scip.get(), scip_sol, n_scip_vars, scip_vars, sol_storage.get());
            sol = sol_storage.get();
        }
    }

    std::filesystem::remove_all(temp_dir);

    if (!succeeded) {
        throw std::runtime_error("OrderedSimplifiedShannonProof: Failed to solve ILP.");
    }

    std::cout << "Reordered to cost " << obj << '\n';

    OrderedSimplifiedShannonProof output(*this);
    if (dual_solution.get(0) != 0.0)
        output.order.push_back(0);

    std::vector<bool> added(steps, false);
    for (int i = 0; i < steps - 1; ++i)
    {
        for (int j = 0; j < steps; ++j)
        {
            if (sol[step_to_constraint_start + i * steps + j] > 0.5)
            {
                assert(!added[j]);
                added[j] = true;
                output.order.push_back(constraint_map[j]);
                break;
            }
        }
    }

    // Last step is the one that didn't get used.
    int j;
    for (j = 0; j < steps; ++j)
        if (!added[j])
        {
            output.order.push_back(constraint_map[j]);
            break;
        }

    for (++j; j < steps; ++j)
        assert(added[j]);

    return output;
}

std::ostream& operator<<(std::ostream& out, const OrderedSimplifiedShannonProof& proof)
{
    if (!proof)
    {
        out << "FALSE";
        return out;
    }

    for (int i : proof.order)
        proof.print_step(out, i, proof.dual_solution.get(i));

    out << "\n => ";
    proof.print_custom_constraint(out, proof.objective);
    out << '\n';

    return out;
}


//----------------------------------------
// globals
//----------------------------------------

ParserOutput parse(const std::vector<std::string>& exprs)
{
    ParserOutput out;
    for (int row = 0; row < exprs.size(); ++row) {
        const std::string& line = exprs[row];
        std::istringstream in(line);
        yy::scanner scanner(&in);
        yy::parser parser(&scanner, &out);
        try {
            int result = parser.parse();
            if (result != 0) {
                // Not sure if this can even happen
                throw std::runtime_error("Unknown parsing error");
            }
        }
        catch (yy::parser::syntax_error& e) {
            // For undefined tokens, bison currently just tells us something
            // like 'unexpected $undefined' without printing the offending
            // character. This is much more useful:
            int col = e.location.begin.column;
            int len = 1 + e.location.end.column - col;
            // TODO: The reported location is not entirely satisfying. Any
            // chances for improvement?
            std::string new_message = sprint_all(
                    e.what(), "\n",
                    "in row ", row, " col ", col, ":\n\n"
                    "    ", line, "\n",
                    "    ", std::string(col-1, ' '), std::string(len, '^'));
            throw yy::parser::syntax_error(e.location, new_message);
        }
    }

    out.process();

    return move(out);
}


// TODO: implement optimization as in Xitip: collapse variables that only
// appear together
