#include <math.h>       // NAN
#include <utility>      // move
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

static constexpr double eps = 3e-4;


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

static std::array<int, 2> var_to_set_and_scenario(int v, int num_vars)
{
    int dim_per_scenario = (1<<num_vars) - 1;
    return {((v - 1) % dim_per_scenario) + 1, (v - 1) / dim_per_scenario};
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

static int scenario_var(const ImplicitRules& implicits, int num_vars, int scenario, int a)
{
    int dim_per_scenario = (1 << num_vars) - 1;
    return scenario * dim_per_scenario + apply_implicits(implicits, a);
}


void ShannonTypeProblem::add_columns()
{
    for (int i = 1; i < (1<<random_var_names.size()); ++i)
    {
        auto [it, inserted] = column_map.insert({apply_implicits(implicits, i), column_map.size()});
        if (inserted)
            inv_column_map.push_back(it->first);
    }

    // Repeat same pattern for each scenario.
    int dim_per_scenario = (1 << random_var_names.size()) - 1;
    int real_dim_per_scenario = column_map.size();
    for (int scenario = 1; scenario < scenario_names.size(); ++scenario)
    {
        for (int i = 0; i < real_dim_per_scenario; ++i)
        {
            int s = inv_column_map[i];
            int j = dim_per_scenario * scenario + s;
            column_map.insert({j, column_map.size()});
            inv_column_map.push_back(j);
        }
    }

    add_columns(column_map.size());
}

void ShannonTypeProblem::add_elemental_inequalities(int num_vars, int num_scenarios)
{
    // Identify each variable with its index i from I = {0, 1, ..., N-1}.
    // Then entropy is a real valued set function from the power set of
    // indices P = 2**I. The value for the empty set can be defined to be
    // zero and is irrelevant. Therefore the dimensionality of the problem
    // is 2**N-1 (times num_scenarios).

    // After choosing 2 variables there are 2**(N-2) possible subsets of
    // the remaining N-2 variables.
    int sub_dim = 1 << (num_vars-2);

    row_to_cmi.reserve(num_scenarios * (num_vars + (num_vars * (num_vars - 1) / 2) * sub_dim + 1));

    for (int scenario = 0; scenario < num_scenarios; ++scenario)
    {
        // NOTE: We use 1-based indices for variable numbers, because 0 would correspond to H() = 0
        // and so be useless. However, Coin expects 0-based indices, so translation is needed.
        int indices[4];
        double values[4];
        int i, a, b;

        if (num_vars == 1) {
            assert(column_map.size() == 1);
            indices[0] = scenario;
            values[0] = 1;
            coin.add_row_lb(0.0, 1, indices, values);
            row_to_cmi.emplace_back(CmiTriplet(implicits, 1,1,1, scenario));
            continue;
        }

        // index of the entropy component corresponding to the joint entropy of
        // all variables. NOTE: since the left-most column is not used, the
        // variables involved in a joint entropy correspond exactly to the bit
        // representation of its index.
        size_t all = (1<<num_vars) - 1;

        // Add all elemental conditional entropy positivities, i.e. those of
        // the form H(X_i|X_c)>=0 where c = ~ {i}:
        for (i = 0; i < num_vars; ++i) {
            int c = all ^ (1 << i);
            if (all == apply_implicits(implicits, c))
                continue;

            indices[0] = column_map.at(scenario_var(implicits, random_var_names.size(), scenario, all));
            indices[1] = column_map.at(scenario_var(implicits, random_var_names.size(), scenario, c));
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

                    indices[0] = column_map.at(scenario_var(implicits, random_var_names.size(), scenario, A|K));
                    indices[1] = column_map.at(scenario_var(implicits, random_var_names.size(), scenario, B|K));
                    indices[2] = column_map.at(scenario_var(implicits, random_var_names.size(), scenario, A|B|K));
                    values[0] = +1;
                    values[1] = +1;
                    values[2] = -1;
                    if (K)
                    {
                        indices[3] = column_map.at(scenario_var(implicits, random_var_names.size(), scenario, K));
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

void ParserOutput::add_term(SparseVector& v, SparseVectorT<CmiTriplet>& cmi_v,
                            const ast::Term& t, int scenario_wildcard, double scale)
{
    const ast::Quantity& q = t.quantity;
    double coef = scale * t.coefficient;
    int num_parts = q.parts.lists.size();
    if (num_parts == 0) {   // constant
        v.inc(0, coef);
        return;
    }

    const int scenario = q.parts.scenario == "" ? scenario_wildcard : scenarios.at(q.parts.scenario);

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
        set_indices[i] = get_set_index(q.parts.lists[i]);
    int c = get_set_index(q.cond);

    if (num_parts == 1)
    {
        add_cmi(v, cmi_v, CmiTriplet(implicits, set_indices[0], set_indices[0], c, scenario), coef);
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
            add_cmi(v, cmi_v, CmiTriplet(implicits, set_indices[num_parts-2], set_indices[num_parts-1], z, scenario), s*coef);
        }
    }
}

int ParserOutput::get_var_index(const std::string& s)
{
    auto&& it = vars.find(s);
    if (it != vars.end())
        return it->second;
    int next_index = var_names.size();
    check_num_vars(next_index + 1);
    vars[s] = next_index;
    var_names.push_back(s);
    return next_index;
}

int ParserOutput::get_set_index(const ast::VarList& l)
{
    int idx = 0;
    for (auto&& v : l)
        idx |= 1 << get_var_index(v);
    return apply_implicits(implicits, idx);
}

void ParserOutput::add_relation(SparseVector v, SparseVectorT<CmiTriplet> cmi_v, bool is_inquiry)
{
    if (is_inquiry)
    {
        inquiries.push_back(move(v));
        cmi_inquiries.push_back(move(cmi_v));
    }
    else
    {
        constraints.push_back(move(v));
        cmi_constraints.push_back(move(cmi_v));
    }
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
    for (const auto& s : statement_list)
        std::visit(overload {
            [&](const ast::Relation& r)
            {
                for (const ast::Expression& side : {r.left, r.right})
                {
                    for (const auto& term : side)
                    {
                        if (term.quantity.parts.scenario != "")
                            add_scenario(term.quantity.parts.scenario);

                        for (const auto& vl : term.quantity.parts.lists)
                            add_symbols(vl);
                        add_symbols(term.quantity.cond);
                    }
                }
            },
            [&](const ast::MarkovChain& mc)
            {
                if (mc.scenario != "")
                    add_scenario(mc.scenario);

                for (const auto& vl : mc.lists)
                    add_symbols(vl);
            },
            [&](const ast::MutualIndependence& mi)
            {
                if (mi.sets.scenario != "")
                    add_scenario(mi.sets.scenario);

                for (const auto& vl : mi.sets.lists)
                    add_symbols(vl);
            },
            [&](const ast::FunctionOf& f)
            {
                if (f.scenario != "")
                    add_scenario(f.scenario);

                add_symbols(f.function);
                add_symbols(f.of);
            },
            [&](const ast::IndistinguishableScenarios& is)
            {
                for (const auto& group: is)
                {
                    for (const auto& sc: group.scenarios)
                        add_scenario(sc);

                    add_symbols(group.view);
                }
            }
        }, s);

    if (scenario_names.empty())
        add_scenario("");

    // Retrieve the implicit function_of statements.
    for (const auto& s : statement_list)
    {
        if (s.index() != FUNCTION_OF)
            continue;
        const auto& fo = std::get<FUNCTION_OF>(s);
        if (!fo.implicit)
            continue;
        if (fo.scenario != "")
            throw std::runtime_error("Implicit function-of rules must apply to all scenarios.");

        int func = get_set_index(fo.function);
        int of = get_set_index(fo.of);
        implicits.funcs.push_back({func, of});
    }

    // Apply the functions_of statements to each other, so that later they can be resolved faster.
    bool changed;
    do
    {
        changed = false;
        for (auto& f : implicits.funcs)
        {
            int new_func = f.func | (apply_implicits(implicits, f.func | f.of) & ~f.of);
            if (new_func != f.func)
            {
                f.func = new_func;
                changed = true;
            }
        }
    } while (changed);

    // Retrieve the implicit independence statements.
    for (const auto& s : statement_list)
    {
        if (s.index() != MUTUAL_INDEPENDENCE)
            continue;
        const auto& mi = std::get<MUTUAL_INDEPENDENCE>(s);
        if (!mi.implicit)
            continue;
        if (mi.sets.scenario != "")
            throw std::runtime_error("Implicit independence rules must apply to all scenarios.");

        std::vector<int> sets(mi.sets.lists.size());
        for (int i = 0; i < sets.size(); ++i)
            // Implicit function rules are applied to the independence sets here.
            sets[i] = get_set_index(mi.sets.lists[i]);

        std::vector<int> set_forward_unions(sets.size(), 0);
        for (int i = 1; i < sets.size(); ++i)
            set_forward_unions[i] = set_forward_unions[i - 1] | sets[i - 1];

        std::vector<int> set_backward_unions(sets.size(), 0);
        for (int i = sets.size() - 1; i > 0; --i)
            set_backward_unions[i - 1] = set_backward_unions[i] | sets[i];

        for (int i = 0; i < sets.size(); ++i)
            implicits.indeps.push_back(
                ImplicitIndependence {sets[i], set_forward_unions[i] | set_backward_unions[i]});
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

void ParserOutput::add_symbols(const ast::VarList& vl)
{
    for (auto&& v : vl)
        get_var_index(v);
}

void ParserOutput::add_cmi(SparseVector& v, SparseVectorT<CmiTriplet>& cmi_v,
                           CmiTriplet t, double coeff) const
{
    t = CmiTriplet(implicits, t[0], t[1], t[2], t.scenario);
    if (t.is_zero())
        return;

    auto [a, b, z] = t;
    v.inc(scenario_var(implicits, var_names.size(), t.scenario, a|z), coeff);
    v.inc(scenario_var(implicits, var_names.size(), t.scenario, b|z), coeff);
    v.inc(scenario_var(implicits, var_names.size(), t.scenario, a|b|z), -coeff);
    if (z)
        v.inc(scenario_var(implicits, var_names.size(), t.scenario, z), -coeff);

    cmi_v.inc(t, coeff);
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
    bool is_inquiry = inquiries.empty();

    bool has_wildcard_scenario = false;
    for (const ast::Expression& side : {re.left, re.right})
    {
        for (const auto& term : side)
        {
            if (!term.quantity.parts.lists.empty() && term.quantity.parts.scenario == "")
            {
                has_wildcard_scenario = true;
                goto done;
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
        SparseVectorT<CmiTriplet> cmi_v;
        cmi_v.is_equality = v.is_equality = (re.relation == ast::REL_EQ);
        for (auto&& term : re.left)
            add_term(v, cmi_v, term, wildcard_scenario, l_sign);
        for (auto&& term : re.right)
            add_term(v, cmi_v, term, wildcard_scenario, r_sign);
        add_relation(move(v), move(cmi_v), is_inquiry);
    }
}

void ParserOutput::process_mutual_independence(const ast::MutualIndependence& mi)
{
    const int first_scenario = mi.sets.scenario == "" ? 0 : scenarios.at(mi.sets.scenario);
    const int last_scenario = mi.sets.scenario == "" ? scenario_names.size() : first_scenario + 1;

    bool is_inquiry = inquiries.empty();

    std::vector<int> set_indices(mi.sets.lists.size());
    for (int i = 0; i < set_indices.size(); ++i)
        set_indices[i] = get_set_index(mi.sets.lists[i]);

    for (int scenario = first_scenario; scenario < last_scenario; ++scenario)
    {
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
            SparseVectorT<CmiTriplet> cmi_v;
            cmi_v.is_equality = v.is_equality = true;
            for (auto subset : subsets) {
                // Check if subset contains the sentinal element saying that this subset is left out
                // of the partition.
                if ((subset >> set_indices.size()) & 1)
                    continue;

                int idx = convert_set(subset);
                all |= idx;
                add_cmi(v, cmi_v, CmiTriplet(implicits, idx,idx,0, scenario), 1);
                //std::cout << ExtendedShannonVar {CmiTriplet(implicits, idx,idx,0, scenario), &var_names, &scenario_names, &implicits} << " + ";
            }
            add_cmi(v, cmi_v, CmiTriplet(implicits, all,all,0, scenario), -1);
            //std::cout << "0 = " << ExtendedShannonVar {CmiTriplet(implicits, all,all,0, scenario), &var_names, &scenario_names, &implicits} << '\n';
            add_relation(move(v), move(cmi_v), is_inquiry);
        };

        // Skip the redundant constraints entirely and just do the full subset if there's way too
        // many.
        if (set_indices.size() > 13)
        {
            partition_constraint(*util::partitions(set_indices.size()).begin());
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
        if (mi.implicit)
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

                    SparseVectorT<CmiTriplet> cmi_v;
                    cmi_v.is_equality = true;
                    cmi_v.inc(cmi, 1.0);
                    cmi_constraints_redundant.push_back(move(cmi_v));
                }
            }
        }
    }
}

void ParserOutput::process_markov_chain(const ast::MarkovChain& mc)
{
    const int first_scenario = mc.scenario == "" ? 0 : scenarios.at(mc.scenario);
    const int last_scenario = mc.scenario == "" ? scenario_names.size() : first_scenario + 1;

    bool is_inquiry = inquiries.empty();

    for (int scenario = first_scenario; scenario < last_scenario; ++scenario)
    {
        int a = 0;
        for (int i = 0; i+2 < mc.lists.size(); ++i) {
            int b, c;
            a |= get_set_index(mc.lists[i+0]);
            b = get_set_index(mc.lists[i+1]);
            c = get_set_index(mc.lists[i+2]);
            // 0 = I(a:c|b) = H(a|b) + H(c|b) - H(a,c|b)
            SparseVector v;
            SparseVectorT<CmiTriplet> cmi_v;
            cmi_v.is_equality = v.is_equality = true;
            add_cmi(v, cmi_v, CmiTriplet(implicits, a,c,b, scenario), 1);
            add_relation(move(v), move(cmi_v), is_inquiry);
        }
    }
}

void ParserOutput::process_function_of(const ast::FunctionOf& fo)
{
    // Already handled.
    if (fo.implicit)
        return;

    bool is_inquiry = inquiries.empty();
    int func = get_set_index(fo.function);
    int of = get_set_index(fo.of);

    const int first_scenario = fo.scenario == "" ? 0 : scenarios.at(fo.scenario);
    const int last_scenario = fo.scenario == "" ? scenario_names.size() : first_scenario + 1;

    for (int scenario = first_scenario; scenario < last_scenario; ++scenario)
    {
        // 0 = H(func|of) = H(func,of) - H(of)
        SparseVector v;
        SparseVectorT<CmiTriplet> cmi_v;
        cmi_v.is_equality = v.is_equality = true;
        add_cmi(v, cmi_v, CmiTriplet(implicits, func,func,of,scenario), 1);
        add_relation(move(v), move(cmi_v), is_inquiry);
    }
}

void ParserOutput::process_indist(const ast::IndistinguishableScenarios& is)
{
    bool is_inquiry = inquiries.empty();

    // Require that all entropies defined by the view match between the scenarios.
    auto indist_views = [&](int scenario0, const ast::VarList& view0,
                            int scenario1, const ast::VarList& view1,
                            bool all_redundant)
    {
        assert(view0.size() == view1.size());
        assert(view0.size() > 0);

        int full_set = (1 << view0.size()) - 1;

        auto convert_set = [&](const ast::VarList& view, int set)
        {
            int out_set = 0;
            for (int i = 0; i < view.size(); ++i)
                if ((set >> i) & 1)
                    out_set |= (1 << get_var_index(view[i]));
            return out_set;
        };

        // Require that all entropies defined by the view match between the scenarios. This is
        // skipped when all_redundant (i.e., when i > 1), as those are implied by the two equalities
        // going through i = 1.
        if (!all_redundant)
            for (int a : util::skip_n(util::disjoint_subsets(0, full_set), 1))
            {
                int a0 = convert_set(view0, a);
                int a1 = convert_set(view1, a);

                CmiTriplet cmi0(implicits, a0, a0, 0, scenario0);
                CmiTriplet cmi1(implicits, a1, a1, 0, scenario1);
                if (cmi0.is_zero() && cmi1.is_zero())
                    continue;

                SparseVector v;
                SparseVectorT<CmiTriplet> cmi_v;
                cmi_v.is_equality = v.is_equality = true;
                add_cmi(v, cmi_v, cmi0, 1.0);
                add_cmi(v, cmi_v, cmi1, -1.0);
                add_relation(move(v), move(cmi_v), is_inquiry);
            }

        // Avoid adding redundant rules.
        std::set<std::array<int, 6>> added_rules;

        // Skip z > 0 when this would make the set of constraints way too big.
        int max_z = (view0.size() > 6 ? 0 : full_set);

        // Skip the redundant constraints entirely if there's way too many.
        if (view0.size() > 14)
            return;

        // Redundantly include all pairs of CMIs, rather than just the base entropies, so
        // that the simplifier can pick the most useful equalities.
        for (int z = 0; z <= max_z; ++z)
        {
            for (int b : util::skip_n(util::disjoint_subsets(z, full_set), 1))
            {
                for (int a : util::skip_n(util::disjoint_subsets(z, full_set), 1))
                {
                    if (a > b)
                        break;
                    if (a != b && (a | b) == b)
                        continue;

                    CmiTriplet cmi0(implicits, convert_set(view0, a), convert_set(view0, b), convert_set(view0, z), scenario0);
                    CmiTriplet cmi1(implicits, convert_set(view1, a), convert_set(view1, b), convert_set(view1, z), scenario1);

                    if (cmi0.is_zero() && cmi1.is_zero())
                        continue;
                    if (!added_rules.insert({cmi0[0], cmi0[1], cmi0[2], cmi1[0], cmi1[1], cmi1[2]}).second)
                        continue;

                    SparseVectorT<CmiTriplet> cmi_v;
                    cmi_v.is_equality = true;
                    cmi_v.inc(cmi0, 1.0);
                    cmi_v.inc(cmi1, -1.0);
                    cmi_constraints_redundant.push_back(move(cmi_v));
                }
            }
        }
    };

    // Includes redundant pairs of scenarios so that the simplifier can pick the most useful pairs.
    int i = 0;
    for (const auto& group0 : is)
    {
        const auto& scenario_list0 = group0.scenarios.empty() ? scenario_names : group0.scenarios;
        for (const auto& scenario_name0 : scenario_list0)
        {
            ++i;
            int scenario0 = scenarios.at(scenario_name0);

            int j = 0;
            for (const auto& group1 : is)
            {
                const auto& scenario_list1 = group1.scenarios.empty() ? scenario_names : group1.scenarios;
                for (const auto& scenario_name1 : scenario_list1)
                {
                    ++j;
                    if (i >= j)
                        continue;
                    int scenario1 = scenarios.at(scenario_name1);

                    indist_views(scenario0, group0.view, scenario1, group1.view, i > 1);
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

LinearProof<> LinearProblem::prove_impl(const SparseVector& I, int num_regular_rules, bool want_proof)
{
    // check for equalities as I>=0 and -I>=0
    if (I.is_equality) {
        std::cout << "Warning: unimplemented equality proof generation.\n";
        SparseVector v2(I);
        v2.is_equality = false;
        if (!check(v2))
            return LinearProof();
        for (auto&& ent : v2.entries)
            ent.second = -ent.second;
        return prove_impl(I, num_regular_rules, want_proof);
    }

    coin.obj.resize(coin.num_cols);
    for (int i = 1; i <= coin.num_cols; ++i)
        coin.obj[i - 1] = I.get(i);

    coin.load_problem_into_solver(*si);
    //si->writeLp("debug");

    si->setLogLevel(3);
    si->getModelPtr()->setPerturbation(50);
    si->initialSolve();

    if (!si->isProvenOptimal()) {
        throw std::runtime_error("LinearProblem: Failed to solve LP.");
    }

    //glp_smcp parm;
    //glp_init_smcp(&parm);
    //parm.msg_lev = GLP_MSG_ERR;
    //parm.meth = GLP_DUAL;
    //
    //int outcome = glp_simplex(lp, &parm);
    //if (outcome != 0) {
    //    throw std::runtime_error(sprint_all(
    //                "Error in glp_simplex: ", outcome));
    //}
    //
    //int status = glp_get_status(lp);
    //if (status == GLP_OPT) {
    //} else if (status == GLP_UNBND) {
    //    return LinearProof();
    //} else {
    //    // I am not sure about the exact distinction of GLP_NOFEAS, GLP_INFEAS,
    //    // GLP_UNDEF, so here is a generic error message:
    //    throw std::runtime_error(sprint_all(
    //                "no feasible solution (status code ", status, ")"
    //                ));
    //}

    // the original check was for the solution (primal variable values)
    // rather than objective value, but let's do it simpler for now (if
    // an optimum is found, it should be zero anyway):
    if (si->getObjValue() + I.get(0) + eps >= 0.0)
    {
        LinearProof proof;
        proof.initialized = true;

        if (!want_proof)
            return proof;

        const double* si_solution = si->getColSolution();
        proof.primal_solution = std::vector<double>(si_solution, si_solution + coin.num_cols);

        proof.objective = I;
        proof.objective.is_equality = false;

        double const_term = si->getObjValue() + I.get(0);
        if (std::abs(const_term) > eps)
            proof.dual_solution.entries[0] = const_term;

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
            int row_type = 0;
            if (coin.rowlb.empty() || coin.rowlb[i] == -coin.infinity)
                row_type = -1;
            if (coin.rowub.empty() || coin.rowub[i] == coin.infinity)
            {
                assert(row_type == 0);
                row_type = 1;
            }

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


ShannonTypeProblem::ShannonTypeProblem(std::vector<std::string> random_var_names_,
                                       std::vector<std::string> scenario_names_,
                                       ImplicitRules implicit_function_ofs_,
                                       std::vector<SparseVectorT<CmiTriplet>> cmi_constraints_redundant_) :
    LinearProblem(),
    random_var_names(move(random_var_names_)),
    scenario_names(move(scenario_names_)),
    implicits(move(implicit_function_ofs_)),
    cmi_constraints_redundant(cmi_constraints_redundant_)
{
    int num_vars = random_var_names.size();
    check_num_vars(num_vars);

    add_columns();
    add_elemental_inequalities(num_vars, scenario_names.size());
}

ShannonTypeProblem::ShannonTypeProblem(const ParserOutput& out) :
    ShannonTypeProblem(out.var_names, out.scenario_names, out.implicits, out.cmi_constraints_redundant)
{
    for (int i = 0; i < out.constraints.size(); ++i)
        add(out.constraints[i], out.cmi_constraints[i]);
}

void ShannonTypeProblem::add(const SparseVector& v, SparseVectorT<CmiTriplet> cmi_v)
{
    SparseVector realV;
    realV.is_equality = v.is_equality;
    for (const auto& [x, y] : v.entries)
        realV.inc(x ? column_map.at(x) + 1 : 0, y);
    LinearProblem::add(realV);
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
    return scenario_names[var_to_set_and_scenario(v, random_var_names.size())[1]];
}

std::ostream& operator<<(std::ostream& out, const ShannonVar::PrintVarsOut& pvo)
{
    int set = var_to_set_and_scenario(pvo.parent.v, pvo.parent.random_var_names.size())[0];
    print_var_subset(out, set, pvo.parent.random_var_names);
    return out;
}

std::ostream& operator<<(std::ostream& out, const ShannonVar& sv)
{
    return out << 'H' << sv.scenario() << '(' << sv.print_vars() << ')';
}

bool ShannonRule::print(std::ostream& out, const ShannonVar* vars, double scale) const
{
    const std::vector<std::string>* random_var_names = &vars[0].random_var_names;
    const std::vector<std::string>* scenario_names = &vars[0].scenario_names;
    const ImplicitRules* implicits = &vars[0].implicits;

    if (scale == 0.0 || is_zero())
        return false;

    print_coeff(out, scale, true);
    out << ExtendedShannonVar {*this, random_var_names, scenario_names, implicits};
    out << " >= 0";
    return true;
}

ShannonTypeProof ShannonTypeProblem::prove(const SparseVector& I,
                                           SparseVectorT<CmiTriplet> cmi_I)
{
    SparseVector realObj;
    realObj.is_equality = I.is_equality;
    for (const auto& [x, y] : I.entries)
        realObj.inc(x ? column_map.at(x) + 1 : 0, y);

    LinearProof lproof = LinearProblem::prove(realObj, row_to_cmi);
    ShannonTypeProof proof(
        lproof,
        [&] (const LinearVariable& v)
        {
            return ShannonVar{
                random_var_names, scenario_names, column_map, implicits, inv_column_map[v.id]
            };
        },
        [&] (const NonNegativityOrOtherRule<CmiTriplet>& r) -> ShannonRule {
            if (r.index() == 0)
            {
                int var = inv_column_map[lproof.variables[std::get<0>(r).v].id];
                auto [set, scenario] = var_to_set_and_scenario(var, random_var_names.size());
                return ShannonRule({set, set, 0, scenario});
            }
            else
                return ShannonRule((std::get<1>(r)));
        });
    proof.cmi_constraints = cmi_constraints;
    proof.cmi_constraints_redundant = cmi_constraints_redundant;
    proof.cmi_objective = move(cmi_I);

    return proof;
}


// Simplify the Shannon bounds in a proof by combining them into conditional mutual informations.
struct ShannonProofSimplifier
{
    typedef ExtendedShannonRule Rule;

    ShannonProofSimplifier() = delete;
    ShannonProofSimplifier(const ShannonTypeProof&);

    bool simplify(int depth);

    operator bool() const { return orig_proof; }
    operator SimplifiedShannonProof();

private:
    void add_all_rules();
    void add_adjacent_rules(CmiTriplet);
    void add_adjacent_rules_quadratic(CmiTriplet);

    double custom_rule_complexity_cost(const SparseVectorT<CmiTriplet>&) const;

    // How much to use each type of (in)equality:
    std::map<CmiTriplet, double> cmi_coefficients;
    std::map<Rule, double> rule_coefficients;
    std::map<int, double> custom_rule_coefficients;

    double cost;

    const ShannonTypeProof& orig_proof;
    MatrixT<CmiTriplet> cmi_constraints;

    std::map<CmiTriplet, int> cmi_indices; // rows represent conditional mutual informations.
    int get_row_index(CmiTriplet t);

    std::map<Rule, int> rule_indices;
    int add_rule(const Rule& r);
    int add_constraint(const SparseVectorT<CmiTriplet>& c, double cost = 0.0);

    CoinOsiProblem coin;

    const ImplicitRules& implicits;
    const std::vector<std::string>& random_var_names;
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

const double information_cost                    = 1.0;
const double conditional_information_cost        = 1.1;
const double mutual_information_cost             = 1.5;
const double conditional_mutual_information_cost = 1.6;

// Cost of using the bound I(a;b|c) >= 0 where t == (a, b, c).
double CmiTriplet::complexity_cost() const
{
    if (is_zero())
        return 0.0;

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
    double cost = 0.0;
    auto c = get_constraint(implicits);

    for (const auto& [cmi, v] : c.entries)
        cost += cmi.complexity_cost();

    // Give a reason to avoid using trivial rules.
    return std::max(cost, 0.1);
}

double ShannonProofSimplifier::custom_rule_complexity_cost(const SparseVectorT<CmiTriplet>& c) const
{
    double cost = 0.0;
    for (const auto& [cmi, v] : c.entries)
        cost += cmi.complexity_cost();

    // Give a reason to avoid using trivial rules.
    return std::max(cost, 0.1);
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

    auto constraint = r.get_constraint(implicits);

    // Don't add trivial rules. Also, don't add rules that are identical to just CMI >= 0 rules.
    bool skip = true;
    for (const auto& [cmi, v] : constraint.entries)
        if (!cmi.is_zero() && (v < 0.0 || constraint.is_equality))
        {
            skip = false;
            break;
        }
    skip = (skip || r.is_trivial(constraint));

    if (skip)
    {
        rule_indices.erase(it);
        return -1;
    }

    add_constraint(constraint);
    return idx;
}

int ShannonProofSimplifier::add_constraint(const SparseVectorT<CmiTriplet>& c, double cost)
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

int ShannonProofSimplifier::get_row_index(CmiTriplet t)
{
    auto [it, inserted] = cmi_indices.insert(std::make_pair(t, coin.num_rows));
    int idx = it->second;
    if (!inserted)
        return idx;

    coin.add_row_ub(0.0);
    return idx;
}

ShannonProofSimplifier::ShannonProofSimplifier(const ShannonTypeProof& orig_proof_) :
    orig_proof(orig_proof_),
    random_var_names(orig_proof.variables[0].random_var_names),
    scenario_names(orig_proof.variables[0].scenario_names),
    implicits(orig_proof.variables[0].implicits),
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

            cost += coeff * std::max(0.1, c.complexity_cost()); // Give a reason to avoid using trivial rules.
        }
        else
        {
            // Use the original CMI representation of the custom rules.
            int j = i - orig_proof.regular_constraints.size() - 1;
            custom_rule_coefficients[j] = coeff;

            const auto& c = cmi_constraints[j];
            for (const auto& [cmi, v] : c.entries)
                cmi_usage.inc(cmi, coeff * v);
            cost += std::abs(coeff) * custom_rule_complexity_cost(c);
        }
    }

    // Also use the original CMI representation of the objective.
    for (const auto& [cmi, v] : orig_proof.cmi_objective.entries)
        cmi_usage.inc(cmi, v);

    // Add rules to convert every CMI into individual entropies.
    for (const auto& [t, v] : cmi_usage.entries)
    {
        Rule r;
        if (t[0] == t[1] && t[2] == 0)
            // Already a single entropy;
            continue;

        // I(a;b|c) = I(a,c) + I(b,c) - I(a,b,c) - I(c)
        r = Rule(implicits, Rule::CMI_DEF_I, 0, t[0], t[1], t[2], t.scenario);

        rule_coefficients[r] = -v;
        cost += std::abs(v) * r.complexity_cost(implicits);
    }

    std::cout << "Simplifying from cost " << cost << '\n';
}

void ShannonProofSimplifier::add_all_rules()
{
    int num_vars = random_var_names.size();
    int full_set = (1 << num_vars) - 1;

    // Add rules (other than CMI non negativity, which is implicit.)
    for (int scenario = 0; scenario < scenario_names.size(); ++scenario)
    {
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
    int num_vars = random_var_names.size();
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
    int num_vars = random_var_names.size();
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
        std::map<CmiTriplet, int> prev_prev_cmi;
        for (int i = 0; i < depth; ++i)
        {
            std::map<CmiTriplet, int> prev_cmi = cmi_indices;

            for (auto [cmi, v] : prev_cmi)
                add_adjacent_rules(cmi);
            if (i > 0)
                for (auto [cmi, v] : prev_prev_cmi)
                    add_adjacent_rules_quadratic(cmi);

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
        double cost = r.complexity_cost(implicits);
        coin.obj[col] += cost;
        if (r.is_equality())
            coin.obj[col + 1] -= cost;
    }

    // rows have to be sent through the constraint map to get the objective.

    std::vector<double> row_obj(coin.num_rows, 0.0);
    for (auto [t, row] : cmi_indices)
    {
        double cost = t.complexity_cost();
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
        double v = cmi_coefficients.contains(t) ? cmi_coefficients.at(t) : 0.0;
        rstat[i] = v > 0.0 ? 1 : 3; // At upper bound, but 2 and 3 are swapped for rows.
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
        double coeff = coin.rowub[row] - row_sol[row];
        if (std::abs(coeff) > eps)
            cmi_coefficients[t] = coeff;
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
    proof.initialized = true;

    coin = CoinOsiProblem(OsiClpSolverInterface());
    cmi_indices.clear();
    rule_indices.clear();

    if (orig_proof.dual_solution.get(0) != 0.0)
        proof.dual_solution.inc(0, orig_proof.dual_solution.get(0));

    for (auto [t, v] : cmi_coefficients)
    {
        int idx = get_row_index(t);
        proof.regular_constraints.emplace_back(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<0>(), idx));
        proof.dual_solution.inc(idx + 1, v);
    }

    int n = coin.num_rows;
    for (auto [r, v] : rule_coefficients)
    {
        proof.regular_constraints.emplace_back(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<1>(), r));

        auto c = r.get_constraint(implicits);
        for (const auto& [cmi, v] : c.entries)
            get_row_index(cmi);

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
            constraint.inc(get_row_index(cmi) + 1, v2);

        proof.dual_solution.inc(++n, v);
    }

    proof.variables.resize(coin.num_rows);
    for (auto [t, i] : cmi_indices)
        proof.variables[i] = ExtendedShannonVar{t, &random_var_names, &scenario_names, &implicits};

    if (orig_proof.objective.get(0) != 0.0)
        proof.objective.inc(0, orig_proof.objective.get(0));
    for (auto [cmi, v] : orig_proof.cmi_objective.entries)
        proof.objective.inc(cmi_indices.at(cmi) + 1, v);

    return proof;
}

std::ostream& operator<<(std::ostream& out, ExtendedShannonVar t)
{
    if (t[0] > t[1])
        std::swap(t[0], t[1]);

    out << (t[0] == t[1] ? 'H' : 'I') << (*t.scenario_names)[t.scenario] << '(';
    print_var_subset(out, t[0], *t.random_var_names);

    if (t[1] != t[0])
    {
        out << "; ";
        print_var_subset(out, t[1], *t.random_var_names);
    }

    if (t[2])
    {
        out << " | ";
        print_var_subset(out, t[2], *t.random_var_names);
    }

    out << ')';
    return out;
}

bool ExtendedShannonRule::print(std::ostream& out, const ExtendedShannonVar* vars, double scale) const
{
    const std::vector<std::string>* random_var_names = vars[0].random_var_names;
    const std::vector<std::string>* scenario_names = vars[0].scenario_names;
    const ImplicitRules* implicits = vars[0].implicits;

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
        out << ExtendedShannonVar {cmi, random_var_names, scenario_names, implicits};
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

    const ImplicitRules& implicits = *variables[0].implicits;

    OsiClpSolverInterface si;
    CoinOsiProblem coin(si, true);

    std::vector<int> constraint_map;
    MatrixT<CmiTriplet> used_constraints;

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
                    SparseVectorT<CmiTriplet> c;
                    c.inc(variables[r.v], 1.0);
                    return c;
                },
                [&](const ExtendedShannonRule& r)
                {
                    return r.get_constraint(implicits);
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

    std::map<CmiTriplet, int> used_triplets;
    for (const auto& constraint : used_constraints)
    {
        for (const auto& [t, val] : constraint.entries)
        {
            auto [it, inserted] = used_triplets.insert({t, used_triplets.size()});

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
    const int terms = used_triplets.size();

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
                indices.push_back(partial_sum_correctness_start + i * terms + used_triplets.at(t));
                values.push_back(v);
            }

            int col_idx = coin.add_col(0.0, 1.0, 0.0, indices.size(), indices.data(), values.data());
            integer_vars.push_back(col_idx);
        }
    }

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
        throw std::runtime_error("LinearProblem: Failed to solve LP.");
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
    if (out.inquiries.empty()) {
        throw std::runtime_error("undefined information expression");
    }

    return move(out);
}


// TODO: implement optimization as in Xitip: collapse variables that only
// appear together
