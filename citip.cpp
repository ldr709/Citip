#include <math.h>       // NAN
#include <utility>      // move
#include <sstream>      // istringstream
#include <stdexcept>    // runtime_error
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
        //std::cout << "Adding index " << (size - 1) << '\n';
        vec.resize(size, def);
        vec.back() = val;
    }
    else if (!vec.empty())
        vec.resize(size, def);
}

int CoinOsiProblem::add_row(double lb, double ub, int count, int* indices, double* values)
{
    int idx = num_rows++;

    //std::cout << "Adding row " << idx << '\n';
    //for (int i = 0; i < count; ++i)
    //    std::cout << '(' << idx << ", " << indices[i] << ')' << " = " << values[i] << '\n';

    constraints.appendRow(count, indices, values);
    append_with_default(rowlb, num_rows, lb, -infinity);
    append_with_default(rowub, num_rows, ub, infinity);

    return idx;
}

int CoinOsiProblem::add_col(double lb, double ub, double obj_, int count, int* indices, double* values)
{
    int idx = num_cols++;

    //std::cout << "Adding col " << idx << '\n';
    //for (int i = 0; i < count; ++i)
    //    std::cout << '(' << indices[i] << ", " << idx << ')' << " = " << values[i] << '\n';

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


void ShannonTypeProblem::add_elemental_inequalities(int num_vars, int num_scenarios)
{

    // Identify each variable with its index i from I = {0, 1, ..., N-1}.
    // Then entropy is a real valued set function from the power set of
    // indices P = 2**I. The value for the empty set can be defined to be
    // zero and is irrelevant. Therefore the dimensionality of the problem
    // is 2**N-1 (times num_scenarios).
    int dim_per_scenario = (1<<num_vars) - 1;

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
            indices[0] = scenario;
            values[0] = 1;
            coin.add_row_lb(0.0, 1, indices, values);
            row_to_cmi.emplace_back(CmiTriplet{1,1,1, scenario});
            continue;
        }

        // index of the entropy component corresponding to the joint entropy of
        // all variables. NOTE: since the left-most column is not used, the
        // variables involved in a joint entropy correspond exactly to the bit
        // representation of its index.
        size_t all = dim_per_scenario;

        // Add all elemental conditional entropy positivities, i.e. those of
        // the form H(X_i|X_c)>=0 where c = ~ {i}:
        for (i = 0; i < num_vars; ++i) {
            int c = all ^ (1 << i);
            indices[0] = dim_per_scenario * scenario + all - 1;
            indices[1] = dim_per_scenario * scenario + c - 1;
            values[0] = +1;
            values[1] = -1;
            int row = coin.add_row_lb(0.0, 2, indices, values);
            row_to_cmi.emplace_back(CmiTriplet{1 << i, 1 << i, c, scenario});
        }

        // Add all elemental conditional mutual information positivities, i.e.
        // those of the form I(X_a:X_b|X_K)>=0 where a,b not in K
        for (a = 0; a < num_vars-1; ++a) {
            for (b = a+1; b < num_vars; ++b) {
                int A = 1 << a;
                int B = 1 << b;
                for (i = 0; i < sub_dim; ++i) {
                    int K = skip_bit(skip_bit(i, a), b);
                    indices[0] = dim_per_scenario * scenario + (A|K) - 1;
                    indices[1] = dim_per_scenario * scenario + (B|K) - 1;
                    indices[2] = dim_per_scenario * scenario + (A|B|K) - 1;
                    indices[3] = dim_per_scenario * scenario + (K) - 1;
                    values[0] = +1;
                    values[1] = +1;
                    values[2] = -1;
                    values[3] = -1;
                    int row = coin.add_row_lb(0.0, K ? 4 : 3, indices, values);
                    row_to_cmi.emplace_back(CmiTriplet{A,B,K, scenario});
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
    const int dim_per_scenario = (1<<var_names.size()) - 1;

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
    // Here, it is calculated as the alternating sum of (conditional)
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

    int num_subsets = 1 << num_parts;
    int c = get_set_index(q.cond);
    // Start at i=1 because i=0 which corresponds to H(empty set) gives no
    // contribution to the sum. Furthermore, the i=0 is already reserved
    // for the constant term for our purposes.
    for (int set = 1; set < num_subsets; ++set) {
        int a = 0;
        int s = -1;
        for (int i = 0; i < num_parts; ++i) {
            if (set & 1<<i) {
                a |= set_indices[i];
                s = -s;
            }
        }
        v.inc(scenario * dim_per_scenario + (a|c), s*coef);
    }
    if (c)
        v.inc(scenario * dim_per_scenario + c, -coef);

    // Also write the term using conditional mutual informations. That is, don't go all the way down
    // to individual entropies.
    if (num_parts == 1)
    {
        cmi_v.inc(CmiTriplet{set_indices[0], set_indices[0], c, scenario}, coef);
    }
    else
    {
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
            cmi_v.inc(CmiTriplet{set_indices[num_parts-2], set_indices[num_parts-1], z, scenario}, s*coef);
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
    return idx;
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
    statement_list.emplace_back(std::in_place_index_t<MARKOV_CHAIN>(), move(mi));
}
void ParserOutput::markov_chain(ast::MarkovChain mc)
{
    statement_list.emplace_back(std::in_place_index_t<MUTUAL_INDEPENDENCE>(), move(mc));
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
            [&](const ast::VarCore& vc)
            {
                if (vc.scenario != "")
                    add_scenario(vc.scenario);

                for (const auto& vl : vc.lists)
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
                for (const auto& sc: is.scenarios)
                    add_scenario(sc);

                for (const auto& vl : is.views)
                    add_symbols(vl);
            }
        }, s);

    if (scenario_names.empty())
        add_scenario("");

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
    const int dim_per_scenario = (1<<var_names.size()) - 1;
    bool is_inquiry = inquiries.empty();

    bool has_wildcard_scenario = false;
    for (const ast::Expression& side : {re.left, re.right})
    {
        for (const auto& term : side)
        {
            if (!term.quantity.parts.lists.empty() && term.quantity.parts.scenario == "")
            {
                has_wildcard_scenario = true;
                std::cout << "has_wildcard_scenario\n";
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
    const int first_scenario = mi.scenario == "" ? 0 : scenarios.at(mi.scenario);
    const int last_scenario = mi.scenario == "" ? scenario_names.size() : first_scenario + 1;
    const int dim_per_scenario = (1<<var_names.size()) - 1;

    bool is_inquiry = inquiries.empty();

    for (int scenario = first_scenario; scenario < last_scenario; ++scenario)
    {
        // 0 = H(a) + H(b) + H(c) + … - H(a,b,c,…)
        int all = 0;
        SparseVector v;
        SparseVectorT<CmiTriplet> cmi_v;
        cmi_v.is_equality = v.is_equality = true;
        for (auto&& vl : mi.lists) {
            int idx = get_set_index(vl);
            all |= idx;
            v.inc(scenario * dim_per_scenario + idx, 1);
            cmi_v.inc(CmiTriplet{idx,idx,0, scenario}, 1);
        }
        v.inc(scenario * dim_per_scenario + all, -1);
        cmi_v.inc(CmiTriplet{all,all,0, scenario}, -1);
        add_relation(move(v), move(cmi_v), is_inquiry);
    }
}

void ParserOutput::process_markov_chain(const ast::MarkovChain& mc)
{
    const int first_scenario = mc.scenario == "" ? 0 : scenarios.at(mc.scenario);
    const int last_scenario = mc.scenario == "" ? scenario_names.size() : first_scenario + 1;
    const int dim_per_scenario = (1<<var_names.size()) - 1;

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
            cmi_v.inc(CmiTriplet{a,c,b, scenario}, 1);
            v.inc(scenario * dim_per_scenario + (a|b), 1);
            v.inc(scenario * dim_per_scenario + (c|b), 1);
            v.inc(scenario * dim_per_scenario + (b), -1);
            v.inc(scenario * dim_per_scenario + (a|b|c), -1);
            add_relation(move(v), move(cmi_v), is_inquiry);
        }
    }
}

void ParserOutput::process_function_of(const ast::FunctionOf& fo)
{
    const int first_scenario = fo.scenario == "" ? 0 : scenarios.at(fo.scenario);
    const int last_scenario = fo.scenario == "" ? scenario_names.size() : first_scenario + 1;
    const int dim_per_scenario = (1<<var_names.size()) - 1;

    bool is_inquiry = inquiries.empty();
    int func = get_set_index(fo.function);
    int of = get_set_index(fo.of);

    for (int scenario = first_scenario; scenario < last_scenario; ++scenario)
    {
        // 0 = H(func|of) = H(func,of) - H(of)
        SparseVector v;
        SparseVectorT<CmiTriplet> cmi_v;
        cmi_v.is_equality = v.is_equality = true;
        cmi_v.inc(CmiTriplet{func,func,of,scenario}, 1);
        v.inc(scenario * dim_per_scenario + (func|of), 1);
        v.inc(scenario * dim_per_scenario + (of), -1);
        add_relation(move(v), move(cmi_v), is_inquiry);
    }
}

void ParserOutput::process_indist(const ast::IndistinguishableScenarios& is)
{
    const auto& scenario_list = is.scenarios.empty() ? scenario_names : is.scenarios;
    const int dim_per_scenario = (1<<var_names.size()) - 1;

    bool is_inquiry = inquiries.empty();

    // Includes redundant pairs of scenarios so that the simplifier can pick the most useful pairs.
    for (auto it0 = scenario_list.begin(); it0 != scenario_list.end(); ++it0)
    {
        int scenario0 = scenarios.at(*it0);
        for (auto it1 = it0 + 1; it1 != scenario_list.end(); ++it1)
        {
            int scenario1 = scenarios.at(*it1);

            for (const auto& view : is.views)
            {
                int full_set = (1 << var_names.size()) - 1;
                int view_set = get_set_index(view);
                int non_view_set = full_set & ~view_set;

                // Require that all entropies defined by the view match between the scenarios.
                // Redundantly include all pairs of CMIs, rather than just the base entropies, so
                // that the simplifier can pick the most useful equalities.
                for (int z : util::disjoint_subsets(non_view_set, full_set))
                {
                    for (int b : util::skip_n(util::disjoint_subsets(non_view_set | z, full_set), 1))
                    {
                        for (int a : util::skip_n(util::disjoint_subsets(non_view_set | z, full_set), 1))
                        {
                            if (a > b)
                                break;
                            if (a != b && (a | b) == b)
                                continue;

                            SparseVector v;
                            SparseVectorT<CmiTriplet> cmi_v;
                            cmi_v.is_equality = v.is_equality = true;
                            v.inc(scenario0 * dim_per_scenario + (a|z), 1.0);
                            v.inc(scenario1 * dim_per_scenario + (a|z), -1.0);
                            v.inc(scenario0 * dim_per_scenario + (b|z), 1.0);
                            v.inc(scenario1 * dim_per_scenario + (b|z), -1.0);
                            v.inc(scenario0 * dim_per_scenario + (a|b|z), -1.0);
                            v.inc(scenario1 * dim_per_scenario + (a|b|z), 1.0);
                            if (z)
                            {
                                v.inc(scenario0 * dim_per_scenario + (z), -1.0);
                                v.inc(scenario1 * dim_per_scenario + (z), 1.0);
                            }

                            cmi_v.inc(CmiTriplet{a,b,z, scenario0}, 1.0);
                            cmi_v.inc(CmiTriplet{a,b,z, scenario1}, -1.0);
                            add_relation(move(v), move(cmi_v), is_inquiry);
                        }
                    }
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
    out << 'v' << v;
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

        for (int j = 1; j <= coin.num_cols; ++j)
        {
            proof.variables.emplace_back(LinearVariable{j});
            proof.regular_constraints.emplace_back(NonNegativityRule{j-1});

            double coeff = col_price[j - 1];
            if (std::abs(coeff) > eps)
                proof.dual_solution.entries[j] = coeff;
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
                                       std::vector<std::string> scenario_names_) :
    LinearProblem(),
    random_var_names(move(random_var_names_)),
    scenario_names(move(scenario_names_))
{
    int num_vars = random_var_names.size();
    check_num_vars(num_vars);
    add_columns(scenario_names.size() * ((1<<num_vars) - 1));
    add_elemental_inequalities(num_vars, scenario_names.size());
}

ShannonTypeProblem::ShannonTypeProblem(const ParserOutput& out) :
    ShannonTypeProblem(out.var_names, out.scenario_names)
{
    for (int i = 0; i < out.constraints.size(); ++i)
        add(out.constraints[i], out.cmi_constraints[i]);
}

void ShannonTypeProblem::add(const SparseVector& v, SparseVectorT<CmiTriplet> cmi_v)
{
    LinearProblem::add(v);
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

void ShannonRule::print(std::ostream& out, const ShannonVar* vars, double scale) const
{
    auto [a, b, z] = CmiTriplet(*this);
    print_coeff(out, scale, true);
    out << 'H' << vars[1].scenario_names[scenario] << '(' << vars[a].print_vars();
    if (a != b)
        out << "; " << vars[b].print_vars();
    if (z != 0)
        out << " | " << vars[z].print_vars();
    out << ") >= 0";
}

ShannonTypeProof ShannonTypeProblem::prove(const SparseVector& I,
                                           SparseVectorT<CmiTriplet> cmi_I)
{
    ShannonTypeProof proof(
        LinearProblem::prove(I, row_to_cmi),
        [&] (const LinearVariable& v) { return ShannonVar{random_var_names, scenario_names, v.id}; },
        [&] (const NonNegativityOrOtherRule<CmiTriplet>& r) -> ShannonRule {
            if (r.index() == 0)
            {
                auto [set, scenario] = var_to_set_and_scenario(std::get<0>(r).v, random_var_names.size());
                return ShannonRule({set, set, 0, scenario});
            }
            else
                return ShannonRule((std::get<1>(r)));
        });
    proof.cmi_constraints = cmi_constraints;
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

    double custom_rule_complexity_cost(const SparseVectorT<CmiTriplet>&) const;

    // How much to use each type of (in)equality:
    std::map<CmiTriplet, double> cmi_coefficients;
    std::map<Rule, double> rule_coefficients;
    std::map<int, double> custom_rule_coefficients;

    double cost;

    const ShannonTypeProof& orig_proof;

    std::map<CmiTriplet, int> cmi_indices; // rows represent conditional mutual informations.
    int get_row_index(CmiTriplet t);

    std::map<Rule, int> rule_indices;
    int add_rule(const Rule& r);

    CoinOsiProblem coin;

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
    for (int d = 1; d <= depth; ++d)
        while (simplifier.simplify(d));

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
// CMI defn. 2                    I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z)
// CMI defn. 3                    I(a;b|z) = I(a|z) - I(a|b,z)
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

const double information_cost                    = 1.0;
const double conditional_information_cost        = 1.1;
const double mutual_information_cost             = 1.5;
const double conditional_mutual_information_cost = 1.6;

// Cost of using the bound I(a;b|c) >= 0 where t == (a, b, c).
double CmiTriplet::complexity_cost() const
{
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

double ExtendedShannonRule::complexity_cost() const
{
    CmiTriplet triplets[5];
    double values[5];
    int count = get_constraint(triplets, values);

    double cost = 0.0;
    for (int i = 0; i < count; ++i)
        cost += triplets[i].complexity_cost();
    return cost;
}


double ShannonProofSimplifier::custom_rule_complexity_cost(const SparseVectorT<CmiTriplet>& c) const
{
    double cost = 0.0;
    for (const auto& [cmi, v] : c.entries)
        cost += cmi.complexity_cost();
    return cost;
}

// Outputs into triplets[] and values[]. I've tried to avoid using any rule that contains
// duplicates, but I might have missed some. These arrays much be at least 5 elements long. Returns
// the size of the constraint
int ExtendedShannonRule::get_constraint(CmiTriplet triplets[], double values[]) const
{
    auto [z, a, b, c] = subsets;

    SparseVector constraint;
    int count;
    switch (type)
    {
    case CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
        triplets[count] = CmiTriplet{a, b, c|z, scenario};       values[count++] =  1.0;
        triplets[count] = CmiTriplet{a|c, a|c, z, scenario};     values[count++] = -1.0;
        triplets[count] = CmiTriplet{b|c, b|c, z, scenario};     values[count++] = -1.0;
        triplets[count] = CmiTriplet{a|b|c, a|b|c, z, scenario}; values[count++] =  1.0;
        triplets[count] = CmiTriplet{c, c, z, scenario};         values[count++] =  1.0;
        break;

    case MI_DEF_I:
        // I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z)
        triplets[count] = CmiTriplet{a, b, z, scenario};         values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, a, z, scenario};         values[count++] = -1.0;
        triplets[count] = CmiTriplet{b, b, z, scenario};         values[count++] = -1.0;
        triplets[count] = CmiTriplet{a|b, a|b, z, scenario};     values[count++] =  1.0;
        break;

    case MI_DEF_CI:
        // I(a;b|z) = I(a|z) - I(a|b,z)
        triplets[count] = CmiTriplet{a, b, z, scenario};         values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, a, z, scenario};         values[count++] = -1.0;
        triplets[count] = CmiTriplet{a, a, b|z, scenario};       values[count++] =  1.0;
        break;

    case CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        triplets[count] = CmiTriplet{c, c, z, scenario};         values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, b, c|z, scenario};       values[count++] =  1.0;
        triplets[count] = CmiTriplet{a|c, b|c, z, scenario};     values[count++] = -1.0;
        break;

    case MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        triplets[count] = CmiTriplet{a, c, z, scenario};         values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, b, c|z, scenario};       values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, b|c, z, scenario};       values[count++] = -1.0;
        break;

    case MONOTONE_COND:
        // I(a,b|z) >= I(a|c,z)
        triplets[count] = CmiTriplet{a|b, a|b, z, scenario};     values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, a, c|z, scenario};       values[count++] = -1.0;
        break;

    case MONOTONE_MUT:
        // I(a;b,c|z) >= I(a;b|z)
        triplets[count] = CmiTriplet{a, b|c, z, scenario};     values[count++] =  1.0;
        triplets[count] = CmiTriplet{a, b, z, scenario};       values[count++] = -1.0;
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
        return -1;
    }

    return count;
}

int ShannonProofSimplifier::add_rule(const Rule& r)
{
    // Validate rule.
    auto [z, a, b, c] = r.subsets;
    switch (r.type)
    {
    case Rule::CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
        if (!((a & z) == 0 && (b & z) == 0 && (c & (z | a | b)) == 0 && a > 0 && a < b && (a | b) != b && c > 0))
            return -1;
        break;

    case Rule::MI_DEF_I:
        // I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z)
        if (!((a & z) == 0 && (b & z) == 0 && a > 0 && a < b && (a | b) != b))
            return -1;
        break;

    case Rule::MI_DEF_CI:
        // I(a;b|z) = I(a|z) - I(a|b,z)
        if (!((a & z) == 0 && (b & z) == 0 && a > 0 && b > 0 && a != b))
            return -1;
        break;

    case Rule::CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        if (!((a & z) == 0 && (b & z) == 0 && (c & (z | a | b)) == 0 && a > 0 && a <= b && c > 0))
            return -1;
        break;

    case Rule::MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        if (!((a & z) == 0 && (b & z) == 0 && (c & (z | a | b)) == 0 && a > 0 && b > 0 && c > 0))
            return -1;
        break;

    case Rule::MONOTONE_COND:
        // I(a,b|z) >= I(a|c,z)
        if (!((a & z) == 0 && (b & z) == 0 && (a & b) == 0 && (c & (z | a)) == 0 && a > 0))
            return -1;
        break;

    case Rule::MONOTONE_MUT:
        // I(a;b,c|z) >= I(a;b|z)
        if (!((a & z) == 0 && (b & z) == 0 && (c & (z | b)) == 0 && a > 0 && b > 0 && (a | b) != a && (a | b) != b))
            return -1;
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
        return -1;
    }

    auto [it, inserted] = rule_indices.insert(std::make_pair(r, coin.num_cols));
    int idx = it->second;
    if (!inserted)
        return idx;

    bool eq = r.is_equality();


    int indices[5];
    CmiTriplet triplets[5];
    double values[5];
    int count = r.get_constraint(triplets, values);
    for (int i = 0; i < count; ++i)
        indices[i] = get_row_index(triplets[i]);

    coin.add_col_lb(0.0, 0.0, count, indices, values);
    if (eq)
        coin.add_col_ub(0.0, 0.0, count, indices, values);

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
    scenario_names(orig_proof.variables[0].scenario_names)
{
    if (!orig_proof)
        return;

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
            cost += coeff * c.complexity_cost();
        }
        else
        {
            // Use the original CMI representation of the custom rules.
            int j = i - orig_proof.regular_constraints.size() - 1;
            custom_rule_coefficients[j] = coeff;

            const auto& c = orig_proof.cmi_constraints[j];
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
        if (t[0] == t[1])
            if (t[2] == 0)
                // Already a single entropy;
                continue;
            else
                // I(a|c) = I(a,c) - I(c)
                r = Rule{Rule::CHAIN, 0, t[0], t[0], t[2], t.scenario};
        else
            if (t[2] == 0)
                // I(a;b) = I(a) + I(b) - I(a,b)
                r = Rule{Rule::MI_DEF_I, 0, t[0], t[1], 0, t.scenario};
            else
                // I(a;b|c) = I(a,c) + I(b,c) - I(a,b,c) - I(c)
                r = Rule{Rule::CMI_DEF_I, 0, t[0], t[1], t[2], t.scenario};

        rule_coefficients[r] = -v;
        cost += std::abs(v) * r.complexity_cost();
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
                    if (a != b)
                        add_rule(Rule{Rule::MI_DEF_CI, z, a, b, 0, scenario});

                    if (a < b && (a | b) != b)
                    {
                        add_rule(Rule{Rule::MI_DEF_I, z, a, b, 0, scenario});

                        for (int c : util::skip_n(util::disjoint_subsets(z|a|b, full_set), 1))
                            add_rule(Rule{Rule::CMI_DEF_I, z, a, b, c, scenario});
                    }

                    for (int c : util::skip_n(util::disjoint_subsets(z|a|b, full_set), 1))
                    {
                        if (a <= b)
                            add_rule(Rule{Rule::CHAIN, z, a, b, c, scenario});
                        add_rule(Rule{Rule::MUTUAL_CHAIN, z, a, b, c, scenario});
                    }

                    if ((a & b) == 0)
                        for (int c : util::disjoint_subsets(z|a, full_set))
                            add_rule(Rule{Rule::MONOTONE_COND, z, a, b, c, scenario});

                    if ((a | b) != a && (a | b) != b)
                        for (int c : util::disjoint_subsets(z|b, full_set))
                            add_rule(Rule{Rule::MONOTONE_MUT, z, a, b, c, scenario});
                }

                for (int c : util::skip_n(util::disjoint_subsets(z|a, full_set), 1))
                    add_rule(Rule{Rule::MONOTONE_COND, z, a, 0, c, scenario});
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
            add_rule(Rule{Rule::CMI_DEF_I, z, t[0], t[1], t[2] & ~z, t.scenario});

        if (t[0] == t[1])
        {
            // I(a,c|z)
            for (int b : util::disjoint_subsets(t[2], full_set))
                for (int c : util::all_subsets(t[0] & ~b, full_set))
                    add_rule(Rule{Rule::CMI_DEF_I, t[2], t[0] & ~c, b, c, t.scenario});

            // I(b,c|z)
            for (int a : util::disjoint_subsets(t[2], full_set))
                for (int c : util::all_subsets(t[0] & ~a, full_set))
                    add_rule(Rule{Rule::CMI_DEF_I, t[2], a, t[0] & ~c, c, t.scenario});

            // I(a,b,c|z)
            for (int a : util::all_subsets(t[0], full_set))
                for (int b : util::all_subsets(t[0], full_set))
                    add_rule(Rule{Rule::CMI_DEF_I, t[2], a, b, t[0] & ~(a|b), t.scenario});

            // I(c|z)
            for (int a : util::disjoint_subsets(t[0] | t[2], full_set))
                for (int b : util::disjoint_subsets(t[0] | t[2], full_set))
                    add_rule(Rule{Rule::CMI_DEF_I, t[2], a, b, t[0], t.scenario});
            // TODO: ^ add a way to control the quadratic rules like I(c|z).
        }

        // MI_DEF_I:

        // I(a;b|z)
        add_rule(Rule{Rule::MI_DEF_I, t[2], t[0], t[1], 0, t.scenario});

        if (t[0] == t[1])
        {
            // I(a|z)
            for (int b : util::disjoint_subsets(t[2], full_set))
                add_rule(Rule{Rule::MI_DEF_I, t[2], t[0], b, 0, t.scenario});

            // I(b|z)
            for (int a : util::disjoint_subsets(t[2], full_set))
                add_rule(Rule{Rule::MI_DEF_I, t[2], a, t[0], 0, t.scenario});

            // I(a,b|z)
            for (int a : util::all_subsets(t[0], full_set))
            {
                for (int b_ : util::all_subsets(a, full_set))
                {
                    int b = (t[0] & ~a) | b_;
                    add_rule(Rule{Rule::MI_DEF_I, t[2], a, b, 0, t.scenario});
                }
            }
        }

        // MI_DEF_CI:

        // I(a;b|z)
        add_rule(Rule{Rule::MI_DEF_CI, t[2], t[0], t[1], 0, t.scenario});

        if (t[0] == t[1])
        {
            // I(a|z)
            for (int b : util::disjoint_subsets(t[2], full_set))
                add_rule(Rule{Rule::MI_DEF_CI, t[2], t[0], b, 0, t.scenario});

            // I(a|b,z)
            for (int b : util::all_subsets(t[2], full_set))
                add_rule(Rule{Rule::MI_DEF_CI, t[2] & ~b, t[0], b, 0, t.scenario});
        }

        // CHAIN:

        // I(c|z)
        if (t[0] == t[1])
        {
            for (int a : util::disjoint_subsets(t[0] | t[2], full_set))
                for (int b : util::disjoint_subsets(t[0] | t[2], full_set))
                    add_rule(Rule{Rule::CHAIN, t[2], a, b, t[0], t.scenario});
            // TODO
        }

        // I(a;b|c,z)
        for (int z : util::all_subsets(t[2], full_set))
            add_rule(Rule{Rule::CHAIN, z, t[0], t[1], t[2] & ~z, t.scenario});

        // I(a,c;b,c|z)
        for (int c : util::all_subsets(t[0] & t[1], full_set))
            add_rule(Rule{Rule::CHAIN, t[2], t[0] & ~c, t[1] & ~c, c, t.scenario});

        // MUTUAL_CHAIN:

        // I(a;c|z)
        for (int b : util::disjoint_subsets(t[1] | t[2], full_set))
            add_rule(Rule{Rule::MUTUAL_CHAIN, t[2], t[0], b, t[1], t.scenario});

        // I(a;b|c,z)
        for (int z : util::all_subsets(t[2], full_set))
            add_rule(Rule{Rule::MUTUAL_CHAIN, z, t[0], t[1], t[2] & ~z, t.scenario});

        // I(a;b,c|z)
        for (int c : util::all_subsets(t[1] & ~t[0], full_set))
            add_rule(Rule{Rule::MUTUAL_CHAIN, t[2], t[0], t[1] & ~c, c, t.scenario});

        // MONOTONE_COND:

        if (t[0] == t[1])
        {
            // I(a,b|z)
            for (int a : util::all_subsets(t[0], full_set))
                for (int c : util::disjoint_subsets(t[2] | a, full_set))
                    add_rule(Rule{Rule::MONOTONE_COND, t[2], a, t[0] & ~a, c, t.scenario});

            // I(a|c,z)
            for (int z : util::all_subsets(t[2], full_set))
                for (int b : util::disjoint_subsets(t[0] | z, full_set))
                    add_rule(Rule{Rule::MONOTONE_COND, z, t[0], b, t[2] & ~z, t.scenario});
        }

        // MONOTONE_MUT:

        // I(a;b,c|z)
        for (int c : util::all_subsets(t[1], full_set))
            add_rule(Rule{Rule::MONOTONE_MUT, t[2], t[0], t[1] & ~c, c, t.scenario});

        // I(a;b|z)
        for (int c : util::disjoint_subsets(t[1] | t[2], full_set))
            add_rule(Rule{Rule::MONOTONE_MUT, t[2], t[0], t[1], c, t.scenario});

        // Repeat with symmetrical CMI, if necessary.
        if (t[0] == t[1])
            break;
        std::swap(t[0], t[1]);
    }
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
        for (int i = 0; i < orig_proof.cmi_constraints.size(); ++i)
        {
            const auto& c = orig_proof.cmi_constraints[i];
            for (const auto& [cmi, v] : c.entries)
                get_row_index(cmi);
        }

        // And include the objective of course.
        for (auto [cmi, v] : orig_proof.cmi_objective.entries)
            get_row_index(cmi);

        for (int i = 0; i < depth; ++i)
        {
            std::map<CmiTriplet, int> cmi_snapshot = cmi_indices;
            for (auto [cmi, v] : cmi_snapshot)
                add_adjacent_rules(cmi);
        }
    }

    // Constraint to force the constant term to match the original proof.
    std::vector<int> const_indices;
    std::vector<double> const_values;

    // Include columns for the custom constraints.
    std::vector<int> custom_constraint_indices;
    for (int i = 0; i < orig_proof.cmi_constraints.size(); ++i)
    {
        const auto& c = orig_proof.cmi_constraints[i];
        const auto& c_ind = orig_proof.custom_constraints[i];
        bool eq = c.is_equality;

        std::vector<int> indices(c.entries.size());
        std::vector<double> values(c.entries.size());
        int count = 0;
        for (const auto& [cmi, v] : c.entries) {
            indices[count] = get_row_index(cmi);
            values[count++] = v;
        }

        double cost = custom_rule_complexity_cost(c);
        custom_constraint_indices.push_back(
            coin.add_col_lb(0.0, cost, count, indices.data(), values.data()));
        if (eq)
            coin.add_col_ub(0.0, -cost, count, indices.data(), values.data());

        double v_const = c_ind.get(0);
        if (v_const != 0.0)
        {
            if (eq)
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
        double cost = r.complexity_cost();
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

    for (int i = 0; i < orig_proof.cmi_constraints.size(); ++i)
    {
        const auto& orig_constraint = orig_proof.cmi_constraints[i];
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

    for (int i = 0; i < orig_proof.cmi_constraints.size(); ++i)
    {
        const auto& orig_constraint = orig_proof.cmi_constraints[i];
        double coeff = col_sol[custom_constraint_indices[i]];
        if (orig_constraint.is_equality)
            coeff += col_sol[custom_constraint_indices[i] + 1];

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
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<0>(), idx + 1));
        proof.dual_solution.inc(idx + 1, v);
    }

    int n = coin.num_rows;
    for (auto [r, v] : rule_coefficients)
    {
        proof.regular_constraints.emplace_back(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<1>(), r));

        CmiTriplet triplets[5];
        double values[5];
        int count = r.get_constraint(triplets, values);
        for (int i = 0; i < count; ++i)
            get_row_index(triplets[i]);

        proof.dual_solution.inc(++n, v);
    }

    for (auto [i, v] : custom_rule_coefficients)
    {
        const auto& orig_constraint = orig_proof.cmi_constraints[i];
        proof.custom_constraints.emplace_back();
        auto& constraint = proof.custom_constraints.back();
        constraint.is_equality = orig_constraint.is_equality;
        if (orig_proof.custom_constraints[i].get(0) != 0.0)
            constraint.inc(0, orig_proof.custom_constraints[i].get(0));
        for (const auto& [cmi, v2] : orig_constraint.entries)
            constraint.inc(get_row_index(cmi) + 1, v2);

        proof.dual_solution.inc(++n, v);
    }

    proof.variables.resize(coin.num_rows);
    for (auto [t, i] : cmi_indices)
        proof.variables[i] = ExtendedShannonVar{t, &random_var_names, &scenario_names};

    for (auto [cmi, v] : orig_proof.cmi_objective.entries)
        proof.objective.inc(cmi_indices.at(cmi) + 1, v);

    return proof;
}

std::ostream& operator<<(std::ostream& out, ExtendedShannonVar t)
{
    if (t[0] > t[1])
        std::swap(t[0], t[1]);

    out << 'I' << (*t.scenario_names)[t.scenario] << '(';
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

void ExtendedShannonRule::print(std::ostream& out, const ExtendedShannonVar* vars, double scale) const
{
    const std::vector<std::string>* random_var_names = vars[1].random_var_names;
    const std::vector<std::string>* scenario_names = vars[1].scenario_names;
    auto [z, a, b, c] = subsets;

    auto print_cmi = [&] (const CmiTriplet& t) {
        out << ExtendedShannonVar {t, random_var_names, scenario_names};
    };

    CmiTriplet triplets[5];
    double values[5];
    int count = get_constraint(triplets, values);

    for (int i = 0; i < count; ++i)
    {
        print_coeff(out, scale * values[i], (i == 0));
        print_cmi(triplets[i]);
    }

    if (is_equality())
        out << " == 0";
    else
        out << " >= 0";
}

OrderedSimplifiedShannonProof SimplifiedShannonProof::order() const
{
    if (!*this)
        return OrderedSimplifiedShannonProof();

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
            CmiTriplet triplets[5];
            double values[5];
            int count = std::visit(overload {
                [&](const NonNegativityRule& r)
                {
                    triplets[0] = variables[r.v - 1]; values[0] = 1.0;
                    return 1;
                },
                [&](const ExtendedShannonRule& r)
                {
                    return r.get_constraint(triplets, values);
                }
            }, regular_constraints[i - 1]);

            for (int i = 0; i < count; ++i)
                used_constraints.back().inc(triplets[i], v * values[i]);
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

    si.writeLp("order_debug");
    si.writeMps("order_problem");

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
        const char* argv[] = {"", "-threads", threads_str.c_str(), "-import", "order_problem.mps.gz", "-solve"};
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

        SCIPreadProb(scip.get(), "order_problem.mps.gz", nullptr);
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
