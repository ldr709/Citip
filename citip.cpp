#include <math.h>       // NAN
#include <utility>      // move
#include <sstream>      // istringstream
#include <stdexcept>    // runtime_error
#include <iostream>

#include <glpk.h>
#include <coin/OsiClpSolverInterface.hpp>
#include <coin/CoinPackedMatrix.hpp>
#include <coin/CoinPackedVector.hpp>

#include "citip.hpp"
#include "parser.hxx"
#include "scanner.hpp"
#include "common.hpp"

using std::move;
using util::sprint_all;

static constexpr double eps = 1e-5;


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


void ShannonTypeProblem::add_elemental_inequalities(glp_prob* lp, int num_vars)
{
    // NOTE: GLPK uses 1-based indices and never uses the 0th element.
    int indices[5];
    double values[5];
    int i, a, b;

    if (num_vars == 1) {
        indices[1] = 1;
        values[1] = 1;
        int row = glp_add_rows(lp, 1);
        glp_set_row_bnds(lp, row, GLP_LO, 0.0, NAN);
        glp_set_mat_row(lp, row, 1, indices, values);

        row_to_cmi.emplace_back(CmiTriplet{1,1,1});
        return;
    }

    // Identify each variable with its index i from I = {0, 1, ..., N-1}.
    // Then entropy is a real valued set function from the power set of
    // indices P = 2**I. The value for the empty set can be defined to be
    // zero and is irrelevant. Therefore the dimensionality of the problem
    // is 2**N-1.
    int dim = (1<<num_vars) - 1;

    // After choosing 2 variables there are 2**(N-2) possible subsets of
    // the remaining N-2 variables.
    int sub_dim = 1 << (num_vars-2);

    row_to_cmi.reserve(num_vars + (num_vars * (num_vars - 1) / 2) * sub_dim + 1);

    // index of the entropy component corresponding to the joint entropy of
    // all variables. NOTE: since the left-most column is not used, the
    // variables involved in a joint entropy correspond exactly to the bit
    // representation of its index.
    size_t all = dim;

    // Add all elemental conditional entropy positivities, i.e. those of
    // the form H(X_i|X_c)>=0 where c = ~ {i}:
    for (i = 0; i < num_vars; ++i) {
        int c = all ^ (1 << i);
        indices[1] = all;
        indices[2] = c;
        values[1] = +1;
        values[2] = -1;
        int row = glp_add_rows(lp, 1);
        glp_set_row_bnds(lp, row, GLP_LO, 0.0, NAN);
        glp_set_mat_row(lp, row, 2, indices, values);

        row_to_cmi.emplace_back(CmiTriplet{1 << i, 1 << i, c});
    }

    // Add all elemental conditional mutual information positivities, i.e.
    // those of the form I(X_a:X_b|X_K)>=0 where a,b not in K
    for (a = 0; a < num_vars-1; ++a) {
        for (b = a+1; b < num_vars; ++b) {
            int A = 1 << a;
            int B = 1 << b;
            for (i = 0; i < sub_dim; ++i) {
                int K = skip_bit(skip_bit(i, a), b);
                indices[1] = A|K;
                indices[2] = B|K;
                indices[3] = A|B|K;
                indices[4] = K;
                values[1] = +1;
                values[2] = +1;
                values[3] = -1;
                values[4] = -1;
                int row = glp_add_rows(lp, 1);
                glp_set_row_bnds(lp, row, GLP_LO, 0.0, NAN);
                glp_set_mat_row(lp, row, K ? 4 : 3, indices, values);

                row_to_cmi.emplace_back(CmiTriplet{A,B,K});
            }
        }
    }
}


//----------------------------------------
// ParserOutput
//----------------------------------------

void ParserOutput::add_term(SparseVector& v, SparseVectorT<CmiTriplet>& cmi_v,
                            const ast::Term& t, double scale)
{
    const ast::Quantity& q = t.quantity;
    double coef = scale * t.coefficient;
    int num_parts = q.parts.size();
    if (num_parts == 0) {   // constant
        v.inc(0, coef);
        return;
    }

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
        set_indices[i] = get_set_index(q.parts[i]);

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
        v.inc(a|c, s*coef);
    }
    if (c)
        v.inc(c, -coef);

    // Also write the term using conditional mutual informations. That is, don't go all the way down
    // to individual entropies.
    if (num_parts == 1)
    {
        cmi_v.inc(CmiTriplet{set_indices[0], set_indices[0], c}, coef);
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
            cmi_v.inc(CmiTriplet{set_indices[num_parts-2], set_indices[num_parts-1], z}, s*coef);
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

void ParserOutput::relation(ast::Relation re)
{
    bool is_inquiry = inquiries.empty();
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
        add_term(v, cmi_v, term, l_sign);
    for (auto&& term : re.right)
        add_term(v, cmi_v, term, r_sign);
    add_relation(move(v), move(cmi_v), is_inquiry);
}

void ParserOutput::mutual_independence(ast::MutualIndependence mi)
{
    bool is_inquiry = inquiries.empty();
    // 0 = H(a) + H(b) + H(c) + … - H(a,b,c,…)
    int all = 0;
    SparseVector v;
    SparseVectorT<CmiTriplet> cmi_v;
    cmi_v.is_equality = v.is_equality = true;
    for (auto&& vl : mi) {
        int idx = get_set_index(vl);
        all |= idx;
        v.inc(idx, 1);
        cmi_v.inc(CmiTriplet{idx,idx,0}, 1);
    }
    v.inc(all, -1);
    cmi_v.inc(CmiTriplet{all,all,0}, -1);
    add_relation(move(v), move(cmi_v), is_inquiry);

    // // Independence of all disjoint subsets.
    // SparseVectorT<CmiTriplet> cmi_v;
    // cmi_v.is_equality = true;
    // int num_subsets = 1 << mi.size();
    // for (int i = 1; i < num_subsets; ++i)
    // {
    // }
}

void ParserOutput::markov_chain(ast::MarkovChain mc)
{
    bool is_inquiry = inquiries.empty();
    int a = 0;
    for (int i = 0; i+2 < mc.size(); ++i) {
        int b, c;
        a |= get_set_index(mc[i+0]);
        b = get_set_index(mc[i+1]);
        c = get_set_index(mc[i+2]);
        // 0 = I(a:c|b) = H(a|b) + H(c|b) - H(a,c|b)
        SparseVector v;
        SparseVectorT<CmiTriplet> cmi_v;
        cmi_v.is_equality = v.is_equality = true;
        cmi_v.inc(CmiTriplet{a,c,b}, 1);
        v.inc(a|b, 1);
        v.inc(c|b, 1);
        v.inc(b, -1);
        v.inc(a|b|c, -1);
        add_relation(move(v), move(cmi_v), is_inquiry);
    }
}

void ParserOutput::function_of(ast::FunctionOf fo)
{
    bool is_inquiry = inquiries.empty();
    int func = get_set_index(fo.function);
    int of = get_set_index(fo.of);
    // 0 = H(func|of) = H(func,of) - H(of)
    SparseVector v;
    SparseVectorT<CmiTriplet> cmi_v;
    cmi_v.is_equality = v.is_equality = true;
    cmi_v.inc(CmiTriplet{func,func,of}, 1);
    v.inc(func|of, 1);
    v.inc(of, -1);
    add_relation(move(v), move(cmi_v), is_inquiry);
}


//----------------------------------------
// LinearProblem
//----------------------------------------

LinearProblem::LinearProblem()
{
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);
}

LinearProblem::LinearProblem(int num_cols)
    : LinearProblem()
{
    add_columns(num_cols);
}

LinearProblem::~LinearProblem()
{
    glp_delete_prob(lp);
}

void LinearProblem::add_columns(int num_cols)
{
    glp_add_cols(lp, num_cols);
    for (int i = 1; i <= num_cols; ++i) {
        glp_set_col_bnds(lp, i, GLP_LO, 0, NAN);
    }
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
        indices.push_back(ent.first);
        values.push_back(ent.second);
    }

    int kind = v.is_equality ? GLP_FX : GLP_LO;
    int row = glp_add_rows(lp, 1);
    glp_set_row_bnds(lp, row, kind, -v.get(0), NAN);
    glp_set_mat_row(
            lp, row, indices.size(),
            indices.data()-1, values.data()-1);
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

    if (c != 1.0)
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

    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_ERR;
    parm.meth = GLP_DUAL;

    int num_cols = glp_get_num_cols(lp);
    int num_rows = glp_get_num_rows(lp);

    for (int i = 1; i <= num_cols; ++i)
        glp_set_obj_coef(lp, i, I.get(i));

    int outcome = glp_simplex(lp, &parm);
    if (outcome != 0) {
        throw std::runtime_error(sprint_all(
                    "Error in glp_simplex: ", outcome));
    }

    int status = glp_get_status(lp);
    if (status == GLP_OPT) {
        // the original check was for the solution (primal variable values)
        // rather than objective value, but let's do it simpler for now (if
        // an optimum is found, it should be zero anyway):
        if (glp_get_obj_val(lp) >= -I.get(0))
        {
            LinearProof proof;
            proof.initialized = true;

            if (!want_proof)
                return proof;

            proof.objective = I;
            proof.objective.is_equality = false;

            double const_term = glp_get_obj_val(lp) + I.get(0);
            if (std::abs(const_term) > eps)
                proof.dual_solution.entries[0] = const_term;

            for (int j = 1; j <= num_cols; ++j)
            {
                proof.variables.emplace_back(LinearVariable{j});
                proof.regular_constraints.emplace_back(NonNegativityRule{j-1}); // (NonNegativityRule<LinearVariable>{ LinearVariable{i} });

                int stat = glp_get_col_stat(lp, j);
                double coeff = glp_get_col_dual(lp, j);
                if (stat == GLP_NL && std::abs(coeff) > eps)
                    proof.dual_solution.entries[j] = coeff;
            }

            for (int i = 1; i <= num_rows; ++i)
            {
                int stat = glp_get_row_stat(lp, i);
                double coeff = glp_get_row_dual(lp, i);
                if ((stat == GLP_NL || stat == GLP_NS) && std::abs(coeff) > eps)
                    proof.dual_solution.entries[i + num_cols] = coeff;
            }

            std::vector<int> col_idxs;
            std::vector<double> coeffs;
            for (int i = num_regular_rules + 1; i <= num_rows; ++i)
            {
                int num_nonzero_cols = glp_get_mat_row(lp, i, NULL, NULL);
                col_idxs.resize(num_nonzero_cols + 1);
                coeffs.resize(num_nonzero_cols + 1);
                glp_get_mat_row(lp, i, col_idxs.data(), coeffs.data());

                int row_type = glp_get_row_type(lp, i);

                proof.custom_constraints.emplace_back();
                auto& constraint = proof.custom_constraints.back();
                for (int j = 1; j <= num_nonzero_cols; ++j)
                    constraint.inc(col_idxs[j], row_type == GLP_UP ? -coeffs[j] : coeffs[j]);

                double const_offset = 0.0;
                if (row_type == GLP_UP)
                    const_offset = glp_get_row_ub(lp, i);
                else if (row_type == GLP_LO || row_type == GLP_FX)
                    const_offset = -glp_get_row_lb(lp, i);
                if (std::abs(const_offset) > eps)
                    constraint.entries[0] = const_offset;

                constraint.is_equality = (row_type == GLP_FX);
            }

            return proof;
        }
        else
        {
            // TODO: Counterexample.
            return LinearProof();
        }
    }

    if (status == GLP_UNBND) {
        return LinearProof();
    }

    // I am not sure about the exact distinction of GLP_NOFEAS, GLP_INFEAS,
    // GLP_UNDEF, so here is a generic error message:
    throw std::runtime_error(sprint_all(
                "no feasible solution (status code ", status, ")"
                ));
}


ShannonTypeProblem::ShannonTypeProblem(std::vector<std::string> random_var_names_)
    : LinearProblem(), random_var_names(std::move(random_var_names_))
{
    int num_vars = random_var_names.size();
    check_num_vars(num_vars);
    add_columns((1<<num_vars) - 1);
    add_elemental_inequalities(lp, num_vars);
}

ShannonTypeProblem::ShannonTypeProblem(const ParserOutput& out)
    : ShannonTypeProblem(out.var_names)
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

std::ostream& operator<<(std::ostream& out, const ShannonVar::PrintVarsOut& pvo)
{
    print_var_subset(out, pvo.parent.v, pvo.parent.random_var_names);
    return out;
}

std::ostream& operator<<(std::ostream& out, const ShannonVar& sv)
{
    return out << "H(" << sv.print_vars() << ')';
}

void ShannonRule::print(std::ostream& out, const ShannonVar* vars) const
{
    auto [a, b, z] = CmiTriplet(*this);
    out << "H(" << vars[a].print_vars();
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
        [&] (const LinearVariable& v) { return ShannonVar{random_var_names, v.id}; },
        [&] (const NonNegativityOrOtherRule<CmiTriplet>& r) -> ShannonRule {
            if (r.index() == 0)
                return ShannonRule{std::get<0>(r).v, std::get<0>(r).v, 0};
            else
                return ShannonRule(std::get<1>(r));
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

    double cmi_complexity_cost(CmiTriplet t) const;

    // How much to use each type of (in)equality:
    std::map<CmiTriplet, double> cmi_coefficients;
    std::map<Rule, double> rule_coefficients;

    Matrix custom_constraints;
    SparseVector objective;
    const ShannonTypeProof& orig_proof;

    operator SimplifiedShannonProof() const;

    std::map<CmiTriplet, int> cmi_indices; // rows represent conditional mutual informations.
    int get_row_index(CmiTriplet t);

    std::map<Rule, int> rule_indices;
    int add_rule(const Rule& r);

    // Coin problem
    CoinPackedMatrix coin_constraints;
    std::vector<double> coin_collb, coin_colub;
    std::vector<double> coin_rowub;
    std::vector<double> coin_obj;
    double coin_infinity = 0.0;

    const std::vector<std::string>& random_var_names;
};

SimplifiedShannonProof ShannonTypeProof::simplify() const
{
    return ShannonProofSimplifier(*this);
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

const double information_cost                    = 1.0;
const double conditional_information_cost        = 1.5;
const double mutual_information_cost             = 2.0;
const double conditional_mutual_information_cost = 3.0;

// Cost of using the bound I(a;b|c) >= 0 where t == (a, b, c).
double ShannonProofSimplifier::cmi_complexity_cost(CmiTriplet t) const
{
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
    auto [z, a, b, c] = subsets;

    switch (type)
    {
    case CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
        return conditional_mutual_information_cost;

    case MI_DEF_I:
        // I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z)
        return mutual_information_cost;

    case MI_DEF_CI:
        // I(a;b|z) = I(a|z) - I(a|b,z)
        return mutual_information_cost;

    case CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        if (a == b)
            return conditional_information_cost;
        else
            return conditional_mutual_information_cost;

    case MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        return conditional_mutual_information_cost + 1.0;

    case MONOTONE:
        // I(a,b|z) >= I(a|c,z)
        if (b == 0 || c == 0)
            return 0.6;
        else
            return 1.0;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
        return 0.0;
    }
}

int ShannonProofSimplifier::add_rule(const Rule& r)
{
    auto [it, inserted] = rule_indices.insert(std::make_pair(r, coin_collb.size()));
    int idx = it->second;
    if (!inserted)
        return idx;

    bool eq = r.is_equality();
    auto [z, a, b, c] = r.subsets;

    SparseVector constraint;
    switch (r.type)
    {
    case Rule::CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
        constraint.inc(get_row_index({a, b, c|z}),       1.0);
        constraint.inc(get_row_index({a|c, a|c, z}),    -1.0);
        constraint.inc(get_row_index({b|c, b|c, z}),    -1.0);
        constraint.inc(get_row_index({a|b|c, a|b|c, z}), 1.0);
        constraint.inc(get_row_index({c, c, z}),         1.0);
        break;

    case Rule::MI_DEF_I:
        // I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z)
        constraint.inc(get_row_index({a, b, z}),      1.0);
        constraint.inc(get_row_index({a, a, z}),     -1.0);
        constraint.inc(get_row_index({b, b, z}),     -1.0);
        constraint.inc(get_row_index({a|b, a|b, z}),  1.0);
        break;

    case Rule::MI_DEF_CI:
        // I(a;b|z) = I(a|z) - I(a|b,z)
        constraint.inc(get_row_index({a, b, z}),   1.0);
        constraint.inc(get_row_index({a, a, z}),  -1.0);
        constraint.inc(get_row_index({a, a, b|z}), 1.0);
        break;

    case Rule::CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        constraint.inc(get_row_index({c, c, z}),      1.0);
        constraint.inc(get_row_index({a, b, c|z}),    1.0);
        constraint.inc(get_row_index({a|c, b|c, z}), -1.0);
        break;

    case Rule::MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        constraint.inc(get_row_index({a, c, z}),    1.0);
        constraint.inc(get_row_index({a, b, c|z}),  1.0);
        constraint.inc(get_row_index({a, b|c, z}), -1.0);
        break;

    case Rule::MONOTONE:
        // I(a,b|z) >= I(a|c,z)
        constraint.inc(get_row_index({a|b, a|b, z}), 1.0);
        constraint.inc(get_row_index({a, a, c|z}),  -1.0);
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
        return -1;
    }

    int indices[5];
    double values[5];
    int count = 0;
    for (auto [i, v] : constraint.entries)
    {
        indices[count] = i;
        values[count++] = v;
    }

    coin_constraints.appendCol(count, indices, values);
    coin_collb.push_back(0.0);
    coin_colub.push_back(coin_infinity);
    if (eq)
    {
        coin_constraints.appendCol(count, indices, values);
        coin_collb.push_back(-coin_infinity);
        coin_colub.push_back(0.0);
    }

    return idx;
}

int ShannonProofSimplifier::get_row_index(CmiTriplet t)
{
    if (t[0] > t[1])
        std::swap(t[0], t[1]);

    auto [it, inserted] = cmi_indices.insert(std::make_pair(t, coin_rowub.size()));
    int idx = it->second;
    if (!inserted)
        return idx;

    coin_constraints.appendRow(0, nullptr, nullptr);
    coin_rowub.push_back(0.0);

    return idx;
}

// Optimize complexity of proof. Note that here L0 norm (weight if rule used) is approximated by
// L1 norm (weight proportional to use).
ShannonProofSimplifier::ShannonProofSimplifier(const ShannonTypeProof& orig_proof_) :
    orig_proof(orig_proof_),
    random_var_names(orig_proof.variables[0].random_var_names),
    coin_constraints(true, 2.0, 2.0)
{
    if (!orig_proof)
        return;

    OsiSolverInterface* si = new OsiClpSolverInterface();
    coin_infinity = si->getInfinity();

    int num_vars = random_var_names.size();
    check_num_vars(num_vars);
    int full_set = (1 << num_vars) - 1;

    // Add rules (other than CMI non negativity, which is implicit.)
    for (int z = 0; z < full_set; ++z)
    {
        for (int a : util::skip_n(util::disjoint_subsets(z, full_set), 1))
        {
            for (int b : util::skip_n(util::disjoint_subsets(z, full_set), 1))
            {
                if (a != b)
                    add_rule(Rule{Rule::MI_DEF_CI, z, a, b, 0});

                if (a < b)
                {
                    add_rule(Rule{Rule::MI_DEF_I, z, a, b, 0});

                    for (int c : util::skip_n(util::disjoint_subsets(z|a|b, full_set), 1))
                        add_rule(Rule{Rule::CMI_DEF_I, z, a, b, c});
                }

                for (int c : util::skip_n(util::disjoint_subsets(z|a|b, full_set), 1))
                {
                    if (a <= b)
                        add_rule(Rule{Rule::CHAIN, z, a, b, c});
                    add_rule(Rule{Rule::MUTUAL_CHAIN, z, a, b, c});
                }

                if ((a & b) == 0)
                    for (int c : util::disjoint_subsets(z|a, full_set))
                        add_rule(Rule{Rule::MONOTONE, z, a, b, c});
            }

            for (int c : util::skip_n(util::disjoint_subsets(z|a, full_set), 1))
                add_rule(Rule{Rule::MONOTONE, z, a, 0, c});
        }
    }

    // Include exactly the same custom constraints as in the original proof.
    for (auto [i, coeff] : orig_proof.dual_solution.entries)
    {
        if (coeff != 0.0 && i > orig_proof.regular_constraints.size())
        {
            const auto& c = orig_proof.cmi_constraints[i - orig_proof.regular_constraints.size() - 1];
            for (auto [cmi, v] : c.entries)
                coin_rowub[get_row_index(cmi)] -= coeff * v;
        }
    }

    // Instead of using the dual_solution's combination of cmi constraints (which are an arbitrary
    // representation of the objective function), use the original CMI representation of the
    // objective.
    for (auto [cmi, v] : orig_proof.cmi_objective.entries)
    {
        int row = get_row_index(cmi);
        coin_rowub[row] += v;
        objective.inc(row + 1, v);
    }
    objective.inc(0, orig_proof.objective.get(0));

    coin_obj.resize(coin_collb.size(), 0.0);
    double obj_offset = 0.0;

    // cols are easy:
    for (auto [r, col] : rule_indices)
    {
        double cost = r.complexity_cost();
        coin_obj[col] += cost;
        if (r.is_equality())
            coin_obj[col + 1] -= cost;
    }

    // rows have to be sent through the constraint map to get the objective.

    std::vector<double> row_obj(coin_rowub.size(), 0.0);
    for (auto [t, row] : cmi_indices)
    {
        double cost = cmi_complexity_cost(t);
        row_obj[row] -= cost;
        obj_offset += cost * coin_rowub[row];
    }

    std::vector<double> col_obj_from_row_obj(coin_collb.size(), 0.0);
    coin_constraints.transposeTimes(row_obj.data(), col_obj_from_row_obj.data());
    for (int i = 0; i < coin_collb.size(); ++i)
        coin_obj[i] += col_obj_from_row_obj[i];

    si->loadProblem(coin_constraints, coin_collb.data(), coin_colub.data(),
                    coin_obj.data(), nullptr, coin_rowub.data());

    si->writeLp("simplify_debug");

    std::cout << "Setting OsiDoDualInInitial: " << si->setHintParam(OsiDoDualInInitial, false, OsiHintDo) << '\n';
    //std::cout << "Setting OsiDualTolerance: " << si->setDblParam(OsiDualTolerance, 0.01) << '\n';
    si->initialSolve();

    if (!si->isProvenOptimal()) {
        throw std::runtime_error("ShannonProofSimplifier: Failed to solve LP.");
    }

    std::cout << "Simplified to cost " << si->getObjValue() << '\n';

    const double* col_sol = si->getColSolution();
    std::vector<double> row_sol(coin_rowub.size(), 0.0);
    coin_constraints.times(col_sol, row_sol.data());

    for (auto [t, row] : cmi_indices)
    {
        double coeff = coin_rowub[row] - row_sol[row];
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
        custom_constraints.emplace_back();
        const auto& orig_constraint = orig_proof.cmi_constraints[i];
        auto& constraint = custom_constraints.back();
        constraint.is_equality = orig_constraint.is_equality;
        if (orig_proof.custom_constraints[i].get(0) != 0.0)
            constraint.inc(0, orig_proof.custom_constraints[i].get(0));
        for (const auto& [cmi, v] : orig_constraint.entries)
            constraint.inc(get_row_index(cmi) + 1, v);
    }
}

ShannonProofSimplifier::operator SimplifiedShannonProof() const
{
    if (coin_infinity == 0.0)
        return SimplifiedShannonProof();

    SimplifiedShannonProof proof;
    proof.initialized = true;

    proof.variables.resize(coin_rowub.size());
    for (auto [t, i] : cmi_indices)
        proof.variables[i] = ExtendedShannonVar{t, &random_var_names};

    proof.regular_constraints.resize(coin_rowub.size() + coin_colub.size());
    for (auto [t, i] : cmi_indices)
        proof.regular_constraints[i] = NonNegativityOrOtherRule<Rule>(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<0>(), cmi_indices.at(t)));
    for (auto [r, i] : rule_indices)
        proof.regular_constraints[i + coin_rowub.size()] = NonNegativityOrOtherRule<Rule>(
            NonNegativityOrOtherRule<Rule>::Parent(std::in_place_index_t<1>(), r));

    if (orig_proof.dual_solution.get(0) != 0.0)
        proof.dual_solution.inc(0, orig_proof.dual_solution.get(0));
    for (auto [t, v] : cmi_coefficients)
        proof.dual_solution.inc(cmi_indices.at(t) + 1, v);
    for (auto [r, v] : rule_coefficients)
        proof.dual_solution.inc(rule_indices.at(r) + coin_rowub.size() + 1, v);

    for (int i = 0; i < orig_proof.cmi_constraints.size(); ++i)
    {
        int orig_row = orig_proof.regular_constraints.size() + i + 1;
        int row = proof.regular_constraints.size() + i + 1;
        proof.dual_solution.inc(row, orig_proof.dual_solution.get(orig_row));
    }

    proof.custom_constraints = custom_constraints;
    proof.objective = objective;

    return proof;
}

std::ostream& operator<<(std::ostream& out, const ExtendedShannonVar& t)
{
    out << "I(";
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

void ExtendedShannonRule::print(std::ostream& out, const ExtendedShannonVar* vars) const
{
    const std::vector<std::string>* random_var_names = vars[1].random_var_names;
    auto [z, a, b, c] = subsets;

    auto print_cmi = [&] (const CmiTriplet& t) {
        out << ExtendedShannonVar {t, random_var_names};
    };

    switch (type)
    {
    case CMI_DEF_I:
        // I(a;b|c,z) = I(a,c|z) + I(b,c|z) - I(a,b,c|z) - I(c|z)
        print_cmi({a, b, c|z});
        out << " - ";
        print_cmi({a|c, a|c, z});
        out << " - ";
        print_cmi({b|c, b|c, z});
        out << " + ";
        print_cmi({a|b|c, a|b|c, z});
        out << " + ";
        print_cmi({c, c, z});
        break;

    case MI_DEF_I:
        // I(a;b|z) = I(a|z) + I(b|z) - I(a,b|z)
        print_cmi({a, b, z});
        out << " - ";
        print_cmi({a, a, z});
        out << " - ";
        print_cmi({b, b, z});
        out << " + ";
        print_cmi({a|b, a|b, z});
        break;

    case MI_DEF_CI:
        // I(a;b|z) = I(a|z) - I(a|b,z)
        print_cmi({a, b, z});
        out << " - ";
        print_cmi({a, a, z});
        out << " + ";
        print_cmi({a, a, b|z});
        break;

    case CHAIN:
        // I(c|z) + I(a;b|c,z) = I(a,c;b,c|z)
        print_cmi({c, c, z});
        out << " + ";
        print_cmi({a, b, c|z});
        out << " - ";
        print_cmi({a|c, b|c, z});
        break;

    case MUTUAL_CHAIN:
        // I(a;c|z) + I(a;b|c,z) = I(a;b,c|z)
        print_cmi({a, c, z});
        out << " + ";
        print_cmi({a, b, c|z});
        out << " - ";
        print_cmi({a, b|c, z});
        break;

    case MONOTONE:
        // I(a,b|z) >= I(a|c,z)
        print_cmi({a|b, a|b, z});
        out << " - ";
        print_cmi({a, a, c|z});
        break;

    default:
#ifdef __GNUC__
        __builtin_unreachable();
#endif
        return;
    }

    if (is_equality())
        out << " == 0";
    else
        out << " >= 0";
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
    if (out.inquiries.empty()) {
        throw std::runtime_error("undefined information expression");
    }
    return move(out);
}


// TODO: implement optimization as in Xitip: collapse variables that only
// appear together

bool check(const ParserOutput& out)
{
    ShannonTypeProblem prob(out.var_names);
    for (int i = 0; i < out.constraints.size(); ++i)
        prob.add(out.constraints[i], out.cmi_constraints[i]);
    for (int i = 0; i < out.constraints.size(); ++i)
        if (!prob.check(out.inquiries[i]))
            return false;
    return true;
}
