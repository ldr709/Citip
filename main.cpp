// This is the main C++ program file of the ITIP CLI frontend.

#include <iostream>     // cin/cerr etc ...
#include <string>       // getline
#include <vector>       // vector
#include <iterator>     // back_inserter

#include "citip.hpp"
#include "common.hpp"

using util::quoted;
using util::line_iterator;


int main (int argc, char **argv)
try
{
    using namespace std;

    vector<string> expr;

    int depth = 1;
    bool check = false;

    // Parse arguments
    while (true)
    {
        if (argc > 2 && strcmp(argv[1], "-d") == 0)
        {
            depth = std::atoi(argv[2]);
            argv += 2;
            argc -= 2;
            continue;
        }

        if (argc > 1 && strcmp(argv[1], "-c") == 0)
        {
            check = true;
            argv++;
            argc--;
            continue;
        }

        break;
    }

    bool use_stdin = argc == 1;

    if (string(argv[argc-1]) == "-") {
        --argc;
        use_stdin = true;
    }

    copy(argv+1, argv+argc, back_inserter(expr));

    if (use_stdin) {
        copy(line_iterator(cin), line_iterator(), back_inserter(expr));
    }

    ParserOutput out = parse(expr);
    ShannonTypeProblem prob(out);

    ShannonTypeProof proof = prob.prove(std::move(out.target_mat), std::move(out.cmi_target_mat));
    if (proof)
        std::cout << proof << '\n';
    else
    {
        cerr << "The information expression is either:\n"
             << "    1. FALSE, or\n"
             << "    2. a non-Shannon type inequality" << endl;
        return 1;
    }

    if (!check)
    {
        SimplifiedShannonProof simplified_proof = proof.simplify(depth);
        std::cout << simplified_proof << '\n';

        OrderedSimplifiedShannonProof ordered_proof = simplified_proof.order();
        std::cout << ordered_proof << '\n';
    }

    cerr << "The information expression is TRUE." << endl;
    return 0;
}
catch (std::exception& e)
{
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 2;
}
// force stack unwinding
catch (...)
{
    std::cerr << "UNKNOWN ERROR - aborting." << std::endl;
    return 3;
}
