#ifndef __COMMON_HPP__INCLUDED__
#define __COMMON_HPP__INCLUDED__

# include <iostream>
# include <iterator>    // istream_iterator
# include <sstream>     // ostringstream
# include <string>
# include <iterator>


namespace util
{

    // Compose a string message from multiple values

    inline void print_all(std::ostream& out)
    {
        out << std::flush;
    }

    template <class H, class ...T>
    void print_all(std::ostream& out, H head, T... t)
    {
        out << head;
        print_all(out, t...);
    }

    template <class ...T>
    std::string sprint_all(T... t)
    {
        std::ostringstream out;
        print_all(out, t...);
        return out.str();
    }


    // iterate over lines in a stream

    namespace detail
    {
        class Line : public std::string
        {
            friend std::istream& operator>>(std::istream& in, Line& line)
            {
                return std::getline(in, line);
            }
        };

    }

    typedef std::istream_iterator<detail::Line> line_iterator;


    //

    inline std::string quoted(std::string str)
    {
        return '"' + str + '"';
    }

    class disjoint_subsets
    {
        unsigned int disjoint_from;
        unsigned int universe;

    public:
        disjoint_subsets(unsigned int disjoint_from_, unsigned int universe_) :
            disjoint_from(disjoint_from_), universe(universe_) {}

        class iterator
        {
            friend class disjoint_subsets;

            unsigned int disjoint_from = -1;
            unsigned int i = 0;
            iterator(const disjoint_subsets& x) :
                disjoint_from(x.disjoint_from), i(x.disjoint_from), tmp(0) {}

            // Temporary location for dereferenced value in operator->().
            mutable unsigned int tmp = 0;

        public:
            iterator() = default;

            using iterator_category = std::forward_iterator_tag;
            using difference_type = int;
            using value_type = const unsigned int;
            using pointer = value_type*;
            using reference = value_type&;

            value_type operator*() const { return i & ~disjoint_from; }
            pointer operator->() const
            {
                tmp = **this;
                return &tmp;
            }

            iterator& operator++()
            {
                i = (i + 1) | disjoint_from;
                return *this;
            }

            iterator operator++(int)
            {
                iterator tmp = *this;
                ++*this;
                return tmp;
            }

            friend auto operator<=>(const iterator& a, const iterator& b) = default;
        };

        iterator begin() const
        {
            return iterator(*this);
        }

        iterator end() const
        {
            iterator out(*this);
            out.i = universe;
            return ++out;
        }
    };

    static_assert(std::forward_iterator<disjoint_subsets::iterator>);

    template<typename Range>
    class skip_n_range
    {
        Range inner;
        typedef decltype(inner.begin()) iterator;
        typedef typename iterator::difference_type index_type;
        index_type skip;

    public:
        skip_n_range(Range&& inner_, index_type skip_) : inner(inner_), skip(skip_) {}
        skip_n_range(const Range& inner_, index_type skip_) : inner(inner_), skip(skip_) {}

        iterator begin() const
        {
            iterator b = inner.begin();
            std::ranges::advance(b, skip, end());
            return b;
        }

        iterator end() const
        {
            return inner.end();
        }
    };

    template<typename R>
    skip_n_range<R> skip_n(R&& range, typename decltype(range.begin())::difference_type skip)
    {
        return skip_n_range<R>(std::forward<R>(range), skip);
    }
}


#endif // include guard
