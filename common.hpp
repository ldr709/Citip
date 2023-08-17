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

    struct disjoint_subsets
    {
        const unsigned int disjoint_from;

        disjoint_subsets(unsigned int disjoint_from_, unsigned int universe_) :
            disjoint_from(disjoint_from_ | ~universe_) {}

        class iterator
        {
        private:
            friend struct disjoint_subsets;

            unsigned int m_disjoint_from = -1;
            unsigned int i = 0;
            bool done = false;
            iterator(const disjoint_subsets& x) :
                m_disjoint_from(x.disjoint_from), i(x.disjoint_from), tmp(0) {}

            // Temporary location for dereferenced value in operator->().
            mutable unsigned int tmp = 0;

        public:
            iterator() = default;

            using iterator_category = std::forward_iterator_tag;
            using difference_type = int;
            using value_type = const unsigned int;
            using pointer = value_type*;
            using reference = value_type&;

            value_type operator*() const { return i & ~disjoint_from(); }
            pointer operator->() const
            {
                tmp = **this;
                return &tmp;
            }

            iterator& operator++()
            {
                if (is_last())
                    done = true;
                else
                    i = (i + 1) | disjoint_from();
                return *this;
            }

            iterator operator++(int)
            {
                iterator tmp = *this;
                ++*this;
                return tmp;
            }

            bool is_last()
            {
                return i == (unsigned int) -1;
            }

            unsigned int disjoint_from() const
            {
                return m_disjoint_from;
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
            out.i = (unsigned int) -1;;
            return ++out;
        }
    };

    inline disjoint_subsets all_subsets(unsigned int superset, unsigned int universe)
    {
        return disjoint_subsets(universe & ~superset, universe);
    }

    static_assert(std::forward_iterator<disjoint_subsets::iterator>);

    struct partitions
    {
        const unsigned int n;

        partitions(unsigned int n_) : n(n_) {}

        class iterator
        {
            friend struct partitions;

            unsigned int n = 0;
            std::vector<unsigned int> sets;
            std::vector<disjoint_subsets::iterator> iterators;

            void setup_remaining_sets()
            {
                unsigned int universe = (1 << n) - 1;

                unsigned int disjoint_from = 0;
                if (!iterators.empty())
                    disjoint_from = iterators.back().disjoint_from() | sets.back();

                for (unsigned int i = 0; i < n; ++i)
                {
                    if (disjoint_from & (1 << i))
                        continue;
                    sets.push_back(1 << i);
                    disjoint_from |= 1 << i;
                    iterators.push_back(disjoint_subsets(disjoint_from, universe).begin());
                }
            }

            iterator(const partitions& x, bool at_end = false) : n(x.n)
            {
                if (!at_end)
                    setup_remaining_sets();
            }

        public:
            iterator() = default;

            using iterator_category = std::input_iterator_tag;
            using difference_type = int;
            using value_type = const std::vector<unsigned int>;
            using pointer = value_type*;
            using reference = value_type&;

            reference operator*() const { return sets; }
            pointer operator->() const
            {
                return &sets;
            }

            iterator& operator++()
            {
                while (!iterators.empty() && iterators.back().is_last())
                {
                    sets.pop_back();
                    iterators.pop_back();
                }

                if (!iterators.empty())
                {
                    sets.back() &= ~*iterators.back();
                    sets.back() |= *++iterators.back();

                    setup_remaining_sets();
                }

                return *this;
            }

            iterator operator++(int)
            {
                iterator tmp = *this;
                ++*this;
                return tmp;
            }

            friend bool operator==(const iterator& a, const iterator& b) = default;
            friend bool operator!=(const iterator& a, const iterator& b) = default;
        };

        iterator begin() const
        {
            return iterator(*this);
        }

        iterator end() const
        {
            return iterator(*this, true);
        }
    };

    static_assert(std::input_iterator<partitions::iterator>);

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
