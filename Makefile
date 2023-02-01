BUILDDIR = build
OBJS     = $(addprefix $(BUILDDIR)/,main.o parser.o scanner.o citip.o)
CPPFLAGS = -MMD -MP
CXXFLAGS = -std=c++20 -O2 -ggdb -I. -I$(BUILDDIR)

# CBC
ILP_SOLVER_LIBS = -lCbcSolver -lCbc -lpthread -lrt /usr/lib/libnauty.a -lcoinasl -lCgl

LIBS = $(ILP_SOLVER_LIBS) -lOsiClp -lClpSolver -lClp -lcholmod -lamd -lcoinasl -lOsi -lCoinUtils -lbz2 -lz -lglpk -llapack -lblas -lm

all: prepare Citip

Citip: $(OBJS)
	g++ -o $@ $^ $(LIBS)

$(BUILDDIR)/%.o: %.cpp Makefile
	$(CXX) -o $@ -c $< $(CPPFLAGS) $(CXXFLAGS)

$(BUILDDIR)/%.o: $(BUILDDIR)/%.cxx Makefile
	$(CXX) -o $@ -c $< $(CPPFLAGS) $(CXXFLAGS)

$(BUILDDIR)/parser.cxx: parser.y Makefile
	bison -o $@ --defines=$(BUILDDIR)/parser.hxx $<

$(BUILDDIR)/scanner.cxx: scanner.l Makefile
	flex  -o $@ --header-file=$(BUILDDIR)/scanner.hxx $<

$(OBJS): $(BUILDDIR)/scanner.cxx $(BUILDDIR)/parser.cxx

.PHONY: prepare all
prepare:
	@mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

clobber: clean
	rm -f Citip

-include $(OBJS:%.o=%.d)
