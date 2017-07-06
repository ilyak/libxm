CC= cc
CFLAGS= -Wall -Wextra -g -fopenmp
LDFLAGS= -L/usr/local/lib
LIBS= -lblas -lpthread -lm

# Intel Compiler with MKL on Linux (release build)
#CC= icc
#CFLAGS= -DNDEBUG -Wall -Wextra -O3 -fopenmp -mkl=sequential
#LDFLAGS=
#LIBS= -lpthread -lm

EXAMPLE= example
EXAMPLE_O= example.o
TEST= test
TEST_O= test.o

XM_A= xm.a
XM_O= alloc.o blockspace.o contract.o dim.o scalar.o tensor.o util.o xm.o

AR= ar rc
RANLIB= ranlib

all: $(EXAMPLE) $(TEST)

$(EXAMPLE): $(XM_A) $(EXAMPLE_O)
	$(CC) -o $@ $(CFLAGS) $(EXAMPLE_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(TEST): $(XM_A) $(TEST_O)
	$(CC) -o $@ $(CFLAGS) $(TEST_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(XM_A): $(XM_O)
	$(AR) $@ $(XM_O)
	$(RANLIB) $@

check: $(TEST)
	./$(TEST)

checkmpi: $(TEST)
	mpirun -np 1 ./$(TEST)
	mpirun -np 2 ./$(TEST)
	mpirun -np 3 ./$(TEST)

dist:
	git archive --format=tar.gz --prefix=libxm/ -o libxm.tgz HEAD

clean:
	rm -f $(XM_A) $(XM_O) $(EXAMPLE) $(EXAMPLE_O) $(TEST) $(TEST_O)
	rm -f *.core xmpagefile libxm.tgz

.PHONY: all check checkmpi clean dist
