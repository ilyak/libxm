# Scalar type is double
SCALAR_TYPE= XM_SCALAR_DOUBLE
# Scalar type is float
#SCALAR_TYPE= XM_SCALAR_FLOAT
# Scalar type is double complex
#SCALAR_TYPE= XM_SCALAR_DOUBLE_COMPLEX
# Scalar type is float complex
#SCALAR_TYPE= XM_SCALAR_FLOAT_COMPLEX

# cc with Netlib BLAS
CC= cc
CFLAGS= -D$(SCALAR_TYPE) -Wall -Wextra -g
LDFLAGS= -L/usr/local/lib
LIBS= -lblas -lpthread -lm

# Intel Compiler with MKL on Linux (release build)
#CC= icc
#CFLAGS= -D$(SCALAR_TYPE) -DNDEBUG -Wall -Wextra -O3 -fopenmp -mkl=sequential
#LDFLAGS=
#LIBS= -lpthread -lm

# Clang with Netlib BLAS on OpenBSD (debug build)
#CC= clang
#CFLAGS= -D$(SCALAR_TYPE) -Weverything -Wno-padded -Wno-used-but-marked-unused -Wno-missing-noreturn -Wno-format-nonliteral -fcolor-diagnostics -g -DHAVE_ARC4RANDOM -DHAVE_BITSTRING_H -DHAVE_TREE_H
#LDFLAGS= -L/usr/local/lib
#LIBS= -lblas -lg2c -lpthread -lm

EXAMPLE= example
EXAMPLE_O= example.o
TEST= test
TEST_O= test.o

XM_A= xm.a
XM_O= alloc.o blockspace.o contract.o dim.o tensor.o

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

dist:
	git archive --format=tar.gz --prefix=libxm/ -o libxm.tgz HEAD

clean:
	rm -f $(XM_A) $(XM_O) $(EXAMPLE) $(EXAMPLE_O) $(TEST) $(TEST_O)
	rm -f *.core xmpagefile libxm.tgz

.PHONY: all check clean dist
