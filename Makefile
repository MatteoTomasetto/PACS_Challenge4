CXX = mpic++
CXXFLAGS = -std=c++17
CPPFLAGS = -fopenmp -O3 -DNDEBUG -Wall -pedantic -I${mkEigenInc}

.PHONY: all clean distclean

all: main

main: main.o Parallel_Matrix.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.o Parallel_Matrix.o -o main

main.o: main.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c main.cpp

Parallel_Matrix.o: Parallel_Matrix.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c Parallel_Matrix.cpp

clean:
	$(RM) *.o 

distclean: clean
	$(RM) main