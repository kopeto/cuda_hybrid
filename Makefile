SRC += main.cpp
SRC += Graph.cpp
#SRC += Timer.cpp
SRC += all_routes_algo.cu
SRC += build_routes.cpp

CXX = nvcc

all: main

main: $(SRC) Timer.o
	$(CXX) -O3 $^ -o $@

run: main
	./test alzheimer_graph.json

Timer.o:
	g++ -O3 -c Timer.cpp 

clean:
	rm -f test
	rm -f *.o