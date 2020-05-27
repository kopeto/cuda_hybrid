SRC += test.cpp
SRC += Graph.cpp
SRC += Timer.cpp
SRC += all_routes_algo.cu
SRC += build_routes.cpp

CXX = nvcc

all: test

test: $(SRC)
	$(CXX) -O3 $^ -o $@

run: test
	./test alzheimer_graph.json

clean:
	rm -f test