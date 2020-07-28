#SRC += main.cpp
SRC += Graph.cpp
#SRC += Timer.cpp
SRC += all_routes_algo.cu
SRC += build_routes.cpp

CUDAXX = nvcc
CXX = g++

all: main

main: main.cpp Timer.o Graph.o build_routes.o all_routes_algo.o
	$(CUDAXX) -O3 $^ -o $@

bench1: bench1.cpp Timer.o Graph.o build_routes.o all_routes_algo.o
	$(CUDAXX) -O3 $^ -o $@

bench2: bench2.cpp Timer.o Graph.o build_routes.o all_routes_algo.o
	$(CUDAXX) -O3 $^ -o $@

bench3: bench3.cpp Timer.o Graph.o build_routes.o all_routes_algo.o
	$(CUDAXX) -O3 $^ -o $@

run: main
	./test alzheimer_graph.json

Timer.o:
	$(CXX) -O3 -c Timer.cpp 

Graph.o build_routes.o all_routes_algo.o: $(SRC)
	$(CUDAXX) -O3 -c $(SRC)

clean:
	rm -f main bench1
	rm -f *.o