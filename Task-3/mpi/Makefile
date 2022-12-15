SRC_DIR = src/realizations
OBJ_DIR = obj

CC=g++ -g -O3 -std=c++11
MPI=mpicxx -g -O3 -std=c++11
HEADERS_CFLAGS=-c -Wall -I src/headers -std=c++11

SOURCES=$(wildcard $(SRC_DIR)/*.cpp)
OBJECTS=$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
EXECUTABLE=program

simple: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(HEADERS_CFLAGS) $< -o $@

mpi: 
	$(MPI) -c -Wall -I src/headers -std=c++11 src/realizations/computing.cpp -o obj/computing.o
	$(MPI) -g -O3 -std=c++11 -c -Wall -I src/headers src/realizations/main.cpp -o obj/main.o
	$(MPI) -g -O3 -std=c++11 obj/computing.o obj/main.o -o program
	
clean:
	rm $(OBJECTS)
	rm $(EXECUTABLE)
