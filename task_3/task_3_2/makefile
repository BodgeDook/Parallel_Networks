CC = g++

CXXFLAGS = -Wall -Wextra -O2 -std=c++17 -pthread

TARGET = server

SRC = server.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) sin_results.txt sqrt_results.txt pow_results.txt

.PHONY: all clean
