CC = g++

CXXFLAGS = -Wall -Wextra -O2 -std=c++17 -pthread

TARGET = solv

SRC = task_3_solv.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean