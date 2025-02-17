CC = gcc
CFLAGS = -O2 -fopenmp
TARGET = matrix_vector

all: $(TARGET)

$(TARGET): matrix_vector.o
	$(CC) $(CFLAGS) -o $(TARGET) matrix_vector.o

matrix_vector.o: matrix_vector.c
	$(CC) $(CFLAGS) -c matrix_vector.c

clean:
	rm -f $(TARGET) *.o