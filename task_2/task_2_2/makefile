CC = gcc
CFLAGS = -O2 -fopenmp
TARGET = integrate_omp

all: $(TARGET)

$(TARGET): integrate_omp.o
	$(CC) -o $(TARGET) integrate_omp.o $(CFLAGS) -lm

integrate_omp.o: integrate_omp.c
	$(CC) $(CFLAGS) -c integrate_omp.c

clean:
	rm -f $(TARGET) *.o