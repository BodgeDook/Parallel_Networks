CXX = g++
CXXFLAGS = -O2

ARRAY_TYPE ?= float

ifneq ($(MAKECMDGOALS), clean)
    ifeq ($(ARRAY_TYPE), float)
        CXXFLAGS += -DARRAY_TYPE=float
    else ifeq ($(ARRAY_TYPE), double)
        CXXFLAGS += -DUSE_DOUBLE
    else
        $(error ARRAY_TYPE must be either 'float' or 'double')
    endif
endif

all: main.cpp
	$(CXX) $(CXXFLAGS) -o sin_sum main.cpp

clean:
	rm -f sin_sum