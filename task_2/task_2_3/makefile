CXX = g++
CXXFLAGS = -O2 -fopenmp -Wall

TARGET1 = without_parals
TARGET2 = v1_with_parals
TARGET3 = v2_with_parals
TARGET4 = v3_schedule_search

SRC1 = without_parals.cpp
SRC2 = v1_with_parals.cpp
SRC3 = v2_with_parals.cpp
SRC4 = v3_schedule_search.cpp

all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)

$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SRC1)

$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SRC2)

$(TARGET3): $(SRC3)
	$(CXX) $(CXXFLAGS) -o $(TARGET3) $(SRC3)

$(TARGET4): $(SRC4)
	$(CXX) $(CXXFLAGS) -o $(TARGET4) $(SRC4)

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4)