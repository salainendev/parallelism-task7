INFO = -Minfo=all
LIBS = -cudalib=cublas -lboost_program_options
GPU = -acc=gpu
CXX = pgc++
all:main

main:
	$(CXX) $(GPU) $(LIBS) $(INFO) -o $@ main.cpp 

clean:all
	rm main