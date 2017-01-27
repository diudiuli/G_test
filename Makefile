CUDA_FLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart
#LINK_FILES=`pkg-config opencv --cflags --libs`
LIBS+=$(shell pkg-config --libs opencv)
CFLAGS+=$(shell pkg-config --cflags opencv)
CXX_FLAGS=-std=c++11 -std=c++0x -03 -I/opt/local/include/opencv -I/opt/local/include

all: cuda_conv

cuda_conv: cuda_template_match.o
	g++ -o cuda_conv cuda_template_match.o $(LIBS) $(CFLAGS) $(CUDA_FLAGS)

cuda_template_match.o:cuda_template_match.cu
	nvcc -c cuda_template_match.cu 

clean:
	rm *.o cuda_conv
