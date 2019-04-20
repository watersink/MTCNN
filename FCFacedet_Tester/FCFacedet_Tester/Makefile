
CC = g++
	
lib_obj = bbreg.o detect_face.o FCFacedet.o generateBoundingBox.o nms.o pad.o rerec.o

tester_obj = FCFacedet_Tester.o

LIB_CMP_FLAGS = -Wall -fopenmp -fexceptions -O2 -std=c++11 -m64 -fPIC -I /home/jjj/data/caffe_cpu/caffe/include


EXT_LIBS = /home/jjj/data/caffe_cpu/caffe/.build_release/lib/libcaffe.so /usr/lib/x86_64-linux-gnu/libboost_system.so

OPENCV = /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_imgcodecs.so

TARGET = libFCFacedet FCFacedetTester

all: $(TARGET)


libFCFacedet : $(lib_obj)
	$(CC) $(LIB_CMP_FLAGS) -shared  -lgomp -o libFCFacedet.so $(lib_obj) -m64 $(EXT_LIBS)
	
	
FCFacedetTester : $(tester_obj)
	$(CC) $(LIB_CMP_FLAGS) -o FCFacedetTester $(tester_obj) -m64 libFCFacedet.so $(EXT_LIBS) $(OPENCV)
	
bbreg.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/bbreg.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/bbreg.cpp
	
detect_face.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/detect_face.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/detect_face.cpp
	
FCFacedet.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/FCFacedet.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/FCFacedet.cpp
	
generateBoundingBox.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/generateBoundingBox.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/generateBoundingBox.cpp
	
nms.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/nms.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/nms.cpp
	
pad.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/pad.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/pad.cpp
	
rerec.o : ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/rerec.cpp
	$(CC) $(LIB_CMP_FLAGS) -c ../../FCFacedet_SDK/FCFacedet_SDK/FCFacedet/rerec.cpp
	
FCFacedet_Tester.o : FCFacedet_Tester/FCFacedet_Tester.cpp
	$(CC) $(LIB_CMP_FLAGS) -c FCFacedet_Tester/FCFacedet_Tester.cpp




.PHONY : clean
clean :
	rm libFCFacedet.so $(lib_obj)
	rm FCFacedetTester $(tester_obj)
