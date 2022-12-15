
#########################################################################
#
# Makefile for cuda_test
#
#########################################################################

main.out: main.o virtual_memory.o user_program.o
	nvcc main.o virtual_memory.o user_program.o -o main
main.o: main.cu
	nvcc -c -rdc=true main.cu -o main.o
virtual_memory.o: virtual_memory.cu
	nvcc -c -rdc=true virtual_memory.cu -o virtual_memory.o
user_program.o: user_program.cu
	nvcc -c -rdc=true user_program.cu -o user_program.o

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN) main.o virtual_memory.o user_program.o 
