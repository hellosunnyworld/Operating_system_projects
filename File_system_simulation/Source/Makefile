
#########################################################################
#
# Makefile for cuda_test
#
#########################################################################

main.out: main.o file_system.o user_program.o
	nvcc main.o file_system.o user_program.o -o main
main.o: main.cu
	nvcc -c -rdc=true main.cu -o main.o
file_system.o: file_system.cu
	nvcc -c -rdc=true file_system.cu -o file_system.o
user_program.o: user_program.cu
	nvcc -c -rdc=true user_program.cu -o user_program.o

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN) main.o file_system.o user_program.o 
