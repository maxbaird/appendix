PROJECT   = vector_add.out
CC        = nvcc
CFLAGS    = -c -I${MPI_HOME}/include/ -Xcompiler -Wall,-Wextra,-Werror
LDFLAGS   = -L${MPI_HOME}/lib -lmpi

.PHONY : all clean

all : $(PROJECT)

$(PROJECT) : vector_add.o
	$(CC) $(LDFLAGS) -o $@ $<

vector_add.o : vector_add.cu
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o *.out $(PROJECT)
