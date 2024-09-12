/*
 * Copyright (c) 2020, Rutgers Discovery Informatics Institute, Rutgers
 * University
 *
 * See COPYRIGHT in top-level directory.
 */

#include "mpi.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int test_put_run(int dims, int *npdim, uint64_t *spdim, int timestep,
                        size_t elem_size, int num_vars, int local_mode,
                        int terminate, int nonblock, MPI_Comm gcomm);

void print_usage()
{
    fprintf(
        stderr,
        "Usage: test_writer <dims> np[0] .. np[dims-1] sp[0] ... sp[dims-1] "
        "<timesteps> [-s <elem_size>] [-m (server|local)] [-c <var_count>] "
        "[-t] [-i]\n"
        "   dims              - number of data dimensions. Must be at least "
        "one\n"
        "   np[i]             - the number of processes in the ith dimension. "
        "The product of np[0],...,np[dim-1] must be the number of MPI ranks\n"
        "   sp[i]             - the per-process data size in the ith "
        "dimension\n"
        "   timesteps         - the number of timestep iterations written\n"
        "   -s <elem_size>    - the number of bytes in each element. Defaults "
        "to 8\n"
        "   -m (server|local) - the storage mode (stage to server or stage in "
        "process memory). Defaults to server\n"
        "   -c <var_count>    - the number of variables written in each "
        "iteration. Defaults to one\n"
        "   -t                - send server termination after writing is "
        "complete\n"
        "   -i                - use nonblocking puts (dspaces_iput)\n");
}

int parse_args(int argc, char **argv, int *dims, int *npdim, uint64_t *spdim,
               int *timestep, size_t *elem_size, int *num_vars,
               int *store_local, int *terminate, int *nonblock)
{
    char **argp;
    int i;

    *elem_size = 8;
    *num_vars = 1;
    *store_local = 0;
    *dims = 1;
    *terminate = 0;
    *nonblock = 0;
    if(argc > 1) {
        *dims = atoi(argv[1]);
    }

    if(argc < 3 + (*dims * 2)) {
        fprintf(stderr, "Not enough arguments.\n");
        print_usage();
        return (-1);
    }

    argp = &argv[2];

    for(i = 0; i < *dims; i++) {
        npdim[i] = atoi(*argp);
        argp++;
    }
    for(i = 0; i < *dims; i++) {
        spdim[i] = strtoull(*argp, NULL, 10);
        argp++;
    }

    *timestep = atoi(*argp);
    argp++;

    while(argp < (argv + argc)) {
        if(strcmp(*argp, "-s") == 0) {
            if(argp == ((argv + argc) - 1)) {
                fprintf(stderr, "%s takes an argument.\n", *argp);
                print_usage();
                return (-1);
            }
            *elem_size = atoi(*(argp + 1));
            argp += 2;
        } else if(strcmp(*argp, "-m") == 0) {
            if(argp == ((argv + argc) - 1)) {
                fprintf(stderr, "%s takes an argument.\n", *argp);
                print_usage();
                return (-1);
            }
            if(strcmp(*(argp + 1), "local") == 0) {
                *store_local = 1;
            }
            argp += 2;
        } else if(strcmp(*argp, "-c") == 0) {
            if(argp == ((argv + argc) - 1)) {
                fprintf(stderr, "%s takes an argument.\n", *argp);
                print_usage();
                return (-1);
            }
            *num_vars = atoi(*(argp + 1));
            argp += 2;
        } else if(strcmp(*argp, "-t") == 0) {
            *terminate = 1;
            argp++;
        } else if(strcmp(*argp, "-i") == 0) {
            *nonblock = 1;
            argp++;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", *argp);
            print_usage();
            return (-1);
        }
    }

    if(argp < (argv + argc)) {
        fprintf(stderr, "Warning: ignoring extraneous argument '%s'.\n", *argp);
    }

    return (0);
}

int main(int argc, char **argv)
{
    int nprocs, rank;
    MPI_Comm gcomm;
    int i;

    int npapp;             // number of application processes
    int np[10] = {0};      // number of processes in each dimension
    uint64_t sp[10] = {0}; // block size per process in each dimension
    int timestep;          // number of iterations
    int dims;              // number of dimensions
    size_t elem_size;      // Optional: size of one element in the global array.
                           // Default value is 8 (bytes).
    int num_vars; // Optional: number of variables to be shared in the testing.
                  // Default value is 1.
    int local_mode; // Optional: local storage mode flag
    int terminate;  // Optional: send terminate to server flag
    int nonblock;   // Optional: use nonblocking puts

    if(parse_args(argc, argv, &dims, np, sp, &timestep, &elem_size, &num_vars,
                  &local_mode, &terminate, &nonblock) != 0) {
        return (-1);
    }

    npapp = 1;
    for(i = 0; i < dims; i++) {
        npapp *= np[i];
    }


    // Using SPMD style programming
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    gcomm = MPI_COMM_WORLD;

    int color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &gcomm);

    if(npapp != nprocs) {
        fprintf(stderr,
                "Product of np[i] args must equal number of MPI processes!\n");
        print_usage();
        return (-1);
    }

    // Run as data writer

    test_put_run(dims, np, sp, timestep, elem_size, num_vars, local_mode,
                 terminate, nonblock, gcomm);

    MPI_Barrier(gcomm);
    MPI_Finalize();

    if(rank == 0) {
        fprintf(stderr, "That's all from test_writer, folks!\n");
    }

    return 0;
}
