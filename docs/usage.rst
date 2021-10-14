How to Use DataSpaces
=====================

DataSpaces consists of two components: a client library and a server library. 
Additionally, the DataSpaces package comes packaged with a standalone, MPI-based server binary.
The typical usage of DataSpaces is to run the server binary along-side the user's application, 
and use the DataSpaces calls provided by the client library to store and access data from the server. 
It is also possible to run the server in a subset of application proccesses, if it is not desired to run
 the server as an independent binary.

DataSpaces provides a full set of bindings for C/C++, and a subset of the API for fortran and python.
This makes it possible to share data between applications written in different programming languages via the common put/get abstraction.

Building a C/C++ program with DataSpaces
----------------------------------------

Flags necessary for compiling a program that uses DataSpaces can be found from the pkg-config file installed by DataSpaces in `<INSTALL_ROOT>/lib/pkgconfig`
