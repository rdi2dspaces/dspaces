#ifndef _DSPACES_GRPC_WRAPPER_
#define _DSPACES_GRPC_WRAPPER_

typedef void *grpc_server_t;

grpc_server_t dspaces_grpc_server_init(const char *addr);

void dspaces_grpc_server_wait(grpc_server_t self);

void dspaces_grpc_server_del(grpc_server_t self);

typedef void *grpc_client_t;

grpc_client_t dspaces_grpc_client_init(const char *target);

char *dspaces_grpc_client_send_msg(grpc_client_t self, const char *msg);

void dspaces_grpc_client_del(grpc_client_t self);

#endif
