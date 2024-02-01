#ifndef _DSPACES_GRPC_WRAPPER_
#define _DSPACES_GRPC_WRAPPER_

#include "data_services.h"

typedef void *grpc_server_t;

struct GRPCServer *dspaces_grpc_server_init(const char *addr, dspaces_service_t dsrv);

void dspaces_grpc_server_wait(grpc_server_t self);

void dspaces_grpc_server_del(grpc_server_t self);

typedef void *grpc_client_t;

grpc_client_t dspaces_grpc_client_init(const char *target);

char *dspaces_grpc_client_send_msg(grpc_client_t self, const char *msg);

void dspaces_grpc_client_del(grpc_client_t self);

#endif
