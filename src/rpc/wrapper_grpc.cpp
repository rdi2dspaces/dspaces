#include<string>
#include <grpcpp/grpcpp.h>

#include "rpc/dspaces.grpc.pb.h"
#include "data_services.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::Channel;
using grpc::ClientContext;
using dspaces::Greeter;
using dspaces::HelloReply;
using dspaces::HelloRequest;
using dspaces::SharedSpaceInfoRequest;
using dspaces::SharedSpaceInfoReply;

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public Greeter::Service {
    public:
    GreeterServiceImpl(dspaces_service_t dsrv) :
        dsrv_(dsrv) {
    };

    private:
    dspaces_service_t dsrv_;

    Status SayHello(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override {
        std::string prefix("Hello ");
        std::cout << "received " << request->name() << std::endl;
        reply->set_message(prefix + request->name());
        return Status::OK;
    }

    Status SharedSpaceQuery(ServerContext *context, const SharedSpaceInfoRequest *request, SharedSpaceInfoReply *reply) override {
        void *obj_hdr;
        unsigned int len;
        char *check_str;
        get_sspace_info(dsrv_, &obj_hdr, &len, &check_str);
        reply->set_header(std::string((char *)obj_hdr, len));
        reply->set_check(std::string(check_str));
        free(obj_hdr);
        free(check_str);
        return Status::OK;    
    }
};

typedef struct GRPCServer {
    GRPCServer(dspaces_service_t dsrv):
        service(dsrv) {};
    std::unique_ptr<Server> server;
    GreeterServiceImpl service;
} *grpc_server_t;

extern "C" struct GRPCServer *dspaces_grpc_server_init(const char *addr, dspaces_service_t dsrv)
{
    struct GRPCServer *self = new struct GRPCServer(dsrv);
    std::string server_address(addr);
    ServerBuilder builder;

    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&self->service);

    self->server = builder.BuildAndStart();
    return(self);
}

extern "C" void dspaces_grpc_server_wait(grpc_server_t self)
{
    (self->server)->Wait();
}

extern "C" void dspaces_grpc_server_del(grpc_server_t self)
{
    delete self;
}

class GreeterClient {
 public:
  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string SayHello(const std::string& user) {
    // Data we are sending to the server.
    HelloRequest request;
    request.set_name(user);

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->SayHello(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

typedef void *grpc_client_t;

extern "C" grpc_client_t dspaces_grpc_client_init(const char *target)
{
    std::string target_str(target);
    return new GreeterClient(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
}

extern "C" char *dspaces_grpc_client_send_msg(grpc_client_t self, const char *msg)
{
    GreeterClient *greeter = (GreeterClient *)self;
    std::string user(msg);
    std::string reply = greeter->SayHello(user);
    char *c_reply = strdup(reply.c_str());
    return(c_reply);
}

extern "C" void dspaces_grpc_client_del(grpc_client_t self) 
{
    GreeterClient *greeter = (GreeterClient *)self;
    delete greeter;
}
