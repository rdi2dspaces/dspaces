#include "ss_data.h"


int matrix_copy_cuda_f_double(struct matrix *dst, struct matrix *src);
int matrix_copy_cuda_f_float(struct matrix *dst, struct matrix *src);
int matrix_copy_cuda_f_short(struct matrix *dst, struct matrix *src);
int matrix_copy_cuda_f_char(struct matrix *dst, struct matrix *src);

int ssd_copy_cuda(struct obj_data *to_obj, struct obj_data *from_obj)
{
    struct matrix to_mat, from_mat;
    struct bbox bbcom;
    int ret = dspaces_SUCCESS;

    bbox_intersect(&to_obj->obj_desc.bb, &from_obj->obj_desc.bb, &bbcom);

    matrix_init(&from_mat, from_obj->obj_desc.st, &from_obj->obj_desc.bb,
                &bbcom, from_obj->data, from_obj->obj_desc.size);

    matrix_init(&to_mat, to_obj->obj_desc.st, &to_obj->obj_desc.bb, &bbcom,
                to_obj->data, to_obj->obj_desc.size);
    
    if(to_obj->obj_desc.size == 8) {
        ret = matrix_copy_cuda_f_double(&to_mat, &from_mat);
    } else if(to_obj->obj_desc.size == 4) {
        ret = matrix_copy_cuda_f_float(&to_mat, &from_mat);
    } else if(to_obj->obj_desc.size == 2) {
        ret = matrix_copy_cuda_f_short(&to_mat, &from_mat);
    } else {
        ret = matrix_copy_cuda_f_char(&to_mat, &from_mat);
    }

    return ret;
}

struct obj_data *obj_data_alloc_cuda(obj_descriptor *odsc)
{
    struct obj_data *od = 0;

    od = (struct obj_data *) malloc(sizeof(*od));
    if(!od) {
        fprintf(stderr, "Malloc od error\n");
        return NULL;
    }
    memset(od, 0, sizeof(*od));

    int size = obj_data_size(odsc);
    cudaError_t curet = cudaMalloc((void**)&od->data, size);
    if(curet != cudaSuccess) {
        fprintf(stderr, "cudaMalloc od_data error\n");
        free(od);
        return NULL;
    }
    od->obj_desc = *odsc;

    return od;
}

void obj_data_free_cuda(struct obj_data *od)
{
    if(od) {
        if(od->data) {
            cudaError_t curet =cudaFree(od->data);
            if(curet != cudaSuccess) {
                fprintf(stderr, "cudaFree od_data error\n");
            }
        }
        free(od);
    }
}
