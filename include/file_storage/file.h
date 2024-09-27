#ifndef __DSPACES_FILE_STORAGE_H__
#define __DSPACES_FILE_STORAGE_H__

#include "ss_data.h"
#include "file_storage/policy.h"

typedef enum file_type {
#ifdef DSPACES_HAVE_HDF5
    DS_FILE_HDF5, 
#endif
#ifdef DSPACES_HAVE_NetCDF
    DS_FILE_NetCDF
#endif
    } ds_file_t;

int file_write_od(struct swap_config* swap_conf, struct obj_data *od, ds_file_t ftype);
int file_read_od(struct swap_config* swap_conf, struct obj_data *od, ds_file_t ftype);

#endif // __DSPACES_FILE_STORAGE_H__