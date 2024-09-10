#include "stdint.h"


#include "ss_data.h"

struct swap_config
{
    char *file_dir;
    int mem_quota_type;
    char *policy;
    float disk_quota_MB;
    union {
        float MB;
        float percent;
    } mem_quota;
};

// struct obj_data_ptr_flat_list_entry {
//     struct list_head entry;
//     struct obj_data *od;
//     int usecnt;
// };

// struct obj_data_ptr_flat_list_entry* ls_flat_od_list_entry_alloc(struct obj_data* od);
void free_ls_od_list(struct list_head* ls_od_list);

void memory_quota_parser(char* str, struct swap_config* swap);
void disk_quota_parser(char* str, struct swap_config* swap);

int policy_str_check(const char*str);

int need_swap_out(struct swap_config *swap, uint64_t size_MB);

struct obj_data* which_swap_out(struct swap_config* swap, struct list_head* ls_od_list);