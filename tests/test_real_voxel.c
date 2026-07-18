#include "bf_frame.h"
#include "bf_voxel.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv){if(argc!=2)return 2;char error[256]={0};bf_frame_file *file=NULL;if(!bf_frame_open(argv[1],&file,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 3;}const bf_frame_input *frame=bf_frame_input_view(file);bf_voxel_config config={{-54,-54,-5},{54,54,3},{.075f,.075f,.2f},5,10,160000};size_t wb=bf_voxelize_workspace_bytes(&config);float *voxels=malloc(160000ull*10*5*4);bf_coord4 *coords=malloc(160000*sizeof(*coords));int64_t *counts=malloc(160000*sizeof(*counts));void *workspace=malloc(wb);size_t count=0;bf_voxel_stats stats={0};int ok=voxels&&coords&&counts&&workspace&&bf_voxelize_f32_workspace_ref(frame->points,frame->point_count,5,0,&config,voxels,coords,counts,&count,&stats,workspace,wb);printf("real_voxel count=%zu accepted=%zu nonfinite=%zu range=%zu dropped_voxels=%zu dropped_points=%zu\n",count,stats.accepted_points,stats.rejected_nonfinite,stats.rejected_out_of_range,stats.dropped_voxel_capacity,stats.dropped_point_capacity);free(workspace);free(counts);free(coords);free(voxels);bf_frame_close(file);return ok?0:4;}
