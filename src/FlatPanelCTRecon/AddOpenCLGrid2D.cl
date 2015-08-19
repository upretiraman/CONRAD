#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable


__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// OpenCL kernel for adding two OpenCLGrid2D to each other

kernel void AddOpenCLGrid2D(
	__global float* resultGrid,
	__global float* grid1,
	__global float* grid2,
	int gridSizeX,
	int gridSizeY) 
{
	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// x index
	const unsigned int idx = y*gridSizeX + x;
	
	//check if inside image boundaries
	if ( x > gridSizeX*gridSizeY)
		return;

	float grid1val = grid1[idx];
	float grid2val = grid2[idx];
	resultGrid[idx] = grid1val+grid2val;
	
	return;
}