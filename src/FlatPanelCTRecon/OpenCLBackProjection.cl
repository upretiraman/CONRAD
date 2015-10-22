#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


float interpolate(float* sinogram, float theta, float s, int numberProj)
{
	int sinogramIndex = ((int)(theta*numberProj+s));	
	float y = s;
	int y2 = floor(s); // lower s
	int y1 = y2 + 1; // higher s

	return 0;
}

// OpenCL kernel for backprojecting a sinogram
kernel void OpenCLBackProjection(
	__global float* backProjectionPic,
	__global float* sinogram,
	int numberProj,
	double detectorSpacing,
	int numberDetPixel,
	int sizeReconPicX,
	int sizeReconPicY,
	double pixelSpacingReconX,
	double pixelSpacingReconY,
	float originX,
	float originY) 
{
	const unsigned int i = get_global_id(0);// x index
	const unsigned int j = get_global_id(1);// y index
	const unsigned int idx = i*sizeReconPicY + j;
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	//check if inside image boundaries
	if ( i > sizeReconPicX || j > sizeReconPicY)
		return;
		
	//Set Origin from backProjection
	float backProjOriginX = -(sizeReconPicX-1)*pixelSpacingReconX/2;
	float backProjOriginY = -(sizeReconPicY-1)*pixelSpacingReconY/2;

	float pval = 0.f;
	for (int theta = 0; theta < numberProj; theta++)
	{
		float alpha =  ((2.f*M_PI_F/(numberProj))*theta);
		float X = i*pixelSpacingReconX + backProjOriginX;
		float Y = j*pixelSpacingReconY + backProjOriginY;
		float s = X*cos(alpha) + Y*sin(alpha);
		s = (s - originY)/detectorSpacing;
		
		//interpolate
		float y = s;
		int y2 = floor(s); // lower s
		int y1 = y2 + 1;   // higher s			
		float valR1 = sinogram[theta*numberProj+y1];
		float valR2 = sinogram[theta*numberProj+y2];
		float value = ((y2-y)/(y2-y1))*valR1 + ((y-y1)/(y2-y1))*valR2;
		//float aux = M_PI_F*value/numberProj;
		//value = M_PI_F*value/numberProj;

		pval += value;
	}
	
	backProjectionPic[idx] = pval;
	return;
}
