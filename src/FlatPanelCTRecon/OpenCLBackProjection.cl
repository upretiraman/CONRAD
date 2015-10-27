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
	const unsigned int idx = j*sizeReconPicX + i;
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
		
	//check if inside image boundaries
	if ( i > sizeReconPicX || j > sizeReconPicY)
		return;
		
	//Set Origin from backProjection
	float backProjOriginX = -(sizeReconPicX-1)*pixelSpacingReconX/2;
	float backProjOriginY = -(sizeReconPicY-1)*pixelSpacingReconY/2;
	double x = i*detectorSpacing -((sizeReconPicX * pixelSpacingReconX) / 2);
	double y = -j*detectorSpacing +((sizeReconPicY * pixelSpacingReconY) / 2);
	double detIndexMF = 0.0;
	double s = 0.0;
	double pval = 0.f;
	for (int theta = 0; theta < numberProj; theta++)
	{
		float alpha =  ((M_PI_F/(numberProj))*theta);
		if(alpha == 0 || alpha == (M_PI_F/2.0) || alpha == M_PI_F)
        {
            if(alpha == M_PI_F/2.0)
                s = y;
            else
            {
                s = x;
            }
        }
        else
        {
			double x_det = cos(alpha); //i*pixelSpacingReconX + backProjOriginX;
			double y_det = sin(alpha); //j*pixelSpacingReconY + backProjOriginY;
			double detSlope = tan(alpha);
			double projSlope = tan(alpha + (M_PI_F/2.f));
			double yIntercept = y - x*projSlope;
			double dtX = yIntercept/(detSlope-projSlope);
			double dtY = detSlope*dtX;
			double detDir = sqrt((x_det*x_det)+(y_det*y_det));
			double magn_dPX = sqrt((dtX*dtX)+(dtY*dtY));
			double cosAlpha = 0;
			if(magn_dPX == 0 || detDir == 0) 
			{
				cosAlpha = 0;
				s = 0;
			} 
			else
			{ 
				cosAlpha = (dtX*x_det + dtY*y_det)/(magn_dPX * detDir);
				if(cosAlpha < 0){s= -magn_dPX;}
				else{s= magn_dPX;}
			}			
			//float s = x_det*cos(alpha) + y_det*sin(alpha);		
			//s = (s - originY)/detectorSpacing;
		}
		
		//interpolating
		detIndexMF = (s + ((detectorSpacing*numberDetPixel)+1)/2)/detectorSpacing; //Index in the x direction.
		int low = floor(detIndexMF);
		int high = ceil(detIndexMF);
		double vallow = sinogram[theta*numberDetPixel + low];
		double valhigh = sinogram[theta*numberDetPixel + high];
		double valSin = vallow*(high-detIndexMF) + valhigh*(detIndexMF-low); 		
		pval += valSin;
	}
	
	backProjectionPic[idx] = pval;
	return;
}
