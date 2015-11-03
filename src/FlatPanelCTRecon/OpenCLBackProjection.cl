#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


// OpenCL kernel for backprojecting a sinogram
kernel void OpenCLBackProjection(__global float* backProjectionPic, //OpenCL reconstructed phantom
								 __global float* sinogram, //OpenCL sinogram
								int numberProj, //Number of projections
								double detectorSpacing, //Spacing between detector pixels
								int numberDetPixel, //Number of detector pixels
								int sizeReconPicX, //Size of the reconstructed phantom in x direction
								int sizeReconPicY, //Size of the reconstructed phantom in y direction
								double pixelSpacingReconX, //Pixel spacing in the reconstructed phantom in
								x direction
								double pixelSpacingReconY, //Pixel spacing in the reconstructed phantom in
								y direction
								float originX, //Origin for the detector/CT coordinate system in X direction
								float originY) //Origin for the detector/CT coordinate system in y direction
{
	const unsigned int i = get_global_id(0); //Horizontal index for the reconstructed phantom
	const unsigned int j = get_global_id(1); //Vertical index for the reconstructed phantom
	const unsigned int idx = j*sizeReconPicX + i; //Linear index for the phantom
	const double detectorLength = detectorSpacing*numberDetPixel;
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	//check if inside image boundaries
	if ( i > sizeReconPicX || j > sizeReconPicY) //Validating the phantom sizes
		return;

	//Transforming coordinates from the computer space to the detector/CT ('real world') coordinate system.
	double x = i*detectorSpacing -((sizeReconPicX * pixelSpacingReconX) / 2.0);
	double y = j*detectorSpacing -((sizeReconPicY * pixelSpacingReconY) / 2.0);
	double detIndexMF = 0.0; //Index for the detector in the sinogram for the considered projection angle.
	double pval = 0.f; //Variable to sum all the projections for one considered pixel

	for (int a = 0; a < numberProj; a++) //Loop over the number of projections
	{
		float theta =  ((M_PI_F/(numberProj))*a); //Computing the current angle of the considered projection.
		float s; //Distance from the intersection of the central ray with the detector and the considered ray

		//Validating the angle to simplify computation and to avoid indetermined
		// results in trigonometric functions. (Tan)
		if(theta == 0 || theta == (M_PI_F/2.0) || theta == M_PI_F)
        {
            if(theta == M_PI_F/2.0)
                s = y;
            else
            {
                s = x;
            }
         }
        else
        {
	        //Components for the unit vector normal to the central ray. (origin: 0,0)
			double x_det = cos(theta);
			double y_det = sin(theta);
			double detSlope = tan(theta); //Slope of the detector line.
			double projSlope = tan(theta + (M_PI_F/2.f)); //Slope of the central ray. (orthogonal to the detector.
			
			//Finding the interception coordinates between the ray passing through the
			//considered phantom pixel and the detector.
			double yIntercept = y - x*projSlope;
			double dtX = yIntercept/(detSlope-projSlope);
			double dtY = detSlope*dtX;
			
			//Getting the magnitude for the detected pixel (real world position)
			double magn_dPX = sqrt((dtX*dtX)+(dtY*dtY));
			//Auxiliar angle indicating on which side of the detector the intersection is. (left or right from the central ray)
			double cosAlpha = 0;
			if(magn_dPX == 0)
			{
				//Central ray
				cosAlpha = 0;
				s = 0;
			}
			else
			{
				cosAlpha = (dtX*x_det + dtY*y_det)/(magn_dPX);
			}
			
			if(cosAlpha < 0){s= -magn_dPX;} //Left side.
			else{s= magn_dPX;} //Right side.
			
		}
		detIndexMF = (s + (detectorLength/2))/detectorSpacing; //Transforming the
		distance from the central ray to a computer space index. (index for the sinogram line)
		double valSin = 0;
		valSin = sinogram[a*numberDetPixel + (int)detIndexMF]; //Getting the intensity value from the sinogram
		pval += valSin; //Summing up all projections in the considered pixel.
	}
	backProjectionPic[idx] = pval; //Assigning the summed value to the considered pixel of the reconstructed phantom.
	return;
}
