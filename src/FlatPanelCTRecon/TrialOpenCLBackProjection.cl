#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable

__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// OpenCL kernel for backprojecting a sinogram
kernel void TrialOpenCLBackProjection(
										__global float* backProjectionPic,
										__global float* sinogram,
										int numberProj,
										float detectorSpacing,
										int numberDetPixel,
										int sizeReconPic,
										float pixelSpacingReconX,
										float pixelSpacingReconY,
										float originX,
										float originY,
										double sinoSpacing) 
{
	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// y index
	const unsigned int idx = x*sizeReconPic + y;
	
	double s;
	float sum = 0.f;
	//check if inside image boundaries
	if ( x > sizeReconPic || y > sizeReconPic)
		return;
		
	float pval = 0.f;
	for (int theta = 0; theta < numberProj; theta++)
		{
			//double projAngle = 0.010471975511965976*theta; 
			double projAngle = sinoSpacing*theta;
	        double x_det = cos(projAngle);
	        double y_det = sin(projAngle);
	        double projSlope,detSlope,dtX,dtY;
	        double yIntercept;
	        float weight = 1;                  
	        if(projAngle == 0 || projAngle == 90 || projAngle == 180)
	        {
	            if(projAngle == 90)
	                s = y;
	            else
	            {
	                s = x;
	                weight = (float) 0.5;                      
	            }
	        }
	        else
	        {
	            //Line intersections for detecting the corresponding detector pixel for the current image pixel.                   
	            projSlope = tan(projAngle+(M_PI_F/2));    		  //Projecting orthogonal line w.r.t. the detector.                  
	            detSlope = tan(projAngle);                 		  //Detector slope
	            yIntercept = y - x*projSlope;                 	  //y-axis intersection of the projected line.
	            dtX = yIntercept/(detSlope-projSlope);        	  //X intersection
	            dtY = detSlope*dtX;                            	  //Y intersection
	            double detDir = sqrt(x_det*x_det + y_det*y_det);  //Detectors direction.
	            double magn_dPX = sqrt(dtX*dtX + dtY*dtY);   	  //Pixel vector.
	            double cosAlpha = (dtX*x_det + dtY*y_det)/(magn_dPX*detDir);    //Angle between these two vectors. (0° or 180°)
	           	//s = magn_dPX * Math.signum(cosAlpha);
	            s = magn_dPX * cosAlpha;
	        }
	         
	         //interpolate
			int y2 = floor(s); // lower s
			int y1 = y2 + 1;   // higher s			
			float valR1 = sinogram[theta*numberProj+y1];
			float valR2 = sinogram[theta*numberProj+y2];
			float value = (y2-s)*valR1 + (s-y1)*valR2;
	         
	        sum += value * weight;                              //Interpolated value from the sinogram and weighted for redundancy.
	    }
    	backProjectionPic[idx] = sum;
    }
}