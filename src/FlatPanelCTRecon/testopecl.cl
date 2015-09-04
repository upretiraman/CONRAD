kernel void testopecl(
                                                                                __global float* backProjectionPic,
                                                                                __global float* sinogram,
                                                                                int numberProj,
                                                                                float detectorSpacing,
                                                                                int numberDetPixel,
                                                                                int sizeReconPic,
                                                                                float pixelSpacingReconX,
                                                                                float pixelSpacingReconY,
                                                                                float originX,
                                                                                float originY
                                                                                ) 
{
        int x = get_global_id(0);
        int y = get_global_id(1);
        int idx = x*sizeReconPic + y;
        
        float sum = 0.f;

        if ( x >= sizeReconPic || y >= sizeReconPic){
                return;
        }
                
        for (int theta = 0; theta < numberProj; theta++){
                float projAngle = 0.010471975511965976*theta;
            float x_det = cos(projAngle);
                float y_det = sin(projAngle);
            float weight = 1;                  
                                   
            float projSlope = tan(projAngle+(M_PI_F/2));                 
            float detSlope = tan(projAngle);
            float yIntercept = y - x*projSlope;
            float dtX = yIntercept/(detSlope-projSlope);
            float dtY = detSlope*dtX;
            float detDir = sqrt(x_det*x_det + y_det*y_det);
            float magn_dPX = sqrt(dtX*dtX + dtY*dtY);
            float cosAlpha = (dtX*x_det + dtY*y_det)/(magn_dPX*detDir);
            float s = magn_dPX * cosAlpha;   
            
                int y2 = floor(s);
                int y1 = y2 + 1;                
                float valR1 = sinogram[theta*numberDetPixel+y1];
                float valR2 = sinogram[theta*numberDetPixel+y2];
                float value = (y2-s)*valR1 + (s-y1)*valR2;     
                sum += value * weight;
        }

           backProjectionPic[idx] = 0;
        return;
}