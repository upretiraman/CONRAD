package FlatPanelCTRecon;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;

public class myFanProjRD {
	int projectionNumber;
    double deltaS;
    int detectorPixel;
    double D;
    double dD;
    double fanAngle;
    double detectorLength;
    
    public myFanProjRD(int projectionNumber, double detectorSpacing, int detectorPixel, double sourceIsoDistance, double isoDetectorDistance)
    {
    	 this.projectionNumber = projectionNumber;   //Number of projections
         this.deltaS = detectorSpacing;              //Delta s
         this.detectorPixel = detectorPixel;         //Number of pixels in the detector.
         this.D = sourceIsoDistance;                 //Distance from the origin to the x-ray source
         this.dD = isoDetectorDistance;              //Distance from the center of the detector to the x-ray source.
         this.detectorLength = detectorPixel*detectorSpacing;
         this.fanAngle = 2.0 * Math.atan2(detectorLength/2, dD);
    }
    
    public Grid2D fanProjectRD(Grid2D phant, boolean mode, float samplingFreq)		//mode is 0 or 1. 1 for halfscan.
    {
    	Grid2D fan_o_gram = new Grid2D(detectorPixel,projectionNumber);    
        Translation t = new Translation(-(phant.getSize()[0] * phant.getSpacing()[0])/2, -(phant.getSize()[1] * phant.getSpacing()[1])/2,-1);
        Box b = new Box((phant.getSize()[0] * phant.getSpacing()[0]), (phant.getSize()[1] * phant.getSpacing()[1]), 2);
        b.applyTransform(t);
        double delta_T;
        if (mode){
            delta_T = (Math.PI+this.fanAngle)/this.projectionNumber;
        }
        else{
            delta_T = (2*Math.PI)/this.projectionNumber;
        }
        fan_o_gram.setSpacing(1,delta_T);
        double beta;   //Step angle for projection
        for(int i = 0; i <= this.projectionNumber; ++i)
        {          
            beta = delta_T*i;
            //Defining all useful geometric parameters and references for angles different from 0,90 and 180°.
            double sB = Math.sin(beta),cB = Math.cos(beta);
            double sB90 = Math.sin(beta+(Math.PI/2.0)), cB90 = Math.cos(beta+(Math.PI/2.0));
            double[] sourceVt = {D*sB, D*cB};           // Vector pointing to the source point.
            PointND src = new PointND(sourceVt[0],sourceVt[1],.0d);
            double[] detVt = {dD*(-sB), dD*(-cB)};      // Detector middle point; opposite to the detector source.     
            for(int a = 0; a <= this.detectorPixel;++a){
                double s = this.deltaS * a - this.detectorLength/2; // Distance from the center of the detector to the current pixel.                
                double sum = 0.0d;
                double[] currentDP = {s*sB90,s*cB90};              
                currentDP[0] += detVt[0];               // Vector pointing to the current detector pixel. (x coordinate)
                currentDP[1] += detVt[1];               // Vector pointing to the current detector pixel. (y coordinate)
                PointND cdp = new PointND(currentDP[0],currentDP[1],.0d);
                StraightLine li = new StraightLine(cdp,src);  		//Line from the detector pixel to the source point
                ArrayList<PointND> cutSites = b.intersect(li);		//Array of points for the intersection of the box.
                if(cutSites.size()==0){ 							//In case there is no intersection, continue with the loop.
                    continue;
                }
                double[] init = {cutSites.get(0).get(0),cutSites.get(0).get(1)};
                double[] fin = {cutSites.get(1).get(0),cutSites.get(1).get(1)};        
                double[] step = {fin[0]-init[0],fin[1]-init[1]};
                double dist = Math.hypot(step[0], step[1]);
                if (dist == 0){
                    continue;}
                step[0] /= (dist*samplingFreq);
                step[1] /= (dist*samplingFreq); 
                init[0] += (phant.getSize()[0] * phant.getSpacing()[0])/2;
                init[1] += (phant.getSize()[1] * phant.getSpacing()[1])/2;          
                for(float k = .0f;k<=dist*samplingFreq;k++){
                    double x = init[0] + step[0]*k;
                    double y = init[1] + step[1]*k;
                    if(x < 0 || y < 0 || x >= phant.getSize()[0]-1 || y >= phant.getSize()[1]-1){
                        continue;}
                    sum += InterpolationOperators.interpolateLinear(phant, x, y);
                }
                fan_o_gram.setAtIndex(a, i, (float)sum);               
            }//For loop for every pixel in the detector.
        }//For loop for every projection.
        return fan_o_gram;
    }
    
    Grid2D rebinFanogram(Grid2D fanogram, boolean mode){
    	Grid2D sino = new Grid2D(fanogram.getSize()[0],fanogram.getSize()[1]);
    	sino.setSpacing(fanogram.getSpacing()[0],fanogram.getSpacing()[1]);
    	double delta_B,delta_T;    	
    	if (mode){
    		delta_B = (Math.PI+this.fanAngle)/this.projectionNumber;
    		delta_T = Math.PI/this.projectionNumber;
    		}
        else{
        	delta_B = (2*Math.PI)/this.projectionNumber;
        	delta_T = delta_B;
        	}
    	double beta;
    	for (int a = 0;a<=this.projectionNumber;a++){
    		beta = a*delta_B;
    		for (int i = 0;i<=this.detectorPixel;i++){
    			double s = this.deltaS * i - this.detectorLength/2;
    			double gamma = Math.signum(s) * Math.atan2(Math.abs(s), D+dD);
    			double theta = beta + gamma;
    			if(theta<0){theta = (Math.PI/2) + theta;}
    			double s_Para = D * Math.sin(gamma); //* Math.signum(s);	//Choosing a side of the detector.
    			float ids = (float) ((s_Para + this.detectorLength/2.0)/this.deltaS);
    			float idT = (float) (theta/delta_T);
    			if (ids>=0 && ids <= detectorPixel && idT >= 0 && idT <= projectionNumber ){
    				sino.setAtIndex(Math.round(ids), Math.round(idT), fanogram.getPixelValue(i, a));
    			}
    			else{continue;}
    		}
    	}
    	return sino;
    }
}
