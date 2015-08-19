package FlatPanelCTRecon;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class myParallelProject
{
	int projectionNumber;
	double detectorSpacing;
	int detectorPixel;
	double sourceIsoDistance;
	
	public myParallelProject(int projectionNumber, double detectorSpacing, int detectorPixel)
	{
		this.projectionNumber = projectionNumber;
		this.detectorSpacing = detectorSpacing;
		this.detectorPixel = detectorPixel;
	}
	
	public Grid2D projectRayDriven(Grid2D grid) 
	{
		final int fs = 3;		//Sampling rate
		double Dtheta = Math.PI/(projectionNumber);
		float detSize = 400;	//Detector size in millimeters
		
		//Sinogram memory allocation
		Grid2D sino = new Grid2D(new float[projectionNumber*detectorPixel],detectorPixel, projectionNumber );
		sino.setSpacing(1, Dtheta);
		
		//Grid to 'box' conversion for easier detector rotations.
		Translation t = new Translation(-(grid.getSize()[0] * grid.getSpacing()[0])/2, -(grid.getSize()[1] * grid.getSpacing()[1])/2,-1);
		Box box = new Box((grid.getSize()[0] * grid.getSpacing()[0]), (grid.getSize()[1] * grid.getSpacing()[1]), 2);
		Transform move2origin = t.inverse();
		box.applyTransform(t);
		
		//Iteration over the detector positions. (Angle)
		for(int a = 0;a<= projectionNumber;++a)
		{
			double sT = Math.sin(Dtheta*a);
			double cT = Math.cos(Dtheta*a);
			
			//Iteration over the detector 'bins'
			for(int i = 0;i<=detectorPixel;++i)
			{
				//Projecting an orthogonal line from the detector bin to check for intersection with the object.  
				double s = detectorSpacing*i - detSize/2;			//Distance from the center of the detector to the considered detector bin
				double sum = .0;									//Integral result.
				PointND P0 = new PointND(s*cT,s*sT,.0d);			//Point on the detector line.
				PointND P1 = new PointND(s*cT - sT,s*sT + cT,.0d);	//Point sitting on the orthogonal line with respect on the detector.
				StraightLine li = new StraightLine(P0, P1);			//Orthogonal line through the detector bin. 
				ArrayList<PointND> cutSites = box.intersect(li);
				
				//Skipping the projections which does not intersect with the object.
				if(cutSites.size() == 0)
					continue;
				
				//Obtaining the line segment which cuts the object on which the line integral will be sampled.
				P0 = cutSites.get(0);
				P1 = cutSites.get(1);
				SimpleVector pdifference = new SimpleVector(P1.getAbstractVector());
				pdifference.subtract(P0.getAbstractVector());
				double dist = pdifference.normL2();
				pdifference.divideBy(dist*fs);
				
				
				//Transforming the cutting edge of the line from the box coordinates to image coordinates.
				P1 = move2origin.transform(P1);
				
				//Iterating over the line integral samples.
				for (double j = 0.0; j < dist * fs; ++j)
				{
					PointND sample = new PointND(P1);
					sample.getAbstractVector().subtract(pdifference.multipliedBy(j));
					double x = sample.get(0) / grid.getSpacing()[0], y = sample.get(1) / grid.getSpacing()[1];

					if (grid.getSize()[0] <= x + 1
							|| grid.getSize()[1] <= y + 1
							|| x < 0 || y < 0)
						continue;

					sum += InterpolationOperators.interpolateLinear(grid, x, y);
					
				}				
				
				// write integral value into the sinogram.
				sino.setAtIndex(i, a, (float)sum);
			}
			
		}
		
		return sino;
	
	}

}
