package FlatPanelCTRecon;

import edu.stanford.rsl.apps.gui.ReconstructionPipelineFrame;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.CosineWeightingTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.HanningRampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.opencl.OpenCLBackProjector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;

public class myShortScanUsingCONRAD 
{
	 public static void main (String [] args) throws Exception
     {
		 //Exercise 5.1 -> Projection Data
	    new ImageJ();
	    // call the ImageJ routine to open the image:
	    ImagePlus imagePlus = IJ.openImage("/proj/ciptmp/FCRSS15/data/DensityProjection_No248_Static60_0.8deg_REFERENCE.tif");
	    // convert from ImageJ to Grid3D.
	    Grid3D imageGrid3D = ImageUtil.wrapImagePlus(imagePlus);
	    // display the data from given location.
	    imageGrid3D.show("Data from file");
	    
	    //GUI PART
	    //Exercise 5.2 -> Projection Matrices and CONRAD GUI
	    //This configure the GUI calling the global configuration file.
	    CONRAD.setup();
        ReconstructionPipelineFrame pipeLine = new ReconstructionPipelineFrame();
        pipeLine.setVisible(true);
        
        //Exercise 5.3 -> Parker Weights
        /*arkerWeightingTool parkerWeight = new ParkerWeightingTool();
        parkerWeight.configure();
        for(int i = 0; i < imagePlus.getNSlices(); i++)
        {
        	parkerWeight.setImageIndex(i);
            Grid2D parkerGrid = parkerWeight.applyToolToImage(imageGrid3D.getSubGrid(i));
            imageGrid3D.setSubGrid(i, parkerGrid);    
        }
        imageGrid3D.show("Parker Weighted Image");   */            
        
      //Exercise 5.4: Applying cosine weight, ramp-filter and OpenCLBackProjection in CONRAD API  
        CosineWeightingTool cosineWeight = new CosineWeightingTool();
        cosineWeight.configure();
        RampFilteringTool rampFilter = new RampFilteringTool();
        RampFilter hanningRampFilter = new HanningRampFilter();
        hanningRampFilter.configure();
        rampFilter.setRamp(hanningRampFilter);
        OpenCLBackProjector openCL = new OpenCLBackProjector();
        openCL.configure();
        
        ImageFilteringTool [] tools = {cosineWeight, rampFilter, openCL};
		ImageUtil.applyFiltersInParallel(imageGrid3D, tools).show("Cosine and Ramp Filtered image using OpenCLbackProjection");
     }
}
