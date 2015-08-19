package FlatPanelCTRecon;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class myOpenCL {
	
	public static void main (String[] args)
	{
		CLContext context = OpenCLUtil.createContext();
		CLDevice[] devices = context.getDevices();
		CLDevice device = context.getMaxFlopsDevice(); 
		
		myOpenCL openCLObj = new myOpenCL();
		
		//Exercise 4.1 :  Add the phantom to itself for 1.000.000 times on GPU
		//                and CPU and measure the time difference
		myphantom phan = new myphantom(200, 200); 
		openCLObj.AddPhantomToCPU(phan);
	
		OpenCLGrid2D openCLPhan1 = new OpenCLGrid2D(phan, context, device);
		OpenCLGrid2D openCLPhan2 = new OpenCLGrid2D(phan, context, device);
		openCLObj.AddPhantomToGPU(openCLPhan1, openCLPhan2);
		
	}
	
	public void AddPhantomToCPU(Grid2D phantom)
	{
		long starttime= System.nanoTime();
		myphantom localPhant = new myphantom(200,200)  ;
		
		for (int i = 0; i < 100000; i++)
		{
            NumericPointwiseOperators.addBy(phantom, localPhant);
        }
		
		long endtime= System.nanoTime();
		
		System.out.println("Time with CPU " + (endtime - starttime)/1000);
		phantom.show("in CPU");
	}
	
	public void AddPhantomToGPU(OpenCLGrid2D openCLPhantom1, OpenCLGrid2D openCLPhantom2)
	{
		long starttime= System.nanoTime();
		
	
		for (int i = 0; i < 100000; i++)
		{
            NumericPointwiseOperators.addBy(openCLPhantom1, openCLPhantom2);
        }
		
		long endtime= System.nanoTime();
		
		System.out.println("Time with GPU " + (endtime - starttime)/1000);
		openCLPhantom1.show("in GPU");
	}
}
