package FlatPanelCTRecon;

import java.io.IOException;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.RamLakRampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class myOpenCL {
	
	private 
	CLContext clContext;
	CLDevice clDevice;
	
	public myOpenCL()
	{
		clContext = OpenCLUtil.createContext();
		clDevice  = clContext.getMaxFlopsDevice();
	}

	public static void main (String[] args)
	{
		//Setting up CONRAD for configuration else might throw exception in createContext
		CONRAD.setup();
		int size = 200;
		int numberProj      = 300;
		int detectorSpacing = 1;
		int numberDetPixel  = 400;

		myOpenCL openCLObj = new myOpenCL();
		
		//Exercise 4.1 :  Add the phantom to itself for 1.000.000 times on GPU
		//                and CPU and measure the time difference
		myphantom phan1 = new myphantom(size, size);
		/*myphantom phan2 = new myphantom(200, 200);
		openCLObj.AddPhantomToCPU(phan1, phan2);
	
		OpenCLGrid2D openCLPhan1 = new OpenCLGrid2D(phan1);
		OpenCLGrid2D openCLPhan2 = new OpenCLGrid2D(phan1);
		openCLObj.AddPhantomToGPU(openCLPhan1, openCLPhan2);

		// Exercise Sheet 4.2.		
		OpenCLGrid2D grid1 = openCLObj.createGrid1(200);
		OpenCLGrid2D grid2 = openCLObj.createGrid2(200);
		openCLObj.AddTwoOpenCLGrid2Ds(grid1, grid2, 200);*/
		
		// Exercise Sheet 4 - 3.
		// for creating a sinogram of myPhantom
		myParallelProject projector = new myParallelProject(numberProj, detectorSpacing, numberDetPixel);
 		Grid2D sinogram = projector.projectRayDriven(phan1);
		float [] pixelSpacingRecon = {(float) 0.2, (float) 0.2};
		
		// creating openCL sinogram
		OpenCLGrid2D sinogramCL = new OpenCLGrid2D(sinogram, openCLObj.clContext, openCLObj.clDevice);
		
		//CONRAD.setup();
		
		//Setting up Ramp Filter
		RampFilteringTool rampFilterTool = new RampFilteringTool();
		RampFilter rampFilter = new RamLakRampFilter();
		
		try
		{
			rampFilter.configure();
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		rampFilterTool.setRamp(rampFilter);
		Grid2D filteredSino = rampFilterTool.applyToolToImage(sinogramCL);
		filteredSino.show("filtered sinogram");
		
		long starttime= System.nanoTime();
		double sinogramSpacing = sinogram.getSpacing()[1];
		openCLObj.openCLBackProjection( sinogramCL, 
										openCLObj.clContext, 
										openCLObj.clDevice, 
										numberProj, 
										detectorSpacing, 
										numberDetPixel, 
										size, 
										pixelSpacingRecon,
										sinogramSpacing);
		
		long endtime= System.nanoTime();
		
		System.out.println("Time on GPU for PBP " + (endtime - starttime));
	}
	
	public void AddPhantomToCPU(Grid2D phantom1, Grid2D phantom2)
	{
		long starttime= System.nanoTime();
		
		for (int i = 0; i < 100000; i++)
		{
            NumericPointwiseOperators.addBy(phantom1, phantom2);
        }
		
		long endtime= System.nanoTime();
		
		System.out.println("Time with CPU " + (endtime - starttime)/1000);
		phantom1.show("in CPU");
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

	public void AddTwoOpenCLGrid2Ds(OpenCLGrid2D grid1, OpenCLGrid2D grid2, int phantomSize)
	{
		CLProgram program = null;
		try {
			program = clContext.createProgram(this.getClass().getResourceAsStream("AddOpenCLGrid2D.cl")).build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// Setup for first phantom
		int imageSize = grid1.getSize()[0] * grid1.getSize()[1];
		CLBuffer<FloatBuffer> imageBuffer1 = clContext.createFloatBuffer(imageSize, Mem.READ_ONLY);
		for (int i=0;i<grid1.getBuffer().length;++i)
		{
			imageBuffer1.getBuffer().put(grid1.getBuffer()[i]);
		}
		imageBuffer1.getBuffer().rewind();
		
		//Setup for second phantom
		int imageSize2 = grid2.getSize()[0] * grid2.getSize()[1];
		CLBuffer<FloatBuffer> imageBuffer2 = clContext.createFloatBuffer(imageSize2, Mem.READ_ONLY);
		for (int i=0;i<grid2.getBuffer().length;++i)
		{
			imageBuffer2.getBuffer().put(grid2.getBuffer()[i]);
		}
		imageBuffer2.getBuffer().rewind();
		
		// allocate memory for result grid
		CLBuffer<FloatBuffer> resultGrid = clContext.createFloatBuffer(imageSize, Mem.WRITE_ONLY);
				
		// copy parameter
		CLKernel kernel = program.createCLKernel("AddOpenCLGrid2D");
		kernel.putArg(resultGrid).putArg(imageBuffer1).putArg(imageBuffer2).putArg(phantomSize).putArg(phantomSize);
		
		// createCommandQueue
		CLCommandQueue queue = clDevice.createCommandQueue();
		queue.putWriteBuffer(resultGrid, true)
			 .putWriteBuffer(imageBuffer1, true)
			 .putWriteBuffer(imageBuffer2, true)
			 .put2DRangeKernel(kernel, 0, 0, (long)200, (long)200, 1, 1)
			 .finish()
			 .putReadBuffer(resultGrid, true)
			 .finish();
		
		// write resultGrid back to grid2D
		Grid2D result = new Grid2D(phantomSize,phantomSize);
		result.setSpacing(0.1, 0.1);
		resultGrid.getBuffer().rewind();
		for (int i = 0; i < result.getBuffer().length; ++i) {
			result.getBuffer()[i] = resultGrid.getBuffer().get();
		}
		result.show("added Picture");
		
		imageBuffer1.release();
		imageBuffer2.release();
	}
	
	public OpenCLGrid2D createGrid1(int size)
	{
		Grid2D phantomGrid = new Grid2D(size, size);
		int val1 = 1;
        int val2 = 2;
        double r1 = 0.45*size;
        
        int xCenter1 = size/2;
        int yCenter1 = size/2;
        
        int xr1 = xCenter1 -(int) (0.707*r1);
        int xr2 = xr1 + (int) (0.707*r1);
        int yr1 = yCenter1 - (int) (0.5*r1);
        int yr2 = yCenter1 + (int) (0.5*r1);
        
        phantomGrid.setSpacing(1,1);
        
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++) {
                if( Math.pow(i - xCenter1, 2)  + Math.pow(j - yCenter1, 2) <= (r1*r1) ){
                	phantomGrid.setAtIndex(i, j, val1);
                }
                
                if((i > xr1 && i < xr2) && (j > yr1 && j < yr2)){
                	phantomGrid.setAtIndex(i, j, val2);
                }
            }
        }
		
		
		OpenCLGrid2D openCLGrid = new OpenCLGrid2D(phantomGrid, clContext, clDevice);
		openCLGrid.show("Phantom 1");
		return openCLGrid;
	}
	
	public OpenCLGrid2D createGrid2(int size)
	{
		Grid2D phantomGrid = new Grid2D(size, size);
		int val1 = 1;
        double r1 = 0.45*size;
        double r2 = 0.5*r1;

        int xCenter2 = (int) ((0.5*size)-r2);
        int yCenter2 = size/2;
        
        phantomGrid.setSpacing(1,1);
        
        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++) {
                if( Math.pow(i - xCenter2, 2)  + Math.pow(j - yCenter2, 2) <= (r2*r2) ){
                	phantomGrid.setAtIndex(i, j, val1);
                }
            }
        }
		
		OpenCLGrid2D openCLGrid = new OpenCLGrid2D(phantomGrid, clContext, clDevice);
		openCLGrid.show("Phantom 2");
		return openCLGrid;
	}

	public Grid2D openCLBackProjection(OpenCLGrid2D sino, 
									   CLContext	context, 
									   CLDevice		device, 
									   int 			numberProj,
									   float 		detectorSpacing, 
									   int 			numberDetPixel, 
									   int 			sizeRecon, 
									   float 		pixelSpacingRecon[],
									   double sinoSpacing) 
	{
		CLProgram program = null;
		try 
		{
			//program = context.createProgram(this.getClass().getResourceAsStream("TrialOpenCLBackProjection.cl")).build();
			program = context.createProgram(this.getClass().getResourceAsStream("TrialOpenCLBackProjection.cl")).build();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
			System.exit(-1);
		}
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16);
		int gridReconSizeX = OpenCLUtil.roundUp(localWorkSize, sizeRecon);
		int gridReconSizeY = OpenCLUtil.roundUp(localWorkSize, sizeRecon);;
		int imageSize = gridReconSizeX * gridReconSizeY;
		
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		
		// create memory for result back projection grid
		CLBuffer<FloatBuffer> resultBPGrid = context.createFloatBuffer(imageSize, Mem.WRITE_ONLY);
		
		// create buffer for sinogram
		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sino.getHeight() * sino.getWidth(), Mem.READ_ONLY);
		for (int i=0;i<sino.getBuffer().length;++i)
		{
			sinoBuffer.getBuffer().put(sino.getBuffer()[i]);
		}
		sinoBuffer.getBuffer().rewind();
		
		
		// copy parameters
		//CLKernel kernel = program.createCLKernel("TrialOpenCLBackProjection");
		CLKernel kernel = program.createCLKernel("TrialOpenCLBackProjection");
		  kernel.putArg(resultBPGrid)
				.putArg(sinoBuffer)
				.putArg(numberProj)
				.putArg(detectorSpacing)
				.putArg(numberDetPixel)
				.putArg(sizeRecon)
				.putArg(pixelSpacingRecon[0])
				.putArg(pixelSpacingRecon[1])
				.putArg((float)sino.getOrigin()[0])
				.putArg((float)sino.getOrigin()[1])
				.putArg(sinoSpacing);
	
		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue.putWriteBuffer(resultBPGrid, true)
			 .finish()
			 .putWriteBuffer(sinoBuffer, true)
			 .put2DRangeKernel(kernel, 0, 0, (long)gridReconSizeX, (long)gridReconSizeY, localWorkSize, localWorkSize) //Program Crashes here with no error msg.
			 .finish()
			 .putReadBuffer(resultBPGrid, true)
			 .finish();
		
		// write resultGrid back to grid2D
		Grid2D result = new Grid2D(sizeRecon, sizeRecon);
		result.setSpacing(0.1, 0.1);
		resultBPGrid.getBuffer().rewind();
		for (int i = 0; i < result.getBuffer().length; ++i) {
			result.getBuffer()[i] = resultBPGrid.getBuffer().get();
		}
		result.show("Back Projection using openCL");
		return result;
	}

	
}
