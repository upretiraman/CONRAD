package FlatPanelCTRecon;
import java.io.IOException;
import java.nio.FloatBuffer;

import ij.ImageJ;

import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.HanningRampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.RamLakRampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class testOpenCL {
	
	public static OpenCLGrid2D createGrid1(int size, CLContext context, CLDevice device )
	{
		Grid2D p = new Grid2D(size, size);
		//Circle
		int ii;
		int jj;
		for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			ii = i - size/4;
			jj = j - size/4;
			if ((ii * ii + jj * jj) < 12 * size) {
				p.setAtIndex(i, j, 15);
			}
		}
		OpenCLGrid2D pCL = new OpenCLGrid2D(p, context, device);
		pCL.show("grid1");
		return pCL;
	}
	
	public static OpenCLGrid2D createGrid2(int size, CLContext context, CLDevice device )
	{
		Grid2D p = new Grid2D(size, size);
		//Circle
		int ii;
		int jj;
		for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			ii = i - size/2;
			jj = j - size/2;
			if ((ii * ii + jj * jj) < 12 * size) {
				p.setAtIndex(i, j, 100);
			}
		}
		OpenCLGrid2D pCL = new OpenCLGrid2D(p, context, device);
		pCL.show("grid2");
		return pCL;
	}
	
	public void AddPhantomToCPUandGPU(myphantom p, OpenCLGrid2D phantomCL, OpenCLGrid2D addPhanCL)
	{
		int number = 10000;
		myphantom addP = new myphantom(200,200);		
		long starttime= System.nanoTime();

		//on CPU
		for (int i = 0; i < number; i++){
            NumericPointwiseOperators.addBy(addP, p);
        }
		
		long endtime = System.nanoTime();
		long timecost = endtime - starttime;
		addP.show();
		
		System.out.println("Time with CPU " + timecost/1000);
		starttime= System.nanoTime();

		//openCL on GPU
		for (int i = 0; i < number; i++){
            NumericPointwiseOperators.addBy(addPhanCL, phantomCL);
		}
		
		endtime= System.nanoTime();
		timecost= endtime - starttime; 
		addPhanCL.show();
		
		System.out.println("Time on GPU " + timecost/1000);
	}
	
	public void AddTwoOpenCLGrid2Ds(OpenCLGrid2D grid1, OpenCLGrid2D grid2, CLContext context, CLDevice device, int size )
	{
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("AddOpenCLGrid2D.cl"))
					.build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		int gridSizeX = size;
		int gridSizeY = size;
		
		int imageSize = grid1.getSize()[0] * grid1.getSize()[1];
		
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		
		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(imageSize, Mem.READ_ONLY);
		for (int i=0;i<grid1.getBuffer().length;++i){
			imageBuffer.getBuffer().put(grid1.getBuffer()[i]);
		}
		imageBuffer.getBuffer().rewind();
		CLImage2d<FloatBuffer> imageGrid1 = context.createImage2d(
				imageBuffer.getBuffer(), grid1.getSize()[0], grid1.getSize()[1], format);
		
		CLBuffer<FloatBuffer> imageBuffer2 = context.createFloatBuffer(imageSize, Mem.READ_ONLY);
		for (int i=0;i<grid1.getBuffer().length;++i){
			imageBuffer2.getBuffer().put(grid2.getBuffer()[i]);
		}
		imageBuffer2.getBuffer().rewind();
		CLImage2d<FloatBuffer> imageGrid2 = context.createImage2d(
				imageBuffer2.getBuffer(), grid2.getSize()[0], grid2.getSize()[1], format);

		// create memory for result grid
		CLBuffer<FloatBuffer> resultGrid = context.createFloatBuffer(imageSize, Mem.WRITE_ONLY);
		
		// copy params
		CLKernel kernel = program.createCLKernel("AddOpenCLGrid2D");
		kernel.putArg(resultGrid).putArg(imageBuffer).putArg(imageBuffer2)
			.putArg(gridSizeX).putArg(gridSizeY);
		
		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
			//.putWriteImage(imageGrid1, true)
			//.putWriteImage(imageGrid2, true)
			.putWriteBuffer(resultGrid, true)
			.putWriteBuffer(imageBuffer, true)
			.putWriteBuffer(imageBuffer2, true)
			.put2DRangeKernel(kernel, 0, 0,(long)gridSizeX,(long)gridSizeY,1, 1)
			.finish()
			.putReadBuffer(resultGrid, true)
			.finish();
		
		// write resultGrid back to grid2D
		Grid2D result = new Grid2D(size,size);
		result.setSpacing(0.1, 0.1);
		resultGrid.getBuffer().rewind();
		for (int i = 0; i < result.getBuffer().length; ++i) {
			result.getBuffer()[i] = resultGrid.getBuffer().get();
		}
		result.show("addedPicture");
		
		imageBuffer.release();
		imageBuffer2.release();
	}
	
	public Grid2D openCLBackProjection(OpenCLGrid2D sino, CLContext context, CLDevice device, 
			int numberProj, double detectorSpacing, int numberDetPixel, int[] sizeRecon, double pixelSpacingRecon[]) 
	{
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("OpenCLBackProjection.cl")).build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		int gridReconSizeX = sizeRecon[0];
		int gridReconSizeY = sizeRecon[1];
		int imageSize = gridReconSizeX * gridReconSizeY;
		
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		
		// create memory for result backprojection grid
		CLBuffer<FloatBuffer> resultBPGrid = context.createFloatBuffer(imageSize, Mem.WRITE_ONLY);
		
		// create buffer for sinogram
		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sino.getHeight() * sino.getWidth(), Mem.READ_ONLY);
		for (int i=0;i<sino.getBuffer().length;++i){
			sinoBuffer.getBuffer().put(sino.getBuffer()[i]);
		}
		sinoBuffer.getBuffer().rewind();
		
		// copy params
		CLKernel kernel = program.createCLKernel("OpenCLBackProjection");
		kernel.putArg(resultBPGrid)
		.putArg(sinoBuffer)
		.putArg(numberProj)
		.putArg(detectorSpacing)
		.putArg(numberDetPixel)
		.putArg(sizeRecon[0])
		.putArg(sizeRecon[1])
		.putArg(pixelSpacingRecon[0])
		.putArg(pixelSpacingRecon[1])		
		.putArg((float)sino.getOrigin()[0])
		.putArg((float)sino.getOrigin()[1]);
	
		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
			//.putWriteImage(imageGrid1, true)
			//.putWriteImage(imageGrid2, true)
			.putWriteBuffer(resultBPGrid, true)
			.putWriteBuffer(sinoBuffer, true)
			.put2DRangeKernel(kernel, 0, 0
					,(long)gridReconSizeX		//i
					,(long)gridReconSizeY		//j
					,1,1)
			//.put2DRangeKernel(kernel, 0, 0, globalWorkSizeBeta, globalWorkSizeT, localWorkSize, localWorkSize)  maybe add this worksize?
			.finish()
			.putReadBuffer(resultBPGrid, true)
			.finish();
;
		// write resultGrid back to grid2D
		Grid2D result = new Grid2D(sizeRecon[0], sizeRecon[1]);
		result.setSpacing(pixelSpacingRecon[0], pixelSpacingRecon[1]);
		resultBPGrid.getBuffer().rewind();
		for (int i = 0; i < result.getBuffer().length; ++i) {
			result.getBuffer()[i] = resultBPGrid.getBuffer().get();
		}
		result.show("Backprojection");
		return result;
	}
	
	public static void main(String[] args) {
		new ImageJ();
		testOpenCL o = new testOpenCL();
		CLContext context = OpenCLUtil.createContext();
		//CLDevice[] devices = context.getDevices();
		CLDevice device = context.getMaxFlopsDevice();
		
		// Exercise Sheet 4 - 1.		
		
		myphantom p = new myphantom(200,200);
		int[] size = p.getSize();
		/*OpenCLGrid2D phantomCL = new OpenCLGrid2D(p, context, device);
		OpenCLGrid2D addPhanCL = new OpenCLGrid2D(p, context, device);
		o.AddPhantomToCPUandGPU(p, phantomCL, addPhanCL);
		
		// Exercise Sheet 4 - 2.		
		OpenCLGrid2D grid1 = createGrid1(size, context, device);
		OpenCLGrid2D grid2 = createGrid2(size, context, device);
		o.AddTwoOpenCLGrid2Ds(grid1, grid2, context, device, size);*/
		
		// Exercise Sheet 4 - 3.
		// for creating a sinogram from class Phantom
		p.setOrigin(-(size[0] - 1) * p.getSpacing()[0] / 2, -(size[1] - 1) * p.getSpacing()[1]/ 2);
		float d = (float) (Math.sqrt(2) * p.getHeight() * p.getSpacing()[0]);		
		myParallelProject projector = new myParallelProject(300,1,400);
		myParallelBackProjection backProjector = new myParallelBackProjection(300,1,400);
 		//Grid2D sinogram = projector.projectRayDriven(p);
 		//Uncomment following code for the filtered sinogram
		Grid2D sinogram = backProjector.filterSino(projector.projectRayDriven(p));
 		double detectorSpacing = projector.detectorSpacing;
		double [] pixelSpacingRecon = {p.getSpacing()[0], p.getSpacing()[1]};
		int numberProj = projector.projectionNumber;
		int numberDetPixel = projector.detectorPixel;

		OpenCLGrid2D sinogramCL = new OpenCLGrid2D(sinogram, context, device);
		
		//ramp filter
		CONRAD.setup();
		RampFilteringTool r = new RampFilteringTool();
		RampFilter ramp = new RamLakRampFilter();
		try {
			ramp.configure();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		r.setRamp(ramp);
		Grid2D filteredSino = r.applyToolToImage(sinogramCL);
		filteredSino.show("filtered sinogram");
		
		
		Grid2D backprojection = new Grid2D(p.getSize()[0], p.getSize()[1]);
		long starttime= System.nanoTime();
		
		backprojection = o.openCLBackProjection(sinogramCL, context, device, numberProj, detectorSpacing, numberDetPixel, size, pixelSpacingRecon);
	
		long endtime= System.nanoTime();
		
		System.out.println("Time on GPU for PBP " + (endtime - starttime));		
		

	}

}