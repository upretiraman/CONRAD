package brainPerfusionCT;

import ij.io.FileInfo;
import ij.io.ImageWriter;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import brainPerfusionCT.PerfusionBrainPhantomConfig;

public class BrainPhantomIO{
	
	String phantom_directory;
	String projection_output_directory;
	int	   nProjections, detectorWidth, detectorHeight;
	
	public BrainPhantomIO(String phantom_directory, String projection_output_directory, int nProjections, int detectorWidth, int detectorHeight)
	{
		this.phantom_directory = phantom_directory;
		this.projection_output_directory = projection_output_directory;		
		this.nProjections = nProjections;
		this.detectorWidth = detectorWidth;
		this.detectorHeight = detectorHeight;
	}
	
	public float[] loadVolume(String filename) throws IOException
	{
		String fp_fn = phantom_directory + filename;

		DataInputStream dis_volume = new DataInputStream(new BufferedInputStream(new FileInputStream(fp_fn)));
		float[] volumeBuffer = new float[PerfusionBrainPhantomConfig.SZ];
		
		for(int i = 0; i < PerfusionBrainPhantomConfig.SZ; i++) {
			// :1000 + 1: scale from HU values to density (0 HU = 1 mg/mm^3; voxel volume is 1 mm^3)
			volumeBuffer[i] = dis_volume.readFloat()/1000 + 1;
			if(volumeBuffer[i] < 0) volumeBuffer[i] = 0;
		}
		dis_volume.close();

		return volumeBuffer;
	}

	public float[] loadVolumeNoScale(String filename) throws IOException
	{
		String fp_fn = phantom_directory + filename;

		DataInputStream dis_volume = new DataInputStream(new BufferedInputStream(new FileInputStream(fp_fn)));
		float[] volumeBuffer = new float[PerfusionBrainPhantomConfig.SZ];
		
		for(int i = 0; i < PerfusionBrainPhantomConfig.SZ; i++) {
			volumeBuffer[i] = dis_volume.readFloat();
		}
		dis_volume.close();

		return volumeBuffer;
	}
	
	public Grid3D loadMask(String filename, int sz_x, int sz_y, int sz_z) throws IOException
	{
		String fp_fn = projection_output_directory + filename;

		DataInputStream dis_volume = new DataInputStream(new BufferedInputStream(new FileInputStream(fp_fn)));
		Grid3D volume = new Grid3D(sz_x, sz_y, sz_z);
		
		
		for(int z = 0; z < sz_z; z++) {
			for(int y = 0; y < sz_y; y++) {
				for(int x = 0; x < sz_x; x++) {
					volume.setAtIndex(x, y, z, (dis_volume.readByte()==0)?0.f:1.f);
				}
			}
		}
		dis_volume.close();

		return volume;
	}
	
	
	public void saveVolume(float[] volumeBuffer, String filename) throws IOException
	{
		String fp_fn = projection_output_directory + filename;

		DataOutputStream dos_proj = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fp_fn)));
		
		for(int k = 0; k < volumeBuffer.length; k++) {
			dos_proj.writeFloat(volumeBuffer[k]);
		}
		
		dos_proj.close();
	}	

	public void saveGrid3D(Grid3D grid, String filename) throws IOException
	{
		String fp_fn = projection_output_directory + filename;

		DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fp_fn)));
		for(int z = 0; z < grid.getSize()[2]; z++) {
			for(int y = 0; y < grid.getSize()[1]; y++) {
				for(int x = 0; x < grid.getSize()[0]; x++) {
					dos.writeFloat(grid.getAtIndex(x, y, z));
				}
			}
		}
		
		dos.close();
	}	

	public Grid3D loadGrid3D(String filename, int sz_x, int sz_y, int sz_z) throws IOException
	{
		String fp_fn = projection_output_directory + filename;	
		Grid3D grid3D = new Grid3D(sz_x, sz_y, sz_z);
		DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(fp_fn)));
		
		for(int z = 0; z < sz_z; z++) {
			for(int y = 0; y < sz_y; y++) {
				for(int x = 0; x < sz_x; x++) {
					grid3D.setAtIndex(x, y, z, dis.readFloat());	
				}
			}
		}
		dis.close();
		
		return grid3D;

	}		
	
	
	public Grid2D[] loadProjectionData(String filename) throws IOException
	{
		String fp_fn = projection_output_directory + filename;
		
		DataInputStream dis_proj = new DataInputStream(new BufferedInputStream(new FileInputStream(fp_fn)));
		Grid2D[] projection_images = new Grid2D[nProjections];
		
		for(int k = 0; k < projection_images.length; k++) {
			projection_images[k] = new Grid2D(detectorWidth,detectorHeight);
			float[] data = projection_images[k].getBuffer();
			for(int i = 0; i < data.length; i++)
				data[i] = dis_proj.readFloat();
		}
		dis_proj.close();
		
		return projection_images;
	}
	
	public Grid2D loadProjectionImage(String filename, int idx) throws IOException
	{
		final int SIZE_FLOAT = Float.SIZE/Byte.SIZE; 
		String fp_fn = projection_output_directory + filename;
		
		Grid2D proj = new Grid2D(detectorWidth,detectorHeight);
		
		DataInputStream dis_proj = new DataInputStream(new BufferedInputStream(new FileInputStream(fp_fn)));
		
		dis_proj.skipBytes(idx*detectorWidth*detectorHeight*SIZE_FLOAT);
		
		float[] data = proj.getBuffer();
		for(int i = 0; i < data.length; i++)
			data[i] = dis_proj.readFloat();
		
		dis_proj.close();
		return proj;
		
	}
	
	public void saveProjectionData(Grid2D[] projection_images, String filename) throws IOException
	{
					
		String fp_fn = projection_output_directory + filename;
		new File(projection_output_directory).mkdirs();
		DataOutputStream dos_proj = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fp_fn)));
		
		FileInfo fI = new FileInfo();
		fI.fileFormat = FileInfo.RAW;
		fI.fileType = FileInfo.GRAY32_FLOAT;
		fI.height = projection_images[0].getHeight();
		fI.width = projection_images[0].getWidth();
		fI.nImages = 1;
		fI.intelByteOrder = false;
		
		fI.directory = phantom_directory;
		fI.fileName = filename;		
		
		ImageWriter writer = new ImageWriter(fI);
		
		for(int k = 0; k < projection_images.length; k++) {
			fI.pixels = projection_images[k].getBuffer();
			writer.write(dos_proj);
		}
		dos_proj.close();
	}
	
//	public void saveProjectionData(Grid2D[] projection_images, String filename) throws IOException
//	{
//		String fp_fn = projection_output_directory + filename;
//
//		DataOutputStream dos_proj = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fp_fn)));
//		
//		for(int k = 0; k < projection_images.length; k++) {
//			float[] data = projection_images[k].getBuffer();
//			for(int i = 0; i < data.length; i++) {
//				dos_proj.writeFloat(data[i]);
//			}
//		}
//		dos_proj.close();
//	}

	public void appendData(float[] data, String filename) throws IOException
	{
		String fp_fn = projection_output_directory + filename;

		DataOutputStream dos_proj = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(fp_fn,true)));		
		for(int i = 0; i < data.length; i++) {
			dos_proj.writeFloat(data[i]);
		}
		dos_proj.close();
	}
	
	public void showVolBuffer(float[] volBuffer) 
	{
		Grid3D show_vol = new Grid3D(PerfusionBrainPhantomConfig.SZ_X,PerfusionBrainPhantomConfig.SZ_Y,PerfusionBrainPhantomConfig.SZ_Z);
		for(int z=0; z < PerfusionBrainPhantomConfig.SZ_Z; z++) {
			for(int y=0; y < PerfusionBrainPhantomConfig.SZ_Y; y++) {
				for(int x=0; x < PerfusionBrainPhantomConfig.SZ_X; x++) {
					show_vol.setAtIndex(x, y, z, volBuffer[z*PerfusionBrainPhantomConfig.SZ_X*PerfusionBrainPhantomConfig.SZ_Y+y*PerfusionBrainPhantomConfig.SZ_X+x]);
				}
			}
		}
		show_vol.show();
	}
}