/*
 * Copyright (C) 2014 Michael Manhart
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package brainPerfusionCT;

import java.util.Vector;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class PerfusionBrainPhantomConfig {
	
	public final static int SZ_X = 256;
	public final static int SZ_Y = 256;
	public final static int SZ_Z = 256;
	public final static int SZ = SZ_X*SZ_Y*SZ_Z;
	public final static float SZ_VOXEL_MM = 1.f;
	
	
	public String phantom_directory;
	public float phantom_sampling; 

	public String calibration_fwd_proj_matrices_file;
	public String calibration_bwd_proj_matrices_file;
		
	public String projection_output_directory;
	
	public float t_start;
	public float t_rot;
	public float t_stop;
	public int	 n_rot;
	
	public Vector<Float> spectrum_binning_keV;
	public float spectrum_sampling_keV;
	public float spectrum_peak_keV;
	public float spectrum_time_current_product_mAs;
	

	
	public boolean noise_add;
	public int I_0;
}
