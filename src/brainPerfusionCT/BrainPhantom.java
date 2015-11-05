package brainPerfusionCT;


import java.io.File;
import java.io.IOException;
import java.util.Vector;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;

import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.StatisticsUtil;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.trajectories.ConfigFileBasedTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.opencl.OpenCLForwardProjectorDynamicVolume;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.Mixture;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.physics.materials.utils.MaterialUtils;
import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;
import brainPerfusionCT.PerfusionBrainPhantomConfig;

public class BrainPhantom {
	PerfusionBrainPhantomConfig cfg;
	BrainPhantomIO     io;
	
	protected float[] vol_skull = null;
	protected float[] vol_tissue = null;
	protected float[] vol_contrast_prev = null;
	protected float[] vol_contrast_next = null;
	protected int     vol_contrast_prev_idx = -1;
	protected int     vol_contrast_next_idx = -1;
	
	protected Trajectory g_fwd;
	protected Trajectory g_bwd;
	
	OpenCLForwardProjectorDynamicVolume opencl_fwdproj;
	
	protected boolean initialized = false;
	
	public static void main(String [] args)
	{
		Configuration.loadConfiguration();
		PerfusionBrainPhantomConfig cfg = new PerfusionBrainPhantomConfig();
		cfg.phantom_directory = "//home//cip//medtech2014//ir31yrit//Documents//phantom//";
		cfg.phantom_sampling = 1.f;
		
		cfg.calibration_fwd_proj_matrices_file = "//home//cip//medtech2014//ir31yrit//Documents//phantom//PMatrix248.txt";
		cfg.calibration_bwd_proj_matrices_file = "//home//cip//medtech2014//ir31yrit//Documents//phantom//PMatrix248.txt";
		
		cfg.projection_output_directory = "//home//cip//medtech2014//ir31yrit//Documents//phantom//classifierTest//";
		
		cfg.t_start = 0;
		cfg.t_rot = 0;
		cfg.t_stop = 1.f;
		cfg.n_rot = 1;
		
		cfg.spectrum_binning_keV = new Vector<Float>();
		cfg.spectrum_binning_keV.add(10.f);
		cfg.spectrum_binning_keV.add(60.f);
		cfg.spectrum_peak_keV = 60;
		cfg.spectrum_sampling_keV = 1.f;
		cfg.spectrum_time_current_product_mAs = 2.5f;
		
		cfg.noise_add = false;
		
		BrainPhantom pbp = new BrainPhantom(cfg);

		try {
			pbp.createProjectionData();
			//pbp.createSpectralCalibrationPhantom();
		}catch(IOException e){
			System.err.println("PerfusionBrainPhantom.createProjectionData(): IO error!");
			e.printStackTrace();
		}
		System.exit(0);
	}
	
	public BrainPhantom(PerfusionBrainPhantomConfig cfg)
	{
		this.cfg = cfg;

		g_fwd = Configuration.getGlobalConfiguration().getGeometry();//ConfigFileBasedTrajectory.openAsGeometrySource(cfg.calibration_fwd_proj_matrices_file, Configuration.getGlobalConfiguration().getGeometry());
		g_bwd = ConfigFileBasedTrajectory.openAsGeometrySource(cfg.calibration_bwd_proj_matrices_file, Configuration.getGlobalConfiguration().getGeometry());
		
		if(cfg.spectrum_binning_keV.size() < 2)
		{
			System.err.println("PerfusionBrainPhantom.PerfusionBrainPhantom(): Need at least two values for spectral binning (min an max keV)!");
			initialized = false;
			return;
		}
		io = new BrainPhantomIO(cfg.phantom_directory,cfg.projection_output_directory,g_fwd.getNumProjectionMatrices(),
				g_fwd.getDetectorWidth(),g_fwd.getDetectorHeight());		
		
		try {
			vol_skull = io.loadVolume("skull");//skull.raw
			vol_tissue = io.loadVolume("tissue");//brain.raw
			initialized = true;
		}catch(IOException e){
			System.err.println("PerfusionBrainPhantom.PerfusionBrainPhantom(): Error reading phantom input data!");
			e.printStackTrace();
			initialized = false;
		}
	}
	
	public void createProjectionData() throws IOException
	{
		if(!initialized)
		{
			System.err.println("PerfusionBrainPhantom.createProjectionData(): not initialized!");
			return;
		}
		
		// -----------------------------------------------------
		// material decomposed dynamic forward projection 
		// -----------------------------------------------------
		
		//dynamicDensityForwardProjection();
		
		// -----------------------------------------------------
		// generation of binned energy selective projection data 
		// -----------------------------------------------------
		
		generateBinnedProjectionData();	
		//generateMaskProjectionData();
		
		
		// -----------------------------------------------------
		// output parameters
		// -----------------------------------------------------	
		
		try {
			File fos = new File(cfg.projection_output_directory + "//PerfusionBrainPhantomConfig.xml");
			JAXBContext context = JAXBContext.newInstance( PerfusionBrainPhantomConfig.class );
			Marshaller m = context.createMarshaller();
			m.setProperty( Marshaller.JAXB_FORMATTED_OUTPUT, Boolean.TRUE );
			m.marshal( cfg, fos );	
		} catch (JAXBException e) {
			System.err.println("PerfusionBrainPhantom.createProjectionData(): Error in parameter output");
			e.printStackTrace();
		}
		System.out.println("done.");

	}

	protected void dynamicDensityForwardProjection() throws IOException
	{
		int[] volumeSize = new int[3];
		float[] voxelSize = new float[3];
		volumeSize[0] = PerfusionBrainPhantomConfig.SZ_X;
		volumeSize[1] = PerfusionBrainPhantomConfig.SZ_Y;
		volumeSize[2] = PerfusionBrainPhantomConfig.SZ_Z;  
		voxelSize[0] = PerfusionBrainPhantomConfig.SZ_VOXEL_MM;
		voxelSize[1] = PerfusionBrainPhantomConfig.SZ_VOXEL_MM;
		voxelSize[2] = PerfusionBrainPhantomConfig.SZ_VOXEL_MM;
		opencl_fwdproj = new OpenCLForwardProjectorDynamicVolume();
		try {
			opencl_fwdproj.configure(g_fwd, g_bwd, volumeSize, voxelSize);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Grid2D[] projections = new Grid2D[g_fwd.getNumProjectionMatrices()];
		// skull
		setVolumeToProject(vol_skull);
		for(int projection_id = 0; projection_id < g_fwd.getNumProjectionMatrices(); projection_id++)
		{
			System.out.println("Forward projection: skull forward rotation projection " + (projection_id+1));
			projections[projection_id] = applyForwardProjection(projection_id, true);
		}
		io.saveProjectionData(projections, "skull_fwd");
		// tissue
		setVolumeToProject(vol_tissue);
		for(int projection_id = 0; projection_id < g_fwd.getNumProjectionMatrices(); projection_id++)
		{
			System.out.println("Forward projection: tissue projection " + (projection_id+1));
			projections[projection_id] = applyForwardProjection(projection_id, true);
		}
		io.saveProjectionData(projections, "tissue_fwd");
		// contrast agent
		setVolumeToProject(io.loadVolume("mask"));
			
		for(int projection_id = 0; projection_id < g_fwd.getNumProjectionMatrices(); projection_id++)
		{
			System.out.println("Forward projection contrast agent: projection " + (projection_id+1));
			projections[projection_id] = applyForwardProjection(projection_id, true);
		}
		String fn_proj_out = String.format("vessel_contrast");
		io.saveProjectionData(projections, fn_proj_out);		
	}
	
	protected void generateBinnedProjectionData() throws IOException
	{
		System.out.println();
		System.out.println("Create binned spectral projection data ...");
		// load input data
		System.out.println("Loading static projections ...");
		Grid2D[] skull_fwd = io.loadProjectionData("skull_fwd");
		Grid2D[] tissue_fwd = io.loadProjectionData("tissue_fwd");
		Grid2D[] contrast_projections = io.loadProjectionData("vessel_contrast");
		// generate spectrum
		PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum(cfg.spectrum_binning_keV.firstElement(),cfg.spectrum_binning_keV.lastElement(),
				cfg.spectrum_sampling_keV,cfg.spectrum_peak_keV,cfg.spectrum_time_current_product_mAs);	
		double[] spectrum_flux = spectrum.getPhotonFlux();
		double[] spectrum_energies = spectrum.getPhotonEnergies();
		double detectorPixelArea = g_fwd.getPixelDimensionX()*g_fwd.getPixelDimensionY();

		// generate materials
		Material mat_braintissue = MaterialsDB.getMaterial("Brain");
		Material mat_skull =  MaterialsDB.getMaterial("Skull");
		Mixture  mix_ultravist = getUltravist370Mixture();
		
		// generate binned projection data for all sweeps
		for(int bin = 1; bin < cfg.spectrum_binning_keV.size(); bin++) {
			Grid2D[] binned_projections = new Grid2D[contrast_projections.length];
			Grid2D[] binned_projections_noisy = new Grid2D[contrast_projections.length];
			float bin_keV_start = cfg.spectrum_binning_keV.get(bin-1);
			float bin_keV_end = cfg.spectrum_binning_keV.get(bin);
			System.out.println("Processing bin " + bin + " ...");	
			
			for(int projection_id = 0; projection_id < binned_projections.length; projection_id++){
				System.out.println("Projection " + (projection_id+1) + " ... ");
				float[] p_skull = skull_fwd[projection_id].getBuffer();
				float[] p_tissue = tissue_fwd[projection_id].getBuffer();
				float[] p_contrast = contrast_projections[projection_id].getBuffer();
				
				Grid2D proj = new Grid2D(g_fwd.getDetectorWidth(), g_fwd.getDetectorHeight());
				float[] data = proj.getBuffer();
				Grid2D proj_noisy = null;
				float[] data_noisy = null;
				if(cfg.noise_add) {
					proj_noisy = new Grid2D(g_fwd.getDetectorWidth(), g_fwd.getDetectorHeight());
					data_noisy = proj_noisy.getBuffer();
				}
				for(int energy_idx = 0; energy_idx < spectrum_energies.length; energy_idx++)
				{
					System.out.println("\t Working on bin " + Integer.toString(energy_idx+1)+" of "+spectrum_energies.length);
					double energy = spectrum_energies[energy_idx];
					if(energy < bin_keV_start || ( (energy >= bin_keV_end) && (bin < cfg.spectrum_binning_keV.size()-1) )) continue;
					/**
					 * Get unattenuated photon flux acquired at one detector pixel.
					 * Flux in spectrum_flux is in photons/mm^2 and detectorPixelArea in mm^2
					*/
					double photon_flux = spectrum_flux[energy_idx]*detectorPixelArea;
					/**
					 * Get the weighting factors for the different materials at energy level in variable energy.
					 * The getAttenuation() function of the material returns the attenuation in cm^-1.
					 * The attenuation is normalized by the density to cm^2/g and then by division by 10 scaled to mm^2/mg.
					 * This corresponds to the units on the forward projected detector pixels.
					 * The input phantom volumes are scaled to mg/mm^3. Therefore, after computation of the line integral 
					 * through the object density, the unit in the projection data is mg/mm^2.
					 */
					double weighting_skull = (mat_skull.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mat_skull.getDensity())/10;
					double weighting_tissue = (mat_braintissue.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mat_braintissue.getDensity())/10;
					double weighting_contrast = (mix_ultravist.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mix_ultravist.getDensity())/10;					
					
					for(int i = 0; i < data.length; i++)
					{
						double attenuation = weighting_skull*p_skull[i] + weighting_tissue*p_tissue[i] + weighting_contrast*p_contrast[i];
						data[i] += (float) (photon_flux/spectrum.getTotalPhotonFlux()*Math.exp(-attenuation));
					}	
				}
				if(cfg.noise_add) {
					System.out.println("\t Adding Poissonian noise. This may take some time.");
					for(int i = 0; i < data.length; i++)
					{
						data_noisy[i] = poissonRandomProcess(data[i]);
					}								
				}
				binned_projections[projection_id] = proj;
				binned_projections_noisy[projection_id] = proj_noisy;
			}
			String fn_binned_proj_out = String.format("binned", bin);
			io.saveProjectionData(binned_projections,fn_binned_proj_out);
			if(cfg.noise_add) {
				String fn_binned_noisy_out = String.format("binned_noisy", bin);
				io.saveProjectionData(binned_projections_noisy,fn_binned_noisy_out);							
			}
		}
		
	}
	
	protected void generateMaskProjectionData() throws IOException
	{
		System.out.println();
		System.out.println("Create binned spectral mask projection data ...");
		// load input data
		System.out.println("Loading static projections ...");
		Grid2D[] skull_fwd = io.loadProjectionData("skull_fwd");
		Grid2D[] skull_bwd = io.loadProjectionData("skull_bwd");
		Grid2D[] tissue_fwd = io.loadProjectionData("tissue_fwd");
		Grid2D[] tissue_bwd = io.loadProjectionData("tissue_bwd");
		
		// generate spectrum
		PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum(cfg.spectrum_binning_keV.firstElement(),cfg.spectrum_binning_keV.lastElement(),
				cfg.spectrum_sampling_keV,cfg.spectrum_peak_keV,cfg.spectrum_time_current_product_mAs);	
		double[] spectrum_flux = spectrum.getPhotonFlux();
		double[] spectrum_energies = spectrum.getPhotonEnergies();
		
		
		// generate materials
		Material mat_braintissue = MaterialsDB.getMaterial("Brain");
		Material mat_skull =  MaterialsDB.getMaterial("Skull");
		
		
		for(int sweep = 0; sweep < 2; sweep++)
		{
			System.out.println("Processing sweep " + (sweep+1) + " ...");
			boolean fwd = (sweep%2 == 0);
			for(int bin = 1; bin < cfg.spectrum_binning_keV.size(); bin++) {
				Grid2D[] binned_projections = new Grid2D[skull_fwd.length];
				Grid2D[] binned_projections_noisy = new Grid2D[skull_fwd.length];
				float bin_keV_start = cfg.spectrum_binning_keV.get(bin-1);
				float bin_keV_end = cfg.spectrum_binning_keV.get(bin);
				System.out.println("Processing bin " + bin + " ...");	
				for(int projection_id = 0; projection_id < binned_projections.length; projection_id++)
				{
					System.out.println("Projection " + (projection_id+1) + " ... ");
					float[] p_skull = fwd?skull_fwd[projection_id].getBuffer():skull_bwd[projection_id].getBuffer();
					float[] p_tissue = fwd?tissue_fwd[projection_id].getBuffer():tissue_bwd[projection_id].getBuffer();;
					
					Grid2D proj = new Grid2D(g_fwd.getDetectorWidth(), g_fwd.getDetectorHeight());
					float[] data = proj.getBuffer();
					Grid2D proj_noisy = null;
					float[] data_noisy = null;
					if(cfg.noise_add) {
						proj_noisy = new Grid2D(g_fwd.getDetectorWidth(), g_fwd.getDetectorHeight());
						data_noisy = proj_noisy.getBuffer();
					}
					for(int energy_idx = 0; energy_idx < spectrum_energies.length; energy_idx++)
					{
						double energy = spectrum_energies[energy_idx];
						if(energy < bin_keV_start || ( (energy >= bin_keV_end) && (bin < cfg.spectrum_binning_keV.size()-1) )) continue;
						double photon_flux = spectrum_flux[energy_idx];	
						
						double weighting_skull = (mat_skull.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mat_skull.getDensity())/10;
						double weighting_tissue = (mat_braintissue.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mat_braintissue.getDensity())/10;
						
						for(int i = 0; i < data.length; i++)
						{
							double attenuation = weighting_skull*p_skull[i] + weighting_tissue*p_tissue[i];
							data[i] += (float) (photon_flux*Math.exp(-attenuation));
						}	
					}
					if(cfg.noise_add) {
						for(int i = 0; i < data.length; i++)
						{
							data_noisy[i] = poissonRandomProcess(data[i]);
						}								
					}
					binned_projections[projection_id] = proj;
					binned_projections_noisy[projection_id] = proj_noisy;
				}
				String fn_binned_proj_out = String.format("sweep%03d_bin%02d_mask", sweep, bin);
				io.saveProjectionData(binned_projections,fn_binned_proj_out);
				if(cfg.noise_add) {
					String fn_binned_noisy_out = String.format("sweep%03d_bin%02d_mask_noisy", sweep, bin);
					io.saveProjectionData(binned_projections_noisy,fn_binned_noisy_out);							
				}
			}
		}
	}
	
	public static Mixture getUltravist370Mixture()
	{
		// Ultravist 370 contrast agent
		String formularIopromideString = "C18H24I3N3O8"; // from http://en.wikipedia.org/wiki/Iopromide
		double densityInWater = 0.529816; // @37 deg celsius
		// from http://www.rxlist.com/ultravist-drug.htm
		double gramsOfIopromide = 0.76886;
		double densityAt37DegCelsius = 1.399;
		// This would be plain iopromide:
		WeightedAtomicComposition wacIopromide = new WeightedAtomicComposition(formularIopromideString);
		double molarMassIopromide = MaterialUtils.computeMolarMass(wacIopromide);
		MaterialUtils.newMaterial("Iopromide",densityInWater,wacIopromide);	
		WeightedAtomicComposition water = new WeightedAtomicComposition("H2O");
		double molarMassWater = MaterialUtils.computeMolarMass(water);
		double waterParticlesIn1Gram = 1 / molarMassWater;
		WeightedAtomicComposition wacUltravist = new WeightedAtomicComposition("H2O", waterParticlesIn1Gram);
		double iopromideParticles = gramsOfIopromide / molarMassIopromide;
		wacUltravist.add(formularIopromideString, iopromideParticles);
		//Material ultravist = MaterialUtils.newMaterial(,densityAt37DegCelsius[i],wacUltravist);
		Mixture iopromideSolution = new Mixture();
		iopromideSolution.setDensity(densityAt37DegCelsius);
		iopromideSolution.setName("Ultravist 370");
		iopromideSolution.setWeightedAtomicComposition(wacUltravist);		
		
		return iopromideSolution;
	}
	
	protected int poissonRandomProcess(double lambda)
	{
		int res = StatisticsUtil.poissonRandomNumber(lambda);
		if(res < 0) res = 0;
		return res;
	}
	
	protected void setVolumeToProject(float[] volumeBuffer)
	{
		opencl_fwdproj.setVolume(volumeBuffer);
	}
	
	protected Grid2D applyForwardProjection(int projection_id, boolean fwd)
	{	
		Trajectory g = fwd?g_fwd:g_bwd;
		Grid2D projection = new Grid2D(g.getDetectorWidth(),g.getDetectorHeight());
		opencl_fwdproj.applyForwardProjection(projection_id, fwd, projection.getBuffer(), false);
		return projection;	
	}
	
	protected float[] getDynamicContrastVolume(float time) throws IOException
	{
		float[] volume = new float[PerfusionBrainPhantomConfig.SZ];

		int prev_idx = (int)(time/cfg.phantom_sampling)+1;
		float prev_time = (float)((prev_idx-1)*cfg.phantom_sampling);
		int next_idx = (int)(time/cfg.phantom_sampling)+2;
		float next_time = (float)((next_idx-1)*cfg.phantom_sampling);
		
		if(prev_idx != vol_contrast_prev_idx) {
			vol_contrast_prev = io.loadVolume(Integer.toString(prev_idx));
			vol_contrast_prev_idx = prev_idx;
		}
		if(next_idx != vol_contrast_next_idx) {
			vol_contrast_next = io.loadVolume(Integer.toString(next_idx));
			vol_contrast_next_idx = next_idx;
		}
		 
		float weight = (float)(time-prev_time)/(next_time-prev_time);

		for(int i = 0; i < PerfusionBrainPhantomConfig.SZ; i++) {
			volume[i] = (1-weight)*vol_contrast_prev[i] + weight*vol_contrast_next[i];

		}	
		return volume;
	}
	

	public void createSpectralCalibrationPhantom() throws IOException
	{
		int[] volumeSize = new int[3];
		float[] voxelSize = new float[3];
		volumeSize[0] = PerfusionBrainPhantomConfig.SZ_X;volumeSize[1] = PerfusionBrainPhantomConfig.SZ_Y;volumeSize[2] = PerfusionBrainPhantomConfig.SZ_Z;  
		voxelSize[0] = PerfusionBrainPhantomConfig.SZ_VOXEL_MM;voxelSize[1] = PerfusionBrainPhantomConfig.SZ_VOXEL_MM;voxelSize[2] = PerfusionBrainPhantomConfig.SZ_VOXEL_MM;
		Material mat_water = MaterialsDB.getMaterial("Water");
		Material mat_bone = MaterialsDB.getMaterial("Bone");
		float[] vol_water = new float[PerfusionBrainPhantomConfig.SZ];
		float[] vol_bone  = new float[PerfusionBrainPhantomConfig.SZ];
		int idx = 0;
		double x_mid = volumeSize[0]/2.0;
		double y_mid = volumeSize[1]/2.0;
		for(int z = 0; z < volumeSize[2]; z++) {
			for(int y = 0; y < volumeSize[1]; y++) {
				for(int x = 0; x < volumeSize[0]; x++) {
					double r = Math.sqrt( (x-x_mid)*(x-x_mid)+(y-y_mid)*(y-y_mid));
					if(r <= 64) {
						if(x < volumeSize[0]/2)
							vol_water[idx] = (float)mat_water.getDensity();
						else
							vol_bone[idx]  = (float)mat_bone.getDensity();
					}
					idx++;
				}			
			}
		}
			
		io.saveVolume(vol_water, "water_cylinder");
		io.saveVolume(vol_skull, "skull_cylinder");
		opencl_fwdproj = new OpenCLForwardProjectorDynamicVolume();
		opencl_fwdproj.configure(g_fwd, g_bwd, volumeSize, voxelSize);
		
		Grid2D[] projection = new Grid2D[2];

		setVolumeToProject(vol_water);
		projection[0] = applyForwardProjection(0, true);
		setVolumeToProject(vol_bone);
		projection[1] = applyForwardProjection(0, true);
		
		Grid2D[] poly_projection = new Grid2D[2];
		poly_projection[0] = new Grid2D(projection[0].getWidth(),projection[0].getHeight());
		poly_projection[1] = new Grid2D(projection[0].getWidth(),projection[0].getHeight());
		// generate spectrum
		PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum(cfg.spectrum_binning_keV.firstElement(),cfg.spectrum_binning_keV.lastElement(),
					cfg.spectrum_sampling_keV,cfg.spectrum_peak_keV,cfg.spectrum_time_current_product_mAs);	
		double[] spectrum_flux = spectrum.getPhotonFlux();
		double[] spectrum_energies = spectrum.getPhotonEnergies();
		double detectorPixelArea = 1;//g_fwd.getPixelDimensionX()*g_fwd.getPixelDimensionY();
		
		/*for(int i = 0; i < spectrum_flux.length; i++)
			System.out.println(spectrum_energies[i] + "\t" + spectrum_flux[i]);*/
		
		float[] data_water = poly_projection[0].getBuffer();
		float[] p_water = projection[0].getBuffer();
		float[] data_bone = poly_projection[1].getBuffer();
		float[] p_bone = projection[1].getBuffer();

		for(int energy_idx = 0; energy_idx < spectrum_energies.length; energy_idx++)
		{
			double energy = spectrum_energies[energy_idx];
			double photon_flux = spectrum_flux[energy_idx]*detectorPixelArea;
			double weighting_water = (mat_water.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mat_water.getDensity())/10;
			double weighting_bone = (mat_bone.getAttenuation(energy, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION)/mat_bone.getDensity())/10;
			for(int i = 0; i < data_water.length; i++)
			{
				double attenuation = weighting_water*p_water[i];
				data_water[i] += (float) (photon_flux*Math.exp(-attenuation));
				attenuation = weighting_bone*p_bone[i];
				data_bone[i] += (float) (photon_flux*Math.exp(-attenuation));
			}	
		}
		
		io.saveProjectionData(projection, "cylinder_projection_ref");
		io.saveProjectionData(poly_projection, "cylinder_projection");
	}
	
	
}