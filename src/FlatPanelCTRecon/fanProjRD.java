package FlatPanelCTRecon;

public class fanProjRD {
	int projectionNumber;
    double detectorSpacing;
    int detectorPixel;
    double D;
    
    public fanProjRD(int projectionNumber, double detectorSpacing, int detectorPixel, double sourceIsoDistance)
    {
        this.projectionNumber = projectionNumber;	//Number of projections
        this.detectorSpacing = detectorSpacing;		//Delta s
        this.detectorPixel = detectorPixel;			//Number of pixels in the detector.
        this.D = sourceIsoDistance;					//Distance from the origin to the x-ray source
    }
}
