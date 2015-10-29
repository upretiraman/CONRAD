package FlatPanelCTRecon;
 
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
 
public class myParallelBackProjection
{
     
    int angleIndex;
    double Dtheta, maxTheta;
     
    int sIndex;
    double detectorSpacing, detectorLength;
     
    int projectionNumber;
    double sourceIsoDistance;
     
    Grid2D sinogram;
     
    public myParallelBackProjection (int projectionNumber, double detectorSpacing, int detectorLength)
    {
        this.projectionNumber = projectionNumber;
        this.detectorSpacing = detectorSpacing;
        this.detectorLength = detectorLength;
    }
     
    public void setSinogramParameters(Grid2D sino)
    {
        this.sinogram = sino;
        this.angleIndex = sino.getSize()[1];
        this.Dtheta = sino.getSpacing()[1];
        this.maxTheta = (angleIndex -1) * Dtheta;
 
        this.sIndex = sino.getSize()[0];
        this.detectorSpacing = sino.getSpacing()[0];
        this.detectorLength = (sIndex-1)*detectorSpacing;
    }
    /*!
     * Parallel backprojection Ray Driven.
     */
    public Grid2D backProjPD(Grid2D sino, int[] size , double[] spacing)
    {
        this.setSinogramParameters(sino);
        Grid2D grid = new Grid2D(size[0], size[1]);
        grid.setSpacing(spacing[0], spacing[1]);
        grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);
         
        double offsetX  =  (grid.getSize()[0] * grid.getSpacing()[0])/2;
        double offsetY =  (grid.getSize()[1] * grid.getSpacing()[1])/2;
         
        double x,y,s;
        float pixelFromSino = 0;
        double detIndex;
         
        for(int i = 0; i <= grid.getSize()[0]; ++i)
        {
            x = i*this.detectorSpacing - offsetX;
             
            for(int j = 0; j <= grid.getSize()[1]; ++j)
            {
                y = -j*this.detectorSpacing + offsetY;
                float sum = 0;
                for(int a = 0;a< projectionNumber;++a)
                {
                    double projAngle = Dtheta*a;
                    double x_det = Math.cos(projAngle),y_det = Math.sin(projAngle);
                    double projSlope,detSlope,dtX,dtY;
                    double yIntercept;
                    float weight = 1;                  
                    if(projAngle == 0 || projAngle == 90 || projAngle == 180)
                    {
                        if(projAngle == 90)
                            s = y;
                        else{
                            s = x;
                            weight = (float) 0.5;                      
                        }
                    }
                    else{
                        //Line intersections for detecting the corresponding detector pixel for the current image pixel.                   
                        projSlope = Math.tan(projAngle+(Math.PI/2));    //Projecting orthogonal line w.r.t. the detector.                  
                        detSlope = Math.tan(projAngle);                 //Detector slope
                        yIntercept = y - x*projSlope;                   //y-axis intersection of the projected line.
                        dtX = yIntercept/(detSlope-projSlope);          //X intersection
                        dtY = detSlope*dtX;                             //Y intersection
                        double detDir = Math.sqrt(Math.pow(x_det,2)+Math.pow(y_det,2)); //Detector direction.
                        double magn_dPX = Math.sqrt(Math.pow(dtX,2)+Math.pow(dtY,2));   //Pixel vector.
                        double cosAlpha = (dtX*x_det + dtY*y_det)/(magn_dPX*detDir);    //Angle between these two vectors. (0° or 180°)
                        s = magn_dPX * Math.signum(cosAlpha);
                    }
                     
                    detIndex = (s + (detectorLength + 1)/2)/detectorSpacing;    //Detector index computing.                
                    Grid1D subgrid = sino.getSubGrid(a);                        //Row from the sinogram
                    pixelFromSino = InterpolationOperators.interpolateLinear(subgrid, detIndex);
                    sum += pixelFromSino * weight;                              //Interpolated value from the sinogram and weighted for redundancy.
                }
                grid.setAtIndex(i, j, sum);
            }
        }
        return grid;
 
    }
    
    public Grid2D filterSino(Grid2D sino)
    {
        this.setSinogramParameters(sino);
        Grid1D row = new Grid1D(sino.getSubGrid(0));
        Grid2D filteredSino = new Grid2D(sIndex, angleIndex);
        filteredSino.setSpacing(sino.getSpacing()[0], sino.getSpacing()[1]);           
        //Allocating memory for the RamLak Filter
        Grid1DComplex rampFilter = new Grid1DComplex(this.sIndex);
        int filterSize = rampFilter.getSize()[0];
        //Setting the frequency spacing for the filter
        rampFilter.setSpacing((1.0/(detectorSpacing*filterSize)));
        double aux = -1.0/(Math.PI * Math.PI * detectorSpacing);
        for (int p = -(filterSize/2); p < (filterSize/2); ++p)
        {
            if(1==(Math.abs(p)%2)){
                rampFilter.setAtIndex(p+512, (float) (aux/(Math.pow(p, 2))));}
            if (p==0){
                rampFilter.setAtIndex(p+512, (float) (1.0/4.0));}
        }
        rampFilter.transformForward();
        //VisualizationUtil.createPlot(rampFilter.getSubGrid(0, 1024).getBuffer()).show();
         
        for (int i = 0;i<angleIndex;++i){
            Grid1DComplex Crow = new Grid1DComplex(sino.getSubGrid(i)); //Complex row
            //Setting the frequency spacing for the padded sinogram row.
            Crow.transformForward();
            //Crow.setSpacing((1.0/(detectorSpacing*Crow.getSize()[0])));
            //Multiplying row-wise the row from the sinogram and the filter in the frequency domain.
            for(int p = 0;p<filterSize;++p){
                Crow.multiplyAtIndex(p, rampFilter.getRealAtIndex(p), rampFilter.getImagAtIndex(p));                   
            }
            Crow.transformInverse();
            row = Crow.getRealSubGrid((filterSize/2)-1, row.getSize()[0]);
            if(i == angleIndex/2)
                VisualizationUtil.createPlot(Crow.getSubGrid(0, 1024).getBuffer()).show();
            for(int p = 0;p<sIndex;++p){
                filteredSino.setAtIndex(p, i, row.getAtIndex(p));
            }
        }
         
        filteredSino.show("Ramp Filtered Sinogram");
        return filteredSino;
    }
     
    public Grid2D filteredBP(Grid2D sino, int[] size , double[] spacing)
    {
        return this.backProjPD (this.filterSino(sino), size, spacing);
    }
}
