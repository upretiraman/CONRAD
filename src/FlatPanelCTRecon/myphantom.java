package FlatPanelCTRecon;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;



public class myphantom extends Grid2D{
    
    public myphantom(int x,int y){
        super(x, y);
        int val1 = 1;
        int val2 = 2;
        int val3 = 3;
        double r1 = 0.45*x;
        double r2 = 0.5*r1;
        
        
        
        int xCenter1 = x/2;
        int yCenter1 = y/2;
        
        int xCenter2 = (int) ((0.5*x)-r2);
        int yCenter2 = yCenter1;
        
        int xr1 = xCenter1;
        int xr2 = xr1 + (int) (0.707*r1);
        int yr1 = yCenter1 - (int) (0.5*r1);
        int yr2 = yCenter1 + (int) (0.5*r1);
        
        this.setSpacing(1,1);
		
        
        
        for(int i = 0; i < x; i++){
            for(int j = 0; j < y; j++) {
                if( Math.pow(i - xCenter1, 2)  + Math.pow(j - yCenter1, 2) <= (r1*r1) ){
                    super.setAtIndex(i, j, val1);
                }
                
                if( Math.pow(i - xCenter2, 2)  + Math.pow(j - yCenter2, 2) <= (r2*r2) ){
                    super.setAtIndex(i, j, val2);
                }
                
                if((i > xr1 && i < xr2) && (j > yr1 && j < yr2)){
                    super.setAtIndex(i, j, val3);
                }
            }
        }
    }
    
        public static void main (String [] args)
        {
            myphantom phant = new myphantom(200,200);
            phant.show("MyPhantom");
            
            //Sinogram generation 
            myParallelProject projector = new myParallelProject(300,1,400);
    		Grid2D sinogram = projector.projectRayDriven(phant);
    		sinogram.show("The Sinogram");
    		
    		myParallelBackProjection backProj = new myParallelBackProjection(300,1,400);
    		(backProj.backProjPD(sinogram, phant.getSize(), phant.getSpacing())).show("Back Projected");
    		(backProj.filteredBP(sinogram, phant.getSize(), phant.getSpacing())).show("Filtered Back Projected");
    		
    		//myFanProjRD fan_proj = new myFanProjRD(300,1,400,300,150);
            //Grid2D fanogram = fan_proj.fanProjectRD(phant, true, 3);
            //fanogram.show("The Fanogram");
            //sinogram = fan_proj.rebinFanogram(fanogram, true);
            //sinogram.show("Rebinned Fanogram");
            //myParallelBackProjection fanBackProj = new myParallelBackProjection(300,1,400);
            //(fanBackProj.backProjPD(sinogram, phant.getSize(), phant.getSpacing())).show("Back Projected shit");
        }
}


