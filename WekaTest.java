import java.io.File;
import java.text.DecimalFormat;
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.trees.LMT;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
 
public class WekaTest {
	
	public static void main(String[] args) throws Exception {
		
		String finalPrediction = null;
		float easy = 0;
		float medium = 0;
		float hard = 0;
		int counter = 1;
		ArffLoader loader = new ArffLoader();
	    loader.setFile(new File("prediction.txt"));
	    Instances trainingSet = loader.getDataSet();
	    int classIdx = 4;
	        
	    trainingSet.setClassIndex(classIdx);
	        
	     // using the LMT classification algorithm. Many more are available   
	     Classifier classifier = new LMT();
	     classifier.buildClassifier(trainingSet);
	     
	     int countRows=0;
	     
		 DecimalFormat df = new DecimalFormat("#.##");
	     
		 for (Enumeration<Instance> en = trainingSet.enumerateInstances(); en.hasMoreElements();) {
	    	 ++countRows;
	    	 double[] results = classifier.distributionForInstance(en.nextElement());
	         for (double result : results) {
	        	if(countRows == results.length) {
	        	   if(counter == 1) {
	        		   easy = Float.parseFloat(df.format(result));
	        	   }
	        	   if(counter == 2) {
	        		   medium = Float.parseFloat(df.format(result));
	        	   }
	        	   if(counter == 3) {
	        		   hard = Float.parseFloat(df.format(result));
	        	   }
	        	   
	        	   counter++;
	           }
	        }
	         
	     };
	     if(easy > medium && easy > hard) {
	    	 finalPrediction = "Easy";
	     }
	     if(medium > easy && medium > hard) {
	    	 finalPrediction = "Medium";
	     }
	     if(hard > easy && hard > medium) {
	    	 finalPrediction = "Hard";
	     }
	     
	     System.out.println("Final prediction : " + finalPrediction);
	  
	}
	
}
