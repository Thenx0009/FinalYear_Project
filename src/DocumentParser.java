
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.TreeSet;

public class DocumentParser {

    //This variable will hold all terms of each document in an array.
    private List<String[]> termsDocsArray = new ArrayList<String[]>();
    private List<String> allTerms = new ArrayList<String>(); //to hold all terms
    private List<double[]> tfidfDocsVector = new ArrayList<double[]>();

    public void parseFiles(String filePath) throws FileNotFoundException, IOException {
        File[] allfiles = new File(filePath).listFiles();
        BufferedReader in = null;	
        for (File f : allfiles) {
            if (f.getName().endsWith("")) {
                in = new BufferedReader(new FileReader(f));
                StringBuilder sb = new StringBuilder();
                String s = null;
                while ((s = in.readLine()) != null) {
                    sb.append(s);
                }
                String[] tokenizedTerms = sb.toString().replaceAll("[\\W&&[^\\s]]", "").split("\\W+");   //to get individual terms
                for (String term : tokenizedTerms) {
                    if (!allTerms.contains(term)) {  //avoid duplicate entry
                        allTerms.add(term);
                    }
                }
                termsDocsArray.add(tokenizedTerms);
            }
        }
        System.out.println("files read-Done!!!");
    }

    /**
     * Method to create termVector according to its tfidf score.
     */
    public void tfIdfCalculator() {
        double tf; //term frequency
        double idf; //inverse document frequency
        double tfidf; //term requency inverse document frequency
        int count1=termsDocsArray.size();
		System.out.println(count1);
        for (String[] docTermsArray : termsDocsArray) {
        	count1--;
        	System.out.println("count:"+count1);
            double[] tfidfvectors = new double[allTerms.size()];
            int count = 0;
            for (String terms : allTerms) {
                tf = new TfIdf().tfCalculator(docTermsArray, terms);
                idf = new TfIdf().idfCalculator(termsDocsArray, terms);
                tfidf = tf * idf;
                tfidfvectors[count] = tfidf;
                count++;
            }
            tfidfDocsVector.add(tfidfvectors);  //storing document vectors;   
        }
        System.out.println("tfidf-Done!!!");
    }

    /**
     * Method to calculate cosine similarity between all the documents.
     * @throws IOException 
     */
    public void runkmeans() throws IOException{
    	HashMap<double[],TreeSet<Integer>> clusters = new HashMap<double[],TreeSet<Integer>>();
	    HashMap<double[],TreeSet<Integer>> step = new HashMap<double[],TreeSet<Integer>>();
	    HashSet<Integer> rand = new HashSet<Integer>();
	    int k = 3;
	    int maxiter = 500;
	    for(int init=0;init<50;init++){
			
	    	clusters.clear();
	    	step.clear();
	    	rand.clear();
	    	//randomly initialize cluster centers
		    while(rand.size()< k){
		    	rand.add((int)(Math.random()*tfidfDocsVector.size()));
		    	//System.out.println("tfidf-Done!!!"+rand.size());
		    }
		    for(int r:rand){
		    	double[] temp = new double[tfidfDocsVector.get(r).length];
		    	System.arraycopy(tfidfDocsVector.get(r),0,temp,0,temp.length);
		    	step.put(temp,new TreeSet<Integer>());
		    }
		    boolean go = true;
		    int iter = 0;
	    	while(go){
		    	clusters = new HashMap<double[],TreeSet<Integer>>(step);
		    	//cluster assignment step
		    	for(int i=0;i<tfidfDocsVector.size();i++){
			    	double[] cent = null;
			    	double sim = 0;
		    		for(double[] c:clusters.keySet()){
		    			double csim = new CosineSimilarity().cosineSimilarity(tfidfDocsVector.get(i),c);
		    			if(csim > sim){
		    				sim = csim;
		    				cent = c;
		    			}
		    		}
		    		clusters.get(cent).add(i);
					for (double[] d : clusters.keySet()) {
						for (double e : d) {
							System.out.println(e);
						}
					}
					
		    	}
		    	//centroid update step
		    	step.clear();
		    	for(double[] cent:clusters.keySet()){
		    		double[] updatec = new double[cent.length];
		    		for(int d:clusters.get(cent)){
		    			double[] doc = tfidfDocsVector.get(d);
		    			for(int i=0;i<updatec.length;i++)
		    				updatec[i]+=doc[i];
		    		}
		    		for(int i=0;i<updatec.length;i++)
		    			updatec[i]/=clusters.get(cent).size();
		    		step.put(updatec,new TreeSet<Integer>());
		    	}
		    	//check break conditions
		    	String oldcent="", newcent="";
		    	for(double[] x:clusters.keySet())
		    		oldcent+=Arrays.toString(x);
		    	for(double[] x:step.keySet())
		    		newcent+=Arrays.toString(x);
		    	if(oldcent.equals(newcent)) go = false;
		    	if(++iter >= maxiter) go = false;
		    }
	    }
	    System.out.println("clusters:"+clusters.toString().replaceAll("\\[[\\w@]+=",""));
	    double purity=0;
	    double entropy=0;
	    for(double[] cent:clusters.keySet()){
	    	int x=0,y=0,z=0;
    		for(int d:clusters.get(cent)){
    			if(d>=0 && d<=2)
    				x++;
    			else if(d>2	&& d<=5)
    				y++;
    			else if(d>5 && d<=7)
    				z++;
    		}
    		List<Integer> data = new ArrayList<Integer>();
    		data.add(x);data.add(y);data.add(z);
    		Collections.sort(data);
    		purity+=(double)data.get(2)/7;
    		int sum=x+y+z;
    		double midx=0,midy=0,midz=0;
    		if(x!=0){
    			double n=(double)x/sum;
    			double l=Math.log(n);
    			midx=n*l;
    		}
    		if(y!=0){
    			double n=(double)y/sum;
    			double l=Math.log(n);
    			midx=n*l;
    		}
    		if(z!=0){
    			double n=(double)z/sum;
    			double l=Math.log(n);
    			midx=n*l;
    		}
    		entropy+=(((double)(-1/Math.log(3)))*(midx+midy+midz))*((double)(x+y+z)/7);
    	}
	    System.out.println("purity:"+purity+"\nEntropy:"+entropy);
    }
}