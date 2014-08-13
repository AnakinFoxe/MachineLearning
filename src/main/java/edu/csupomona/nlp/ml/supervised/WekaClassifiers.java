package edu.csupomona.nlp.ml.supervised;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import edu.csupomona.nlp.util.Stopword;


public class WekaClassifiers {
	
	private Classifier oModel;
	private Classifier cModel;

	private Instances otrainingSet;
	private Instances trainingSet;
	private Instances otestingSet;
	private Instances testingSet;
	
	private List<String> unigramList;
	private List<String> aspectWordList;
	
	private int[] counter;
	
	private int[] otrain_counter;
	private int[] train_counter;
        
        Stopword sw;
	
	public WekaClassifiers(Integer c) {
		// choose the type of classifier
		switch (c) {
		case 1:
			oModel = (Classifier)new NaiveBayes();
			cModel = (Classifier)new NaiveBayes();
			break;
		case 2: 
			oModel = (Classifier)new RandomForest();
			cModel = (Classifier)new RandomForest();
			break;
		case 3:
			oModel = (Classifier)new SMO();
			cModel = (Classifier)new SMO();
//			try {
//				oModel.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
//				cModel.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
//			} catch (Exception e1) {
//				// TODO Auto-generated catch block
//				e1.printStackTrace();
//			}
			break;
		case 4:
			oModel = (Classifier)new NaiveBayesMultinomial();
			cModel = (Classifier)new NaiveBayesMultinomial();
		default:
			break;
		}
			
		// list contains all unigram
		unigramList = new ArrayList<>();
		
		// list contains all aspect words
		aspectWordList = new ArrayList<>();
		aspectWordList.add("battery");
		aspectWordList.add("camera");
		aspectWordList.add("screen");
		
		// Stopwords init
		sw = new Stopword("en");
		
		// counter init
		counter = new int[4];
		otrain_counter = new int[2];
		train_counter = new int[3];
		
		
		try {
			// create unigram list 
			FileReader fr = new FileReader("bigrams.csv");
			BufferedReader br = new BufferedReader(fr);
			String line;
			Set<String> uniqueList = new HashSet<>();
			while ((line = br.readLine()) != null) {
				String[] lineInfo = line.split(",");
				if (lineInfo[0].equals("local-unigram")) {
					uniqueList.add(lineInfo[1]);
				}
			}
			br.close();
			for (String word : uniqueList)
				unigramList.add(word);
			
			
			// create attribute list using unigram list
			FastVector oattList = new FastVector();
			FastVector attList = new FastVector();
			for (String unigram : unigramList) {
				oattList.addElement(new Attribute(unigram));
				attList.addElement(new Attribute(unigram));
			}
			
			// create a/o label list for 1st classifier
			FastVector oList = new FastVector();
			oList.addElement("all");
			oList.addElement("other");
			oattList.addElement(new Attribute("oLabel", oList));
			
			// create aspect label list
			FastVector aspectList = new FastVector();	
			aspectList.addElement("battery");
			aspectList.addElement("camera");
			aspectList.addElement("screen");
			attList.addElement(new Attribute("AspectLabel", aspectList));
			
			// create training set instances for 1st classifier
			otrainingSet = new Instances("OTRAINING_SET", oattList, 0); 
			otrainingSet.setClassIndex(unigramList.size());
			
			// create training set instances
			trainingSet = new Instances("TRAINING_SET", attList, 0); 
			trainingSet.setClassIndex(unigramList.size());
			
			// create testing set instances for 1st classifier
			otestingSet = new Instances("OTESTING_SET", oattList, 0); 
			otestingSet.setClassIndex(unigramList.size());
			
			// create testing set instances (same attribute list)
			testingSet = new Instances("TESTING_SET", attList, 0); 
			testingSet.setClassIndex(unigramList.size());
			
			System.out.println("Initialization Finished");
			
		} catch (IOException e) {

		}
		
	}
	
	private Instance createInstance(Map<Integer, Double> featureMap, String aspect, Integer time) {
		int indices[] = new int[featureMap.size()+1];
        double values[] = new double[featureMap.size()+1];
        int i=0;
        for(Map.Entry<Integer,Double> entry : featureMap.entrySet())
        {
            indices[i] = entry.getKey();
            values[i] = entry.getValue();
            i++;
        }
        indices[i] = unigramList.size();
        if (time == 1) {
	        if (aspectWordList.contains(aspect)) {
	        	values[i] = 0.0;
//	        	System.out.println("1stC all: " + aspect);
	        	otrain_counter[0]++;
	        }
	        else {
	        	values[i] = 1.0;
//	        	System.out.println("1stC other: " + aspect);
	        	otrain_counter[1]++;
	        }
        } else if (time == 2) {
        	if (aspectWordList.contains(aspect)) {
	        	values[i] = (double)aspectWordList.indexOf(aspect);
//	        	System.out.println("2nd aspect: " + aspect);
	        	train_counter[(int)values[i]]++;
        	}
        	else {
//        		System.out.println("2nd return: " + aspect);
        		return null;
        	}
        }
        
        return new SparseInstance(1.0, values, indices, featureMap.size());
	}
	
	private void extractFeature(String[] words, Integer time) {
		int window = 3;
		List<String> sentence = array2List(words);
		int[] sentenceMarker = new int[words.length];
//		String aspect = "other";
		
		// search aspect words in the sentences
		for(int i = 0; i < aspectWordList.size(); i++){
			if(sentence.contains(aspectWordList.get(i))){
				Map<Integer, Double> featureMap = new TreeMap<>();
				int pos = sentence.indexOf(aspectWordList.get(i));
				int begin = ((pos - window) > 0 ? pos - window : 0);
				int end = ((pos + window) < sentence.size() ? pos + window : sentence.size()-1);
				
				// the feature word is present
				for (int idx = begin; idx <= end; ++idx) {
					if (unigramList.contains(words[idx])) 
						featureMap.put(unigramList.indexOf(words[idx]), 1.0);
					
					sentenceMarker[idx]++;
				}
					
				String aspect = aspectWordList.get(i);
				
				// create instance for each finded aspect
				Instance featureVector = createInstance(featureMap, aspect, time);
				if (time == 1) {
					featureVector.setDataset(otrainingSet);
					otrainingSet.add(featureVector);
				} else if (time == 2) {
					if (featureVector != null) {
						featureVector.setDataset(trainingSet);
						trainingSet.add(featureVector);
					}
				}
				
			}
		}
		
		
		// for sentences or part of sentences not concerning with aspect words
		if (time == 1) {
			Map<Integer, Double> featureMap = new TreeMap<>();
			for(int i = 0; i < words.length; i++){
				if (sentenceMarker[i] == 0) {
					if (unigramList.contains(words[i])) 
						featureMap.put(unigramList.indexOf(words[i]), 1.0);
				}
			}
			if (featureMap.size() > 0) {
				String aspect = "other";
				Instance featureVector = createInstance(featureMap, aspect, time);
//				if (time == 1) {
					featureVector.setDataset(otrainingSet);
					otrainingSet.add(featureVector);
//				} else if (time == 2) {
//					if (featureVector != null) {
//						featureVector.setDataset(trainingSet);
//						trainingSet.add(featureVector);
//					}
//				}
			}
		}
	}
	
	private void processLine(String line, Integer time) {
		BreakIterator iterator = BreakIterator.getSentenceInstance(Locale.US);
		iterator.setText(line);
		int start = iterator.first();
		//loop through each sentence
		for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
			String sentence = line.substring(start,end);
			String adjustedSentence = sentence.replaceAll("( +: ?| +\\*+ ?)|[\\[\\] \\(\\)\\.,;!\\?\\+-]", " ");
			adjustedSentence = adjustedSentence.toLowerCase();	// to lower case
			String words[] = adjustedSentence.split(" +");
			words = sw.rmStopword(words);	// remove stopwords at here
			if(words.length > 0){
				extractFeature(words, time);
			}
		}
	}
	
	public void loadTrainingData(Integer time) {
		String folder = "data/";
		File[] files = new File(folder).listFiles();
		try {
			for (File file : files) {
				FileReader fr = new FileReader(folder + file.getName());
				BufferedReader br = new BufferedReader(fr);
				String line;
				while ((line = br.readLine()) != null) {
					processLine(line, time);
				}
				br.close();
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Done Loading Training Set");
		
	}
	
	public void loadTestingData() {
		Map<Integer, Double> featureMap = new TreeMap<>();
		try {
			FileReader fr = new FileReader("nokia-lumia521-tmobile_tagged.txt");
			BufferedReader br = new BufferedReader(fr);
			String line;
			Pattern pattern = Pattern.compile("\\[([\\w]+)\\]:(.*)");
			while ((line = br.readLine()) != null) {
				Matcher matcher = pattern.matcher(line);
				if (matcher.find()) {
					String label = matcher.group(1);
					String sentence = matcher.group(2);
					String adjustedSentence = sentence.replaceAll("( +: ?| +\\*+ ?)|[\\[\\] \\(\\)\\.,;!\\?\\+-]", " ");
					adjustedSentence = adjustedSentence.toLowerCase();	// to lower case
					String words[] = adjustedSentence.split(" +");
					words = sw.rmStopword(words);	// remove stopwords at here
					
					for (String word : words) {
						if (unigramList.contains(word)) 
							featureMap.put(unigramList.indexOf(word), 1.0);
					}
					
					int indices[] = new int[featureMap.size()+1];
			        double values[] = new double[featureMap.size()+1];
			        int i=0;
			        for(Map.Entry<Integer,Double> entry : featureMap.entrySet())
			        {
			            indices[i] = entry.getKey();
			            values[i] = entry.getValue();
			            i++;
			        }
			        indices[i] = unigramList.size();
			        if (aspectWordList.contains(label))
			        	values[i] = 0.0;
			        else
			        	values[i] = 1.0;
			        
			        
//			        Instance featureVector = new SparseInstance(1.0, values, indices, featureMap.size());
//			        
//			        featureVector.setDataset(otestingSet);
//			        otestingSet.add(featureVector);
//			        
//			        double score = oModel.classifyInstance(featureVector);
//			        System.out.println("1st: " + label + ", (expect: " + values[i] + ")=>(result: " + score + "): " + sentence);
			        
			        // for sentences classified into "all"
			        //if (0.0 == score) {
			        if (!label.equals("other")){
			        	if (aspectWordList.contains(label)) {
			        		values[i] = (double)aspectWordList.indexOf(label);
			        	} else
			        		values[i] = aspectWordList.size();
			        	
			        	Instance featureVector = new SparseInstance(1.0, values, indices, featureMap.size());
			        	
			        	featureVector.setDataset(testingSet);
				        testingSet.add(featureVector);
				        
				        double score = cModel.classifyInstance(featureVector);
			        	
			        	if (values[i] == score)
			        		counter[(int)score]++;
//			        	System.out.println("2nd: " + label + ", (expect: " + values[i] + ")=>(result: " + score + "): " + sentence);
			        } else {
			        	if (label.equals("other"))
			        		counter[counter.length-1]++;
//			        	System.out.println("1st: " + label + " : " + values[i] + " : " + score + " : " + sentence);
			        }
			        	
				}
			}
			br.close();
		} catch (Exception e) {
			System.out.println(">>>>>> Something wrong with testing <<<<<<");
			e.printStackTrace();
		}
		
		System.out.println("Done Loading Testing Set");
	}


	public void train(Integer time) throws Exception {
		try {
			if (time == 1) {
				System.out.println("Commence 1st Training (" + otrainingSet.numInstances() + ")");
				for (int counter : otrain_counter)
					System.out.println(counter);
				
				oModel.buildClassifier(otrainingSet);
			} else if (time == 2) {
				System.out.println("Commence 2nd Training (" + trainingSet.numInstances() + ")");
				for (int counter : train_counter)
					System.out.println(counter);
				
				cModel.buildClassifier(trainingSet);
			}
			System.out.println("Done Training");
		} catch (Exception e) {
			
		}
	}
	
	
	
	public void test() throws Exception {
		System.out.println("Commence Testing");
		
		
		// Test the model
		Evaluation eTest = new Evaluation(testingSet);
		eTest.evaluateModel(cModel, testingSet);
		 
		// Print the result 
		String strSummary = eTest.toSummaryString();
		System.out.println(strSummary);
	}
	
	private List<String> array2List(String[] words){
		List<String> sentence = new ArrayList<String>();
		for(int i = 0; i < words.length; i++){
			sentence.add(words[i]);
		}
		return sentence;
	}
	
	public int[] getCounter() {
		return counter;
	}
	
//	public int[] getOTCounter() {
//		return otrain_counter;
//	}
//	
//	public int[] getTCounter() {
//		return train_counter;
//	}
	
	public static void main (String[] args) throws Exception {
		WekaClassifiers wc = new WekaClassifiers(2);
		
//		wc.loadTrainingData(1);
		wc.loadTrainingData(2);
//		wc.train(1);
		wc.train(2);
		
		wc.loadTestingData();
//		wc.test();
		
		System.out.println("Printing results:");
		int[] count = wc.getCounter();
		int sum = 0;
		for (int c : count) {
			System.out.println(c);
			sum += c;
		}
		System.out.println("total correct:" + sum);
		
		System.out.println("DONE.");
	}

}
