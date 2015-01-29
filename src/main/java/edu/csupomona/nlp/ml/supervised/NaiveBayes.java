/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package edu.csupomona.nlp.ml.supervised;

import edu.csupomona.nlp.util.NGram;
import edu.csupomona.nlp.util.Stemmer;
import edu.csupomona.nlp.util.Stopword;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Xing
 */
public class NaiveBayes {
    
    private class TrainData {
        HashMap<String, List<Integer>> freqMap;
        int[] aspectWordSum;
        Integer W;
        Integer N;

    }
    
    private final List<TrainData> trainData;
    
    private List<String> aspects;
    private HashMap<String, Long> aspectSentences;
    
    private Long aspectSentTotal = 0L;
    
    private final Pattern ptnWN = Pattern.compile("ngram_W([0-9]+)_N([0-9]+).*");
    
    private final Stopword sw;
    private final Stemmer st;

    private boolean enableStemmer;
    
    public NaiveBayes() {
        sw = new Stopword("en");
        st = new Stemmer("en");

        enableStemmer = true;
        
        trainData = new ArrayList<>();
    }
    
    private HashMap<String, List<Integer>> readNGram(File file) 
            throws IOException {
        FileReader fr = new FileReader(file);
        
        String line;
        HashMap<String, List<Integer>> map = new HashMap<>();
        try (BufferedReader br = new BufferedReader(fr)) {
            while ((line = br.readLine()) != null) {
                String[] items = line.split(",");
                List<Integer> counts = new ArrayList<>();
                int sum = 0;
                
                // add aspects
                int i;
                for (i=1; i<items.length-1; i++) {
                    int count = Integer.valueOf(items[i].trim());
                    counts.add(count);
                    sum += count;
                }
                
                // add others
                counts.add(Integer.valueOf(items[i].trim()));
                
                // add all
                counts.add(sum);
                
                map.put(items[0], counts);
            }
        }
        
        return map;
    }
    
    private List<Integer> readWN(String filename) {
        Matcher matcher = ptnWN.matcher(filename);
        
        List<Integer> wn = new ArrayList<>();
        if (matcher.matches()) {
            wn.add(Integer.valueOf(matcher.group(1)));  // W
            wn.add(Integer.valueOf(matcher.group(2)));  // N
        }
        
        return wn;
    }
    
    private HashMap<String, Long> readAspectSent(File file)
            throws IOException{
        FileReader fr = new FileReader(file);
        
        if (this.aspects == null)
            aspects = new ArrayList<>();
        
        Long countAll = 0L;
        String line;
        HashMap<String, Long> map = new HashMap<>();
        try (BufferedReader br = new BufferedReader(fr)) {
            while ((line = br.readLine()) != null) {
                String[] items = line.split(":");
                
                map.put(items[0], Long.valueOf(items[1].trim()));
                
                if (!items[0].equals("others"))
                    countAll += Integer.valueOf(items[1].trim());
                
                this.aspects.add(items[0]);
            }
            
            map.put("all", countAll);
            this.aspects.add("all");
        }

        return map;
    }
    
    private int[] calAspectWordSum(HashMap<String, List<Integer>> map) {
        int[] wordSum = new int[aspectSentences.size()];
        
        for (String key : map.keySet()) {
            int sum = 0;
            int i;
            // update aspect words
            for (i=0; i<map.get(key).size()-2; i++) {
                wordSum[i] += map.get(key).get(i);
                sum += map.get(key).get(i);
            }
            
            // update others
            wordSum[i] += map.get(key).get(i);
            i++;
            
            // update all
            wordSum[i] += sum;
        }
       
        return wordSum;
    }
    
    public void train(String ngramPath) throws IOException {
        File[] files = new File(ngramPath).listFiles();
        
        for (File file : files) {
            TrainData data = new TrainData();
            
            if (file.getName().contains("ngram")) {
                // parse the ngram files
                data.freqMap = readNGram(file);
                
                // parse the name of the files to obtain W and N info
                Matcher matcher = ptnWN.matcher(file.getName());
                if (matcher.matches()) {
                    data.W = Integer.valueOf(matcher.group(1));
                    data.N = Integer.valueOf(matcher.group(2));
                }
                
                // add one training set results into list
                trainData.add(data);
            }
            else 
                // parse the aspect sentences count
                aspectSentences = readAspectSent(file);
        }
        
        aspectSentTotal = aspectSentences.get("all")
                + aspectSentences.get("others");
        
        for (TrainData data : trainData)
            data.aspectWordSum = calAspectWordSum(data.freqMap);
        
    }
    
    public void train(List<String> aspects, 
            HashMap<String, List<Integer>> freqMap, 
            Integer W, Integer N, Long[] aspectSentences) {
        TrainData data = new TrainData();
        
        data.freqMap = freqMap;
        for (String key : data.freqMap.keySet()) {
            List<Integer> aspectWordNum = data.freqMap.get(key);
            int sum = 0;
            for (int i = 0; i < aspectWordNum.size()-1; i++)
                sum += aspectWordNum.get(i);
            aspectWordNum.add(sum);
            data.freqMap.put(key, aspectWordNum);
        }
        
        data.W = W;
        data.N = N;
        
        if (this.aspects == null)
            this.aspects = new ArrayList<>();
        
        this.aspectSentences = new HashMap<>();
        Long sum = 0L;
        for (int i = 0; i < aspects.size(); i++) {
            this.aspectSentences.put(aspects.get(i), aspectSentences[i]);
            sum += aspectSentences[i];
            this.aspects.add(aspects.get(i));
        }
        
        this.aspectSentences.put("others", 
                aspectSentences[aspectSentences.length-1]);
        this.aspects.add("others");
        this.aspectSentences.put("all", sum);
        this.aspects.add("all");
        
        aspectSentTotal = this.aspectSentences.get("all")
                + this.aspectSentences.get("others");
        
        data.aspectWordSum = calAspectWordSum(data.freqMap);
        
        trainData.add(data);
    }
    
    private double calNGramProb(String ngram, String aspect, Integer N, 
            HashMap<String, List<Integer>> map,
            int[] wordSum) {
        List<Integer> count = map.get(ngram);
        int idx = this.aspects.indexOf(aspect);
        int aspectCount = (count != null)? count.get(idx) : 0;
        int v = map.size();
        int total = wordSum[idx];
        
        return (double)(aspectCount + 1.0) / (total + v);
    }
    
    /**
     * Remove useless symbols (To Be Replaced In Future)
     * @param sentence
     * @return 
     */
    private String adjustSent(String sentence) {
//        return sentence.replaceAll(
//                "( +: ?| +\\*+ ?)|[\\[\\] \\(\\)\\.,;!\\?\\+-]", " ");
        String sent = sentence.replaceAll("[^a-zA-Z0-9]+", " ");
        return sent.replaceAll("[ ]+", " ");
    }
    
    private double calProbability(String aspect, String sentence) {
        String adjustedSentence = adjustSent(sentence).toLowerCase();
        String[] rawWords = adjustedSentence.split(" ");



        double sentenceProb;
        if(rawWords.length > 0){
            sentenceProb = Math.log((double)this.aspectSentences.get(aspect) 
                    / this.aspectSentTotal);    // bugfix: add Math.log

            // stemming
            List<String> words = new ArrayList<>();
            if (enableStemmer)
                for (String word : rawWords) {
                    word = st.stemWord(word);
                    words.add(word);
                }
            else
                words.addAll(Arrays.asList(rawWords));
            
            for (TrainData data : trainData) {
                int N = data.N;
                
                // get n-gram from the sentence
                HashMap<String, Integer> map = new HashMap<>();
                if (N == 1) {
                    List<String> trimWords = sw.rmStopword(words);
                    NGram.updateNGram(N, map, trimWords);
                } else 
                    NGram.updateNGram(N, map, words);
                
                // add probability for each n-gram
                for (String ngram : map.keySet()) 
                    sentenceProb += 
                            Math.log(calNGramProb(ngram, aspect, N, 
                                    data.freqMap,
                                    data.aspectWordSum))
                            * map.get(ngram);
            }
        }else{
                sentenceProb = 0.0;
        }
        return sentenceProb;
    }
    
    public NaiveBayesResult classify(String sentence) {
        double max = Double.NEGATIVE_INFINITY;
        String prediction = "others"; //assume it is not talking about any aspects
        List<String> iter1 = aspects.subList(aspects.size()-2, aspects.size());
        for(String aspect : iter1){
            double aspectProb = calProbability(aspect, sentence);
            if(aspectProb > max && aspectProb != 0.0){
                max = aspectProb;
                prediction = aspect;
            }
        }
        
        if(prediction.equals("all")){
            List<String> iter2 = aspects.subList(0, aspects.size()-2);
            
            max = Double.NEGATIVE_INFINITY;
            for(String aspect : iter2){
                double aspectProb = calProbability(aspect, sentence);
                if(aspectProb > max && aspectProb != 0.0){
                    max = aspectProb;
                    prediction = aspect;
                }
            }
        }
        
        return new NaiveBayesResult(prediction, max);
    }

    
}
