package edu.csupomona.nlp.ml.supervised.LibSvm;

import edu.csupomona.nlp.util.*;

import java.io.*;
import java.util.*;

/**
 * Created by xing on 1/28/15.
 */
public class Prepare {

    // list contains user defined aspects
    private List<String> aspects;

    // list contains user defined words related to each aspect
    private List<List<String>> aspectWords;

    // break iterator (not very accurate?)
    private final SentenceDetector sentDect;

    // stopwords removal
    private final Stopword sw;

    // stemming
    private final Stemmer st;

    private boolean enableStemmer;

    // feature : index mapping
    private HashMap<String, Integer> featMap_;

    public Prepare(HashMap<String, List<Integer>> freqMap) {
        featMap_ = new HashMap<>();
        int index = 0;

        for (String feature : freqMap.keySet())
            featMap_.put(feature, index++);

        sentDect = new SentenceDetector("en");

        this.sw = new Stopword("en");
        this.st = new Stemmer("en");

        this.enableStemmer = true;


    }

    /**
     * Get aspect words from the aspect file.
     * Each word should possess a single line
     * @param file          Aspect file containing aspect related words
     * @return              List of aspect words
     * @throws IOException
     */
    private List<String> getAspectWords(File file) throws IOException{
        FileReader fr = new FileReader(file);
        List<String> words;
        try (BufferedReader br = new BufferedReader(fr)) {
            String word;
            words = new ArrayList<>();
            while((word = br.readLine())!=null){
                word = word.trim(); // remove whitespace
                if(!word.isEmpty()){
                    words.add(word);
                }
            }
        }

        return words;
    }


    /**
     * Load all aspect and related words defined by user
     * @param path          Path to the folder contains aspect files
     * @throws java.io.IOException
     */
    private Long[] loadAspects(String path)
            throws IOException {
        // drop old list and construct new
        this.aspects = new ArrayList<>();
        this.aspectWords = new ArrayList<>();

        File[] files = new File(path).listFiles();
        Arrays.sort(files);

//        System.out.println("Loading Aspects:");
        for (File file : files) {
            if (file.getName().equals(".DS_Store"))
                continue;

            String aspect = file.getName();
//            System.out.println(aspect);

            // get user defined words for each aspect
            List<String> words = getAspectWords(file);

            // add to specific list
            this.aspectWords.add(words);
            this.aspects.add(aspect);
        }

        // add a list for aspect not defined by user
        // NOTE: "others" has to be at the tail of the list
        List<String> others = new ArrayList<>();
        others.add("other");
        this.aspectWords.add(others);

        // create aspect sentence
        Long[] aspectSentence = new Long[aspectWords.size()];
        for (int idx =0; idx < aspectSentence.length; ++idx)
            aspectSentence[idx] = 0L;
        return aspectSentence;
    }

    private List<File> getFiles(String path) throws IOException {
        List<File> files = new ArrayList<>();

        File[] candidates = new File(path).listFiles();
        for (File file : candidates) {
            if (file.isFile())
                files.add(file);
            else
                files.addAll(getFiles(file.getCanonicalPath()));
        }

        return files;
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


    private String constructFeatValueStr(int aspectIdx, HashMap<String, Integer> featValues) {
        if (featValues.size() == 0)
            return null;

        StringBuilder featValueStr = new StringBuilder();

        // first part: label
        featValueStr.append(aspectIdx);
        featValueStr.append(" ");

        // convert hashmap
        HashMap<Integer, Integer> pairs = new HashMap<>();
        for (String ngram : featValues.keySet())
            if (featMap_.containsKey(ngram))
                pairs.put(featMap_.get(ngram) + 1, featValues.get(ngram));

        // second part: feature:value pairs
        SortedSet<Integer> indices = new TreeSet<>(pairs.keySet());
        for (Integer idx : indices) {

            featValueStr.append(idx);
            featValueStr.append(":");
            featValueStr.append(pairs.get(idx));
            featValueStr.append(" ");

        }

        // ending with "\n"
        featValueStr.append("\n");

        return featValueStr.toString();
    }

    private List<String> parseNGram(Integer W, Integer N,
                            List<String> words, List<List<String>> aspectWords) {
        List<String> parsedResult = new ArrayList<>();
        int begin = 0;
        int end = 0;    // when sentence contains no aspect words
        int aspectIdxOfSentence;   // init as "others" class

        // search through all the aspects (neglacting "others")
        for (int i = 0; i < aspectWords.size()-1; i++) {
            // search through all aspect words for each aspect
            for (int j = 0; j < aspectWords.get(i).size(); j++) {
                String aspectWord = aspectWords.get(i).get(j);

                // stem the aspect word as well
                if (enableStemmer)
                    aspectWord = st.stemWord(aspectWord);

                if (words.contains(aspectWord)) {
                    // label it
                    aspectIdxOfSentence = i;

                    // boundary of the window area
                    int pos = words.indexOf(aspectWord);
                    begin = ((pos - W) > 0 ? pos - W : 0);
                    end = ((pos + W) < words.size() ?
                            pos + W : words.size()-1);

                    // extract n-gram within the window
                    HashMap<String, Integer> featValues = new HashMap<>();
                    NGram.updateNGram(N, featValues, words.subList(begin, end));
                    String line = constructFeatValueStr(aspectIdxOfSentence, featValues);
                    if (line != null)
                        parsedResult.add(line);
                }
            }
        }

        aspectIdxOfSentence = aspectWords.size() - 1;
        // extract the before window part of sentence
        HashMap<String, Integer> featValues = new HashMap<>();
        NGram.updateNGram(N, featValues, words.subList(0, begin));
        String line = constructFeatValueStr(aspectIdxOfSentence, featValues);
        if (line != null)
            parsedResult.add(line);

        // extract the after window part of sentence
        featValues = new HashMap<>();
        NGram.updateNGram(N, featValues, words.subList(end, words.size()));
        line = constructFeatValueStr(aspectIdxOfSentence, featValues);
        if (line != null)
            parsedResult.add(line);

        return parsedResult;
    }

    private List<String> parseAspect(String sentence, Integer W, Integer N,
                            final List<List<String>> aspectWords) {
        // a little preprocessing
        String adjustedSentence = adjustSent(sentence).toLowerCase();

        // tokenize
        List<String> words = new ArrayList<>(
                Arrays.asList(adjustedSentence.split(" ")));

        // remove stopwords for unigram
        if (N == 1)
            words = sw.rmStopword(words);

        // stemming
        if (enableStemmer)
            words = st.stemWords(words);

        // parse n-gram
        if (words.size() > 0)
            return parseNGram(W, N, words, aspectWords);
        else
            return null;
    }

    private void parseFile(Integer W, Integer N,
                           File file, String parsedFilePath)
            throws IOException {
        List<String> resultOfFile = new ArrayList<>();
        // read the file
        FileReader fr = new FileReader(file);
        try (BufferedReader br = new BufferedReader(fr)) {
            String text;

            // parse each line in the file
            while ((text = br.readLine()) != null) {
                // split the reviews into sentences
                List<String> sentences = sentDect.simple(text);

                // loop through each sentence
                for (String sentence : sentences) {
                    List<String> resultOfSentence = parseAspect(sentence, W, N, aspectWords);

                    if (resultOfSentence != null)
                        resultOfFile.addAll(resultOfSentence);
                }

            }
        }

        // write to file
        FileWriter fw = new FileWriter(parsedFilePath, true);   // append mode
        try (BufferedWriter bw = new BufferedWriter(fw)) {
            for (String line : resultOfFile)
                bw.write(line);
        }

    }


    /**
     * Parse files and obtain N-gram information using aspect path for aspect list
     * @param W                     Window size
     * @param N                     N of N-gram
     * @param aspectsPath           Path to folder contains aspects
     * @param trainSetPath          Path to folder contains training data
     * @param parsedFilePath        Path to parsed LibSVM format file
     * @return
     * @throws IOException
     */
    public void parseTrain(Integer W, Integer N,
                        String aspectsPath,
                        String trainSetPath,
                        String parsedFilePath)
            throws IOException {
        // remove exist file
        File parsedFile = new File(parsedFilePath);
        if (parsedFile.exists())
            parsedFile.delete();

        // load aspect related words
        loadAspects(aspectsPath);

        // for each file in training set path
        List<File> files = getFiles(trainSetPath);
        int idx = 1;
        int total = files.size();
        for (File file : files) {
            parseFile(W, N, file, parsedFilePath);
            System.out.println("parsed " + idx++ + "/" + total);
        }
    }

    public String parseTest(Integer N, String sentence, int aspectIdxOfSentence) {
        // a little preprocessing
        String adjustedSentence = adjustSent(sentence).toLowerCase();

        // tokenize
        List<String> words = new ArrayList<>(
                Arrays.asList(adjustedSentence.split(" ")));

        // remove stopwords for unigram
        if (N == 1)
            words = sw.rmStopword(words);

        // stemming
        if (enableStemmer)
            words = st.stemWords(words);

        // parse n-gram
        if (words.size() > 0) {
            HashMap<String, Integer> featValues = new HashMap<>();
            NGram.updateNGram(N, featValues, words);
            return constructFeatValueStr(aspectIdxOfSentence, featValues);
        } else
            return null;
    }

}
