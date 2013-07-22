package de.tuberlin.dima.ml.preprocess.rcv1.vectorization;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortField.Type;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

import de.tuberlin.dima.ml.preprocess.rcv1.indexing.featureextraction.NewsItemFeatureExtraction;
import de.tuberlin.dima.ml.preprocess.rcv1.indexing.types.RCV1;
import de.tuberlin.dima.ml.preprocess.rcv1.vectorization.types.IdAndLabels;

// TODO could use some refactoring
public class Vectorizer {

    private static final Logger LOGGER = LoggerFactory.getLogger(Vectorizer.class);

    // Defaults settings
    private static final int DEFAULT_MIN_DF = 0;
    private static final double DEFAULT_TRAINING_RATIO = 0.8;
    private static final SplitType DEFAULT_SPLIT_TYPE = SplitType.DATE;
    private static final Weighting DEFAULT_WEIGHTING = Weighting.AIC;
    private static final NumberFilterMethod DEFAULT_NUMBER_FILTER_METHOD = NumberFilterMethod.REMOVE;

    private final DirectoryReader reader;
    private final int minDf;
    private final Weighting weighting;
    private TermDict termDict;
    private NumberFilterMethod numberFilterMethod;

    /**************************************************************************
     * CONSTRUCTORS
     **************************************************************************/
    public Vectorizer(String pathToIndex) throws IOException {
        this(pathToIndex, DEFAULT_MIN_DF, DEFAULT_WEIGHTING, DEFAULT_NUMBER_FILTER_METHOD);
    }
    
    public Vectorizer(String pathToIndex, int minDf) throws IOException {
        this(pathToIndex, minDf, DEFAULT_WEIGHTING, DEFAULT_NUMBER_FILTER_METHOD);
    }
    
    public Vectorizer(String pathToIndex, int minDf, Weighting weighting) throws IOException {
        this(pathToIndex, minDf, weighting, DEFAULT_NUMBER_FILTER_METHOD);
    }

    public Vectorizer(String pathToIndex, int minDf, Weighting weighting, NumberFilterMethod numberFilterMethod) throws IOException {
        this.reader = DirectoryReader.open(new SimpleFSDirectory(new File(pathToIndex)));
        this.minDf = minDf;
        this.weighting = weighting;
        this.numberFilterMethod = numberFilterMethod;
    }

    /**************************************************************************
     * API
     **************************************************************************/
    public void vectorize(String pathToOutput) throws IOException {
        vectorize(pathToOutput, DEFAULT_SPLIT_TYPE, DEFAULT_TRAINING_RATIO);
    }

    public void vectorize(String pathToOutput, SplitType splitBy) throws IOException {
        vectorize(pathToOutput, splitBy, DEFAULT_TRAINING_RATIO);
    }

    public void vectorize(String pathToOutput, SplitType splitBy, double trainingRatio)
            throws IOException {

    	// Output
    	File outputPath = new File(pathToOutput);
        outputPath.mkdirs();
    	
        // Split counts
        int numDocs = this.reader.numDocs();
        int numTrainingDocs = (int) (numDocs * trainingRatio);
        int numTestDocs = numDocs - numTrainingDocs;

        LOGGER.info("{} documents total", numDocs);
        LOGGER.info("{} training documents ({} of all)", numTrainingDocs, trainingRatio);
        LOGGER.info("{} test documents ({} of all)", numTestDocs, 1 - trainingRatio);

        // Split sorting: sort documents Ids and use first numTraining as training split
        int[] docIds = sortDocIds(this.reader.maxDoc(), splitBy);
        LOGGER.info("Sort documents ids on {}", splitBy);
        
        // Term Dict (actually need to call just once, but depends on split sizes for df counts -.-)
        LOGGER.info("Creating term dictionry");
        this.termDict = computeTermDict(numTrainingDocs, numDocs);
        
        // Write term dictionary to file
        File termFile = new File(outputPath,
        	String.format("terms-%d.txt", this.termDict.numTerms()));

        this.termDict.writeToFile(termFile);
        LOGGER.info("Wrote term dictionary to {}", termFile);

        // Writers
        String trainingFilename = String.format("training-%1.1f-%s-%d_docs-%s-%s_mindf.libsvm",
                trainingRatio, splitBy.toString(), numTrainingDocs, this.weighting.toString(), this.minDf);

        String testFilename = String.format("test-%1.1f-%s-%d_docs-%s-%s_mindf.libsvm",
                (1 - trainingRatio), splitBy.toString(), numTestDocs, this.weighting.toString(), this.minDf);

        File trainingFile = new File(outputPath, trainingFilename);
        File testFile = new File(outputPath, testFilename);
        
        BufferedWriter trainingWriter = new BufferedWriter(new FileWriter(trainingFile));
        BufferedWriter testWriter = new BufferedWriter(new FileWriter(testFile));

        // Output
        IdAndLabels idAndLabels = new IdAndLabels();
        VectorWritable featureVector = new VectorWritable();

        int numDocsVectorized = 0;
        
        // Training split
        for (int i = 0; i < numTrainingDocs; i++) {
            setVector(docIds[i], numTrainingDocs, Split.TRAINING, idAndLabels, featureVector);
            writeVector(idAndLabels, featureVector, trainingWriter);

            if (numDocsVectorized % 1000 == 0)
                LOGGER.info("{} documents vectorized", numDocsVectorized);

            numDocsVectorized++;
        }

        Closeables.close(trainingWriter, true);
        
        // Test split
        for (int i = numTrainingDocs; i < numDocs; i++) {
            setVector(docIds[i], numTestDocs, Split.TEST, idAndLabels, featureVector);
            writeVector(idAndLabels, featureVector, testWriter);

            if (numDocsVectorized % 1000 == 0)
                LOGGER.info("{} documents vectorized", numDocsVectorized);

            numDocsVectorized++;
        }

        Closeables.close(testWriter, true);
        
        LOGGER.info("{} documents vectorized in TOTAL", numDocsVectorized);
    }

    /**************************************************************************
     * WRITE FILES AND GET VECTORS FROM INDEX
     **************************************************************************/    
    private void writeVector(IdAndLabels key, VectorWritable val, BufferedWriter writer)
            throws IOException {
        
    	StringBuilder sb = new StringBuilder();

    	boolean first = true;
    	
    	// labels
    	for (Element e : key.getLabels().nonZeroes()) {
    		sb.append(first ? e.index() : "," + e.index());
			first = false;
    	}
    	
    	for (Element e : val.get().nonZeroes()) {
    		sb.append(" " + e.index() + ":" + e.get());
    	}
    	
        writer.write(sb.toString());
        writer.newLine();
    }

	private void setVector(int docId, int numDocs, Split split, IdAndLabels key, VectorWritable val) throws IOException {
    	Document doc = this.reader.document(docId);

        int itemId = doc.getField(NewsItemFeatureExtraction.ITEM_ID).numericValue().intValue();

        key.set(itemId, getLabelVector(docId));
        val.set(getFeatureVector(docId, numDocs, split));
    }
    
    private Vector getFeatureVector(int docId, int numDocs, Split split) throws IOException {
        RandomAccessSparseVector docVector = new RandomAccessSparseVector(this.termDict.numTerms());

        Terms terms = this.reader.getTermVector(docId, NewsItemFeatureExtraction.TEXT);
        TermsEnum termIterator = terms.iterator(null);
        BytesRef term;

        int maxTf = 1;

        // Compute maximumTf
        if (this.weighting.termWeighting == Weighting.TermWeighting.AUGMENTED) {
            while ((term = termIterator.next()) != null) {
                if (!this.termDict.contains(term.utf8ToString()))
                    continue;

                int tf = (int) termIterator.totalTermFreq();
                maxTf = Math.max(maxTf, tf);
            }
        }

        // Weight terms
        termIterator = terms.iterator(termIterator);
        while ((term = termIterator.next()) != null) {
            String termStr = term.utf8ToString();

            if (!this.termDict.contains(termStr))
                continue;

            int termId = this.termDict.id(termStr);

            int tf = (int) termIterator.totalTermFreq();
            int df = (split == Split.TRAINING) ? this.termDict.dfTraining(termStr) : this.termDict.dfTest(termStr);
            double weight = weight(tf, df, maxTf, numDocs);

            docVector.setQuick(termId, weight);
        }

        // Finish
        Vector vector = new SequentialAccessSparseVector(docVector);
        normalize(vector);

        return vector;
    }

    private Vector getLabelVector(int docId) throws IOException {
        Vector topicVector = new DenseVector(RCV1.TOP_LEVEL_TOPICS.size());

        Terms terms = this.reader.getTermVector(docId, NewsItemFeatureExtraction.TOPICS);
        TermsEnum termIterator = terms.iterator(null);
        BytesRef term;

        while ((term = termIterator.next()) != null) {
            int id = RCV1.TOP_LEVEL_TOPICS.get(term.utf8ToString().toUpperCase());
            topicVector.setQuick(id, 1);
        }

        return topicVector;
    }

    /**************************************************************************
     * DATA SPLITS
     **************************************************************************/
    /**
     * Sort array of document IDs between 0 (inclusive) and maxDoc (exclusive) according to the
     * SplitType, either by sorting on document date or shuffling them randomly.
     * 
     * The first numTrainingDocs document IDs can be used for the training split and the remaining
     * ones for the test split.
     * 
     * @param maxDoc
     *            Largest document ID (exclusive)
     * @return Array of documents IDs sorted by SplitType
     * @throws IOException
     */
    private int[] sortDocIds(int maxDoc, SplitType splitBy) throws IOException {
        int[] docIds = new int[maxDoc];

        switch (splitBy) {
            case DATE:
                SortField[] sortBy = {
                                new SortField(NewsItemFeatureExtraction.DATE, Type.STRING),
                                new SortField(NewsItemFeatureExtraction.ITEM_ID, Type.INT)
                };

                TopFieldDocs sortedDocs = new IndexSearcher(this.reader).search(
                        new MatchAllDocsQuery(), maxDoc, new Sort(sortBy));

                for (int i = 0; i < maxDoc; i++) {
                    docIds[i] = sortedDocs.scoreDocs[i].doc;
                }

                break;
            case RANDOM:
                // Sequential docIds 0-(maxDoc-1)
                for (int i = 0; i < docIds.length; i++) {
                    docIds[i] = i;
                }

                // Fisher-Yates shuffle
                Random r = new Random();

                for (int n = maxDoc - 1; n > 0; n--) {
                    int randomSwap = r.nextInt(n + 1);

                    int last = docIds[n];
                    docIds[n] = docIds[randomSwap];
                    docIds[randomSwap] = last;
                }

                break;
        }

        return docIds;
    }

    /**************************************************************************
     * TERM DICTIONARY
     **************************************************************************/
    private TermDict computeTermDict(int splitDocId, int numDocs) throws IOException {
    	
    	TermDict dict = new TermDict();
    	
    	for (int docId = 0; docId < numDocs; docId++) {
    		
    		Terms terms = this.reader.getTermVector(docId, NewsItemFeatureExtraction.TEXT);
    		TermsEnum termIterator = terms.iterator(null);
    		
    		BytesRef termBytes;
    		while((termBytes = termIterator.next()) != null) {
    			
    			String termStr = this.numberFilterMethod.filter(termBytes.utf8ToString());
    			
    			if (termStr == null)
    				continue;
    			
    			if (!dict.contains(termStr)) {    				
    					// add new term to dict
	    				Term t = new Term(NewsItemFeatureExtraction.TEXT, termBytes);
	    				int df = this.reader.docFreq(t);
	    				
	    				if (df < this.minDf) {
	    					continue;
	    				}
	    				
	    				dict.add(termStr, df);
	    				
	    				// increment count in split (set to 1)
	    				if (docId < splitDocId) {
		    				dict.incDfTraining(termStr);
		    			} else {
		    				dict.incDfTest(termStr);
		    			}

    			} else {
    				// term already in dict, just increment count    				
	    			if (docId < splitDocId) {
	    				dict.incDfTraining(termStr);
	    			} else {
	    				dict.incDfTest(termStr);
	    			}
	    			
    			}
    		}
    	}
    	
    	return dict;
    	
    }
    
    /**************************************************************************
     * WEIGHTING AND NORMALIZATION
     **************************************************************************/
    private double weight(int tf, int df, int maxTf, int numDocs) {
        double weight, termWeight, docWeight;
        weight = termWeight = docWeight = 0;

        switch (this.weighting.termWeighting) {
            case NONE:
                termWeight = 1;
                break;
            case LOG:
                termWeight = 1 + Math.log(tf);
                break;
            case AUGMENTED:
                termWeight = 0.5 + ((0.5 * tf) / maxTf);
                break;
        }

        switch (this.weighting.docWeighting) {
            case NONE:
                docWeight = 1;
                break;
            case IDF:
                docWeight = Math.log(((double) numDocs) / df);
                break;
        }
        

        weight = termWeight * docWeight;

        if  (weight < 0) {
        	System.out.println(termWeight + " " + docWeight + " @ numDocs " + numDocs + " and df " + df);
        }
        return weight;
    }

    private void normalize(Vector vector) {
        switch (this.weighting.vectorNormalization) {
            case NONE:
                return;

            case COS:
                double norm = 1d / Math.sqrt(vector.getLengthSquared());

                for (Element e : vector.nonZeroes()) {                	
                    e.set(e.get() * norm);
                }
        }
    }
    
    public enum NumberFilterMethod {
        KEEP(null), // Keep numbers
        REMOVE(Pattern.compile("^([0-9]+)([,.0-9a-z]*)$")), // Remove numbers
        
        // TODO modifying terms is tricky when counting doc frequency etc. - disabled for now
        ROUND(Pattern.compile("^([0-9]+)([,.0-9a-z]*)$")); // Remove part after ',' or '.'

        public final Pattern filterPattern;

        NumberFilterMethod(Pattern pattern) {
            this.filterPattern = pattern;
        }
        
        public String filter(String term) {
        	
        	if (this == KEEP)
                return term;

            Matcher filterMatcher = this.filterPattern.matcher(term);

            if (this == REMOVE)
            	return !filterMatcher.matches() ? term : null;

            if ((this == ROUND) && filterMatcher.matches())
            	return term.substring(0, filterMatcher.end(1));

            return term;
        }
    }

	private enum Split {
		TRAINING, TEST
	}
    
	// Settings for the training/test split
	public enum SplitType {
	    DATE, RANDOM;
	}

	// Weighting settings
	public enum Weighting {
	    NNN(TermWeighting.NONE, DocWeighting.NONE, VectorNormalization.NONE),
	    NNC(TermWeighting.NONE, DocWeighting.NONE, VectorNormalization.COS),
	    NIN(TermWeighting.NONE, DocWeighting.IDF, VectorNormalization.NONE),
	    NIC(TermWeighting.NONE, DocWeighting.IDF, VectorNormalization.COS),
	    LNN(TermWeighting.LOG, DocWeighting.NONE, VectorNormalization.NONE),
	    LNC(TermWeighting.LOG, DocWeighting.NONE, VectorNormalization.COS),
	    LIN(TermWeighting.LOG, DocWeighting.IDF, VectorNormalization.NONE),
	    LIC(TermWeighting.LOG, DocWeighting.IDF, VectorNormalization.COS),
	    ANN(TermWeighting.AUGMENTED, DocWeighting.NONE, VectorNormalization.NONE),
	    ANC(TermWeighting.AUGMENTED, DocWeighting.NONE, VectorNormalization.COS),
	    AIN(TermWeighting.AUGMENTED, DocWeighting.IDF, VectorNormalization.NONE),
	    AIC(TermWeighting.AUGMENTED, DocWeighting.IDF, VectorNormalization.COS);
	
	    private final TermWeighting termWeighting;
	    private final DocWeighting docWeighting;
	    private final VectorNormalization vectorNormalization;
	
	    Weighting(TermWeighting termWeighting, DocWeighting docWeighting,
	            VectorNormalization vectorNormalization) {
	        this.termWeighting = termWeighting;
	        this.docWeighting = docWeighting;
	        this.vectorNormalization = vectorNormalization;
	    }
	
	    // Settings for feature vectors
	    public enum TermWeighting {
	        NONE, LOG, AUGMENTED;
	    }
	
	    public enum DocWeighting {
	        NONE, IDF
	    }
	
	    public enum VectorNormalization {
	        NONE, COS;
	    }
	}
}