package com.celebihacker.ml.preprocess.rcv1.vectorization;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.MultiFields;
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

import com.celebihacker.ml.preprocess.rcv1.indexing.featureextraction.NewsItemFeatureExtraction;
import com.celebihacker.ml.preprocess.rcv1.indexing.types.RCV1;
import com.celebihacker.ml.writables.IDAndLabels;
import com.google.common.io.Closeables;

public class Vectorizer {

    private static final Logger LOGGER = LoggerFactory.getLogger(Vectorizer.class);

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

    // Defaults settings
    private static final int DEFAULT_MIN_DF = 0;
    private static final double DEFAULT_TRAINING_RATIO = 0.8;
    private static final SplitType DEFAULT_SPLIT_TYPE = SplitType.DATE;
    private static final Weighting DEFAULT_WEIGHTING = Weighting.AIC;

    private final DirectoryReader reader;
    private final TermDict termDict;
    private final int minDf;

    private Weighting weighting = DEFAULT_WEIGHTING;

    /**************************************************************************
     * CONSTRUCTORS
     **************************************************************************/
    public Vectorizer(String pathToIndex) throws IOException {
        this(pathToIndex, DEFAULT_MIN_DF, DEFAULT_WEIGHTING);
    }
    
    public Vectorizer(String pathToIndex, int minDf) throws IOException {
        this(pathToIndex, minDf, DEFAULT_WEIGHTING);
    }

    public Vectorizer(String pathToIndex, int minDf, Weighting weighting) throws IOException {
        this.reader = DirectoryReader.open(new SimpleFSDirectory(new File(pathToIndex)));
        this.minDf = minDf;
        this.weighting = weighting;
        
        //
        this.termDict = computeTermDict();
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

        // Write term dictionary to file
        File outputPath = new File(pathToOutput);
        outputPath.mkdirs();

        File termFile = new File(outputPath,
                String.format("terms-%d.txt", this.termDict.numTerms()));
        this.termDict.writeToFile(termFile);
        LOGGER.info("Wrote term dictionary to {}", termFile);

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

        // Writers
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.getLocal(conf);

        String trainingFilename = String.format("training-%1.1f-%s-%d_docs-%s-%s_mindf.seq",
                trainingRatio, splitBy.toString(), numTrainingDocs, this.weighting.toString(), this.minDf);

        String testFilename = String.format("test-%1.1f-%s-%d_docs-%s-%s_mindf.seq",
                (1 - trainingRatio), splitBy.toString(), numTestDocs, this.weighting.toString(), this.minDf);

        SequenceFile.Writer trainingWriter = SequenceFile.createWriter(fs, conf,
                new Path(pathToOutput, trainingFilename), IDAndLabels.class, VectorWritable.class);

        SequenceFile.Writer testWriter = SequenceFile.createWriter(fs, conf,
                new Path(pathToOutput, testFilename), IDAndLabels.class, VectorWritable.class);

        // Output
        IDAndLabels idAndLabels = new IDAndLabels();
        VectorWritable featureVector = new VectorWritable();

        int numDocsVectorized = 0;

        // Training split
        for (int i = 0; i < numTrainingDocs; i++) {
            writeVector(docIds[i], numTrainingDocs, idAndLabels, featureVector, trainingWriter);

            if (numDocsVectorized % 1000 == 0)
                LOGGER.info("{} documents vectorized", numDocsVectorized);

            numDocsVectorized++;
        }

        // Test split
        for (int i = numTrainingDocs; i < numDocs; i++) {
            writeVector(docIds[i], numTestDocs, idAndLabels, featureVector, testWriter);

            if (numDocsVectorized % 1000 == 0)
                LOGGER.info("{} documents vectorized", numDocsVectorized);

            numDocsVectorized++;
        }

        LOGGER.info("{} documents vectorized in TOTAL", numDocsVectorized);

        Closeables.close(trainingWriter, true);
        Closeables.close(testWriter, true);
    }

    /**************************************************************************
     * WRITE SEQUENCE FILES AND GET VECTORS FROM INDEX
     **************************************************************************/
    private void writeVector(int docId, int numDocs, IDAndLabels key, VectorWritable val,
            SequenceFile.Writer writer)
            throws IOException {
        Document doc = this.reader.document(docId);

        int itemId = doc.getField(NewsItemFeatureExtraction.ITEM_ID).numericValue().intValue();

        key.set(itemId, getLabelVector(docId));
        val.set(getFeatureVector(docId, numDocs));

        writer.append(key, val);
    }

    private Vector getFeatureVector(int docId, int numDocs) throws IOException {
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
            int df = this.termDict.df(termStr);
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
    private TermDict computeTermDict() throws IOException {
        TermDict dict = new TermDict();

        Terms terms = MultiFields.getFields(this.reader).terms(NewsItemFeatureExtraction.TEXT);
        TermsEnum termIterator = terms.iterator(null);

        BytesRef termBytes;
        while ((termBytes = termIterator.next()) != null) {
            Term t = new Term(NewsItemFeatureExtraction.TEXT, termBytes);
            int df = this.reader.docFreq(t);

            if (df < this.minDf)
                continue;

            dict.add(termBytes.utf8ToString(), df);
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
                docWeight = Math.log((double) numDocs / df);
                break;
        }

        weight = termWeight * docWeight;

        return weight;
    }

    private void normalize(Vector vector) {
        switch (this.weighting.vectorNormalization) {
            case NONE:
                return;

            case COS:
                double norm = 1 / Math.sqrt(vector.getLengthSquared());

                for (Element e : vector.nonZeroes()) {
                    e.set(e.get() * norm);
                }
        }
    }

    public void setWeighting(Weighting weighting) {
        if (weighting == null)
            throw new NullPointerException("weighting must not be null");

        this.weighting = weighting;
    }
}