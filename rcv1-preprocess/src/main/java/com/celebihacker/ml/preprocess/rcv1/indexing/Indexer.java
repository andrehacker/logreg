package com.celebihacker.ml.preprocess.rcv1.indexing;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.Version;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.celebihacker.ml.preprocess.rcv1.indexing.featureextraction.EnglishWithNumberFilterAnalyzer;
import com.celebihacker.ml.preprocess.rcv1.indexing.featureextraction.NewsItemFeatureExtraction;
import com.celebihacker.ml.preprocess.rcv1.indexing.featureextraction.EnglishWithNumberFilterAnalyzer.NumberFilterMethod;
import com.celebihacker.ml.preprocess.rcv1.indexing.parsing.XmlToNewsItemParser;
import com.celebihacker.ml.preprocess.rcv1.indexing.types.NewsItem;
import com.google.common.base.Optional;
import com.google.common.io.Closeables;

public class Indexer {

    private static final Logger LOGGER = LoggerFactory.getLogger(Indexer.class);

    private static final NumberFilterMethod DEFAULT_NUMBER_FILTER_METHOD = NumberFilterMethod.KEEP;

    private static final FilenameFilter ZIP_FILES = new FilenameFilter() {
        @Override
        public boolean accept(File dir, String name) {
            return name.endsWith(".zip");
        }
    };

    public static void index(String inputPath, String outputPath) throws IOException {
        index(inputPath, outputPath, DEFAULT_NUMBER_FILTER_METHOD);
    }

    public static void index(String pathToInput, String pathToOutput, NumberFilterMethod numberFilterMethod)
            throws IOException {
        // Input/output
        File[] zipFiles = new File(pathToInput).listFiles(ZIP_FILES);
        File indexDir = new File(pathToOutput);

        // Config
        Analyzer analyzer = new EnglishWithNumberFilterAnalyzer(numberFilterMethod);
        IndexWriterConfig conf = new IndexWriterConfig(Version.LUCENE_43, analyzer);

        IndexWriter writer = new IndexWriter(new SimpleFSDirectory(indexDir), conf);

        // Parse XML files and add documents to index
        NewsItemFeatureExtraction featureExtraction = new NewsItemFeatureExtraction();
        for (File f : zipFiles) {
            LOGGER.info("Indexing file: {}", f.getName());

            ZipFile zip = new ZipFile(f);
            Enumeration<? extends ZipEntry> xmlFiles = zip.entries();

            while (xmlFiles.hasMoreElements()) {
                InputStream xml = zip.getInputStream(xmlFiles.nextElement());

                try {
                    Optional<NewsItem> newsItem = Optional.fromNullable(XmlToNewsItemParser
                            .toNewsItem(xml));

                    if (newsItem.isPresent()) {
                        Document doc = featureExtraction.extract(newsItem.get());
                        writer.addDocument(doc);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                Closeables.close(xml, true);
            }
        }

        Closeables.close(writer, true);
    }
}