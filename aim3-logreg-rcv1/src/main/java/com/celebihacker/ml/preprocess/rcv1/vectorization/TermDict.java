package com.celebihacker.ml.preprocess.rcv1.vectorization;

import com.google.common.base.Splitter;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TermDict maps terms to their id and stores their document frequency.
 */
public class TermDict {

    private static final Logger LOGGER = LoggerFactory.getLogger(TermDict.class);

    private final Map<String, TermInfo> terms;

    private int nextTermId;

    public TermDict() {
        this.terms = Maps.newHashMap();
        this.nextTermId = 0;
    }

    public void add(String term, int df) {
        if (this.terms.containsKey(term))
            return;

        this.terms.put(term, new TermInfo(this.nextTermId, df));
        this.nextTermId++;

        if (this.nextTermId % 5000 == 0)
            LOGGER.info("{} terms added", this.nextTermId);
    }

    public boolean contains(String term) {
        return this.terms.containsKey(term);
    }

    public int numTerms() {
        return this.nextTermId;
    }

    public int id(String term) {
        if (this.terms.containsKey(term))
            return this.terms.get(term).id;

        throw new IllegalArgumentException("Unknown term " + term);
    }

    public int df(String term) {
        if (this.terms.containsKey(term))
            return this.terms.get(term).df;

        throw new IllegalArgumentException("Unknown term " + term);
    }

    public void writeToFile(File file) throws IOException {
        SortedMap<Integer, String> idToTerm = new TreeMap<Integer, String>();

        for (Entry<String, TermInfo> e : this.terms.entrySet()) {
            String term = e.getKey();
            int termId = e.getValue().id;

            idToTerm.put(termId, term);
        }

        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter(file));

            // headline
            writer.write(String.format(" %7s | %20s | %6s", "id", "term", "df"));
            writer.newLine();

            writer.write("|---------------------------------------------------|");
            writer.newLine();

            for (Entry<Integer, String> e : idToTerm.entrySet()) {
                String term = e.getValue();
                TermInfo t = this.terms.get(term);

                int termId = t.id;
                int df = t.df;

                writer.write(
                        String.format(" %7d | %20s | %6d", termId, term, df));
                writer.newLine();
            }
        } finally {
            Closeables.close(writer, true);
        }
    }

    public static TermDict readFromFile(File file) throws IOException {
        TermDict terms = new TermDict();

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));

            // skip headline
            reader.readLine();
            reader.readLine();

            Splitter splitter = Splitter.on("|");

            String line;
            while ((line = reader.readLine()) != null) {
                Iterator<String> it = splitter.split(line).iterator();

                // skip id
                it.next();

                String term = it.next();
                int df = Integer.valueOf(it.next().trim());

                terms.add(term, df);
            }
        } finally {
            Closeables.close(reader, true);
        }

        return terms;
    }

    private class TermInfo {
        public final int id;
        public final int df;

        public TermInfo(int termId, int df) {
            this.id = termId;
            this.df = df;
        }
    }
}