package com.andrehacker.ml.datasets;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import com.andrehacker.ml.util.MLUtils;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Closeables;

/**
 * Provides information about RCV1-v2 dataset [1]
 * 
 * Why static? This makes it easy to access it from Mappers and Reducers
 * 
 * Main categories:
 * - CCAT (Corporate/Industrial)
 * - ECAT (Economics)
 * - GCAT (Government/Social)
 * - MCAT (Markets)
 * 
 * Some facts about RCV1-v2
 * - 47,236     highest term id
 * - 23,149     training vectors
 * - 781,265    test vectors
 * - 804,414    total (training + test)
 * - 810,935    highest document-id
 * - 381,327    points labeled with CCAT (RCV1-v2)
 * - 119,920    points labeled with ECAT (RCV1-v2)
 * - 239,267    points labeled with GCAT (RCV1-v2)
 * - 204,820    points labeled with MCAT (RCV1-v2)
 * 
 * REFERENCES:
 * [1] Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: 
 * A New Benchmark Collection for Text Categorization Research. 
 * Journal of Machine Learning Research, 5:361-397, 2004.
 * http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf
 * http://jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
 * 
 */
public class RCV1DatasetInfo {

  private static final int NUM_FEATURES = 47237;    // we already added 1 here
  private static final int TOTAL = 810935;
  
  private static final Map<Integer, String> labelMap = ImmutableMap.of(
      1, "CCAT",
      2, "ECAT",
      3, "GCAT",
      4, "MCAT"
      );

  private static DatasetInfo datasetInfo = new DatasetInfo.Builder(NUM_FEATURES, TOTAL).
      labelMap(labelMap).build();
  
  public static DatasetInfo get() {
    return datasetInfo;
  }
  
  /**
   * Loads the map (feature-id -> feature name) from local file
   * and adds it to the datasetInfo
   * 
   * The file is available on the website (see above):
   * stem.termid.idf.map.txt
   */
  public static void readPredictorNames(String path) throws IOException {

    Splitter spaceSplitter = Splitter.on(Pattern.compile(" "));
    
    // Line format: stem term-id idf
    // e.g. profit 33191 2.01767426583078
    List<String> names = Arrays.asList(new String[NUM_FEATURES]);
    
    BufferedReader reader = null;
    
    int count=0;
    try {
      reader = MLUtils.open(path);
      String line;
      String stem;
      String termid;
      while ((line = reader.readLine()) != null) {
        Iterator<String> iter = spaceSplitter.split(line).iterator();
        stem = iter.next();
        termid = iter.next();
        names.set(Integer.parseInt(termid), stem);
        ++count;
      }
    } finally {
      Closeables.close(reader, true);
    }
    System.out.println("Read Termid->stem map (" + count + " items)");
    
    datasetInfo.setPredictorNames(names);
  }
  
}
