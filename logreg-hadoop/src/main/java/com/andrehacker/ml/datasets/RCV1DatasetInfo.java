package com.andrehacker.ml.datasets;

import java.util.ArrayList;
import java.util.Map;

import com.google.common.collect.ImmutableMap;

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

  private static final int NUM_FEATURES = 47237;
  private static final int VECTOR_SIZE = NUM_FEATURES;
  private static final int TOTAL = 810935;
  
  private static final Map<Integer, String> labelMap = ImmutableMap.of(
      1, "CCAT",
      2, "ECAT",
      3, "GCAT",
      4, "MCAT"
      );

  private static DatasetInfo datasetInfo = new DatasetInfo(NUM_FEATURES, VECTOR_SIZE, TOTAL, new ArrayList<String>(), labelMap);
  
  public static DatasetInfo get() {
    return datasetInfo;
  }
  
}
