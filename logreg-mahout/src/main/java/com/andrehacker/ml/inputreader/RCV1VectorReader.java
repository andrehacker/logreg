package com.andrehacker.ml.inputreader;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.regex.Pattern;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import com.andrehacker.ml.AbstractVectorReader;
import com.andrehacker.ml.util.MLUtils;
import com.google.common.base.Splitter;

public class RCV1VectorReader implements AbstractVectorReader {
  
  // For TRAC Format (see below)
  private static Splitter TRAC_SPLITTER = Splitter.on(Pattern.compile("[ :]"))
      .trimResults()
      .omitEmptyStrings();
  
  private static Splitter SPACE_SPLITTER = Splitter.on(Pattern.compile(" "))
      .trimResults()
      .omitEmptyStrings();
  
  /**
   * Converts single line from RCV1 vector file to sparse vector
   * Line format: document-id  feature-id-1:val feature-id-2:val ...
   * @return document-id
   */
  public int readVector(Vector v, String line) {
    Iterator<String> iter = TRAC_SPLITTER.split(line).iterator();
    int docId = Integer.parseInt(iter.next());
    int featureId;
    Double featureVal;
    while (iter.hasNext()) {
      featureId = Integer.parseInt(iter.next());
      featureVal = Double.parseDouble(iter.next());
      v.set(featureId, featureVal);
    }
    return docId;
  }

  /**
   * TODO Refactoring: Get rid of this method. Use single files instead
   * Reads targets from text file into vectors
   * 1 vector for each main category
   * CCAT(Corporate/Industrial)
   * ECAT(Economics)
   * GCAT(Government/Social)
   * MCAT(Markets)
   */
  public void readTargets(Vector yC, Vector yE, Vector yG, Vector yM) throws IOException {
    // Line format: ECAT 2286 1
    BufferedReader reader = MLUtils.open("/home/andre/dev/datasets/RCV1-v2/rcv1-v2.topics.qrels");
    String line;
    String cat;
    while ((line = reader.readLine()) != null) {
      Iterator<String> iter = SPACE_SPLITTER.split(line).iterator();
      cat = iter.next();
      if (cat.equals("CCAT")) {
        yC.set(Integer.parseInt(iter.next()), 1d);
      }
      if (cat.equals("ECAT")) {
        yE.set(Integer.parseInt(iter.next()), 1d);
      }
      if (cat.equals("GCAT")) {
        yG.set(Integer.parseInt(iter.next()), 1d);
      }
      if (cat.equals("MCAT")) {
        yM.set(Integer.parseInt(iter.next()), 1d);
      }
    }
    reader.close();
  }
  
  /**
   * Reads targets from text file into vectors
   * Assumes that text file only contains labels for one Vector
   */
  public Vector readTarget(int count, String filename, String categoryName) throws IOException {
    
    Vector v = new DenseVector(count);
    
    // Line format: ECAT 2286 1
    BufferedReader reader = MLUtils.open(filename);
    String line;
    String cat;
    while ((line = reader.readLine()) != null) {
      Iterator<String> iter = SPACE_SPLITTER.split(line).iterator();
      cat = iter.next();
      if (cat.equals(categoryName)) {
        v.set(Integer.parseInt(iter.next()), 1d);
      }
    }
    reader.close();
    
    return v;
  }
}
