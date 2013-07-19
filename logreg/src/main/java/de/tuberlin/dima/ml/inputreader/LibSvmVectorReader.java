package de.tuberlin.dima.ml.inputreader;

import org.apache.mahout.math.Vector;

public class LibSvmVectorReader {
  
  private LibSvmVectorReader() {};
  
  /**
   * Reads a single line of 
   * Line format:   label-id feature-id-1:val-1 feature-id-2:val-2 ...
   * where label-id is -1 or 1
   * 
   * @return Label of the data, either 0 (negative class) or 1 (positive class)
   */
  public static short readVector(Vector v, String line) {
    // Same as rcv1-v2 format, except that the first number is the label instead of the document-id
    int label = RCV1VectorReader.readVector(v, line);
    return (label == -1) ? (short)0 : (short)1;
  }
    
}
