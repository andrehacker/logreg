package de.tuberlin.dima.ml.mapred.writables;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * For transfer of a vector and multiple numeric label.
 * Usefull for labeled data (supervised learning)
 * where we have to transmit not only the VectorWritable, but also the label 
 * Uses a VectorWritable instance internally
 * 
 * TODO Performance This is not very efficient, since it is wrapping around VectorWritables
 * Better to write a VectorPairWritable
 */
public class VectorMultiLabeledWritable implements Writable
{
  private VectorWritable vector;
  private VectorWritable labels;
  
  public VectorMultiLabeledWritable() {
    this.vector = new VectorWritable();
    this.labels = new VectorWritable();
  }

  public VectorMultiLabeledWritable(VectorWritable vector, VectorWritable labels) {
    this.vector = vector;
    this.labels = labels;
  }
  
  public void readFields(DataInput in) throws IOException {
    vector.readFields(in);
    labels.readFields(in);
  }

  public void write(DataOutput out) throws IOException {
    vector.write(out);
    labels.write(out);
  }
  
  @Override
  public int hashCode() {
    return vector.hashCode();
  }
  
  public Vector getLabels() {
    return labels.get();
  }
  
  public Vector getVector() {
    return vector.get();
  }
  
  public void setLabels(Vector labels) {
    this.labels.set(labels);
  }
  
  public void setVector(Vector vector) {
    this.vector.set(vector);
  }
}
