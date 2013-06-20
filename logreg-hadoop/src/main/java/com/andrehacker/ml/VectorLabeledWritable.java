package com.andrehacker.ml;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * For transfer of a vector and a numeric label.
 * Usefull for labeled data (supervised learning)
 * where we have to transmit not only the VectorWritable, but also the label 
 * Uses a VectorWritable instance internally
 */
public class VectorLabeledWritable implements Writable
{
  private VectorWritable vector;
  private int label;
  
  public VectorLabeledWritable() {
    this.vector = new VectorWritable();
  }

  public VectorLabeledWritable(VectorWritable vector, int label) {
    this.vector = vector;
    this.label = label;
  }
  
  public void readFields(DataInput in) throws IOException {
    vector.readFields(in);
    label = in.readInt();
  }

  public void write(DataOutput out) throws IOException {
    vector.write(out);
    out.writeInt(label);
  }
  
  @Override
  public int hashCode() {
    return vector.hashCode();
  }
  
  public int getLabel() {
    return label;
  }
  
  public Vector getVector() {
    return vector.get();
  }
  
  public void setLabel(int label) {
    this.label = label;
  }
  
  public void setVector(Vector vector) {
    this.vector.set(vector);
  }
}
