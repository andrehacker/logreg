package com.andrehacker.ml;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * 
 * Notes
 * - Mahout has it's own IntPairWritable
 * - Didn't find a generic pair. Implementing hashCode() might be tricky
 * - See discussion: http://comments.gmane.org/gmane.comp.apache.mahout.devel/8711
 * - Other implementation: https://github.com/sagemintblue/cmu-commons/tree/master/hadoop/src/main/java/cmu/edu/commons/hadoop/tuple
 */
public class DoublePairWritable implements Writable
{
  private double first;
  private double second;

  public DoublePairWritable() {
  }

  public DoublePairWritable(double first, double second) {
    this.first = first;
    this.second = second;
  }
  
  public void readFields(DataInput in) throws IOException {
    first = in.readDouble();
    second = in.readDouble();
  }

  public void write(DataOutput out) throws IOException {
    out.writeDouble(first);
    out.writeDouble(second);
  }
  
  // TODO Implement hashCode()
  
  public double getFirst() {
    return first;
  }
  
  public double getSecond() {
    return second;
  }
  
  public void setFirst(double first) {
    this.first = first;
  }
  
  public void setSecond(double second) {
    this.second = second;
  }
  
}
