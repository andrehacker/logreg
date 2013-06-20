package com.andrehacker.ml.sfo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * Used for the intermediate output of SFO Mapper
 * Transmits a vector and two values
 * 
 * Uses a VectorWritable instance internally
 */
public class SFOIntermediateWritable implements Writable
{
  private double xid;
  private int yi;
  private double pi;
  
  public SFOIntermediateWritable() { }
  
  public SFOIntermediateWritable(double xid, int label, double prob) {
    this.xid = xid;
    this.yi = label;
    this.pi = prob;
  }
  
  public void readFields(DataInput in) throws IOException {
    xid = in.readDouble();
    yi = in.readInt();
    pi = in.readDouble();
  }

  public void write(DataOutput out) throws IOException {
    out.writeDouble(xid);
    out.writeInt(yi);
    out.writeDouble(pi);
  }
  
  public int getLabel() {
    return yi;
  }
  
  public double getXid() {
    return xid;
  }
  
  public double getPi() {
    return pi;
  }
  
  public void setLabel(int label) {
    this.yi = label;
  }
  
  public void setPi(double pi) {
    this.pi = pi;
  }
  
  public void setXid(double xid) {
    this.xid = xid;
  }
}
