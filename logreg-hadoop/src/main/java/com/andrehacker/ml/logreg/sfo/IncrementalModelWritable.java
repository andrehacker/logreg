package com.andrehacker.ml.logreg.sfo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;

public class IncrementalModelWritable implements Writable {
  
  IncrementalModel model;
  
  public IncrementalModelWritable() { }
  
  public IncrementalModelWritable(IncrementalModel model) {
    this.model = model;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    List<Integer> used = model.getUsedDimensions();
    out.writeInt(used.size());
    for (int i=0; i<used.size(); ++i) {
      out.writeInt(used.get(i));
    }
    
    VectorWritable w = new VectorWritable(model.getW());
    w.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int numberUsed = in.readInt();
    List<Integer> used = Lists.newArrayListWithCapacity(numberUsed);
    for (int i=0; i<numberUsed; ++i) {
      used.add(in.readInt());
    }

    Vector w = VectorWritable.readVector(in);
    
    model = new IncrementalModel(used, w);
  }
  
  public IncrementalModel getModel() {
    return model;
  }

}
