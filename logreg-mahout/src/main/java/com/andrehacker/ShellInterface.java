package com.andrehacker;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

public class ShellInterface {

  public static void main(String[] args) {

    List<String> predictorNames = Lists.newArrayList(new String[] {
       //"x", "y", "shape", "k", "k0", "xx", "xy", "yy", "a", "b", "c", "bias"
       "x", "y", "shape", "a", "b", "c"
    });
    
    LinRegTrainer trainer = new LinRegTrainer();
    try {
      trainer.train("donut.csv");
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    
//    double initialWeight = 1;
//    LogRegTrainer logTrainer = new LogRegTrainer();
//    try {
//      logTrainer.trainBatchGD("donut.csv", predictorNames, 100, 0.1d, initialWeight);
//    } catch (IOException e1) {
//      e1.printStackTrace();
//    }
    
  }
  
  private void dummy() {
    RandomAccessSparseVector randVec = new RandomAccessSparseVector(2);
    SequentialAccessSparseVector seqVec = new SequentialAccessSparseVector(2);
    randVec.set(0, 0.5);
    randVec.set(1, 1.5);
    seqVec.set(0, 0.5);
    seqVec.set(1, 1.5);
    //System.out.println(randVec.toString());
    //System.out.println(seqVec.toString());
  }

}
