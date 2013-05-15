package com.andrehacker;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;

public class LogRegShell {

  /**
   * @param args
   */
  public static void main(String[] args) {
    RandomAccessSparseVector randVec = new RandomAccessSparseVector(2);
    SequentialAccessSparseVector seqVec = new SequentialAccessSparseVector(2);
    randVec.set(0, 0.5);
    randVec.set(1, 1.5);
    seqVec.set(0, 0.5);
    seqVec.set(1, 1.5);
    //System.out.println(randVec.toString());
    //System.out.println(seqVec.toString());
    
    LinRegTrainer trainer = new LinRegTrainer();
    try {
      trainer.train("donut.csv");
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

}
