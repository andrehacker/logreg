package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.andrehacker.ml.logreg.LogisticRegression;
import com.andrehacker.ml.writables.DoublePairWritable;
import com.google.common.io.Closeables;

public class EvalMapper extends Mapper<IntWritable, VectorWritable, IntWritable, DoublePairWritable> {
  
  private static IntWritable outputKey = new IntWritable();
  private static DoublePairWritable outputValue = new DoublePairWritable();
  
  private LogisticRegression logreg = new LogisticRegression();
  
  private IncrementalModel model;
  
  List<Double> coefficients;
  
//  private static AdaptiveLogger log = new AdaptiveLogger(
//      SFOJob.RUN_LOCAL_MODE, Logger.getLogger(SFOMapper.class.getName()), Level.DEBUG); 
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    // TODO Read Base Model!
    model = new IncrementalModel((int)SFOJob.modelInfo.getVectorSize());
    
    // Read trained coefficients into map: dimension -> coefficient
    coefficients = Arrays.asList(new Double[(int)SFOJob.modelInfo.getVectorSize()+1]);
    
    Path dir = new Path(SFOJobTest.TRAIN_OUTPUT_PATH);
    FileSystem fs = FileSystem.get(context.getConfiguration());
    FileStatus[] statusList = fs.listStatus(dir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        if (path.getName().startsWith("part-r")) return true;
        else return false;
      }
    });
    System.out.println("Read trained coefficients from " + statusList.length + " files");
    for (FileStatus status : statusList) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), context.getConfiguration());
      try {
        IntWritable dimension = new IntWritable();
        DoubleWritable coefficient = new DoubleWritable();
        while (reader.next(dimension, coefficient)) {
          coefficients.set(dimension.get(), coefficient.get());
          System.out.println(dimension.get() + ": " + coefficient.get());
        }
      } finally {
        Closeables.close(reader, true);
      }
    }
  }
  
  @Override
  public void map(IntWritable y, VectorWritable xi, Context context) throws IOException, InterruptedException {

    // Emit prediction for new and old model
    // 1) Compute prediction for current x_i using the base model (without new coefficient)
    // 2) Compute prediction
    double piBase = logreg.predict(xi.get(), model.getW(), SFOJob.INTERCEPT);
      // New feature?
    for (Vector.Element feature : xi.get().nonZeroes()) {
      if (! model.getUsedDimensions().contains(feature.get())) {
        int dim = feature.index();
        
        model.getW().set(dim, coefficients.get(dim));
        double piNew = LogisticRegression.logisticFunction(xi.get().dot(model.getW()) + SFOJob.INTERCEPT);
        model.getW().set(dim, 0d);    // reset to base model
        
        outputKey.set(feature.index());
        outputValue.setFirst(piBase);
        outputValue.setSecond(piNew);
        context.write(outputKey, outputValue);
      }
    }
  }
}