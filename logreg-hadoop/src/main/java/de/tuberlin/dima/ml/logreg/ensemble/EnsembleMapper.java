package de.tuberlin.dima.ml.logreg.ensemble;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.mahout.math.VectorWritable;

import de.tuberlin.dima.ml.util.AdaptiveLogger;
import de.tuberlin.dima.ml.writables.IDAndLabels;
import de.tuberlin.dima.ml.writables.VectorMultiLabeledWritable;

public class EnsembleMapper extends Mapper<IDAndLabels, VectorWritable, IntWritable, VectorMultiLabeledWritable> {
  
  Random random = new Random();

  int numberReducers;

  IntWritable curPartition = new IntWritable();
  
  private static AdaptiveLogger log = new AdaptiveLogger(
      Logger.getLogger(EnsembleMapper.class.getName()), 
      Level.DEBUG); 
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    log.debug("Map Setup");
    
    numberReducers = Integer.parseInt(context.getConfiguration().get("mapred.reduce.tasks"));
    log.debug("- Number partitions (=number reducers): " + numberReducers);
  }

  private static VectorMultiLabeledWritable currentLabeledVector = new VectorMultiLabeledWritable();
  
  @Override
  public void map(IDAndLabels key, VectorWritable value, Context context) throws IOException, InterruptedException {
    
    // Randomly distribute to Reducers to get a random partitioning
    
    // TODO If Reducer uses SGD (Online, one-pass), we should also randomize the order within each partition!?
    
    // TODO Bug: Add custom partitioner to make sure that different partitions are sent to different reducers 
    
    curPartition.set(random.nextInt(numberReducers));
    currentLabeledVector.setVector(value.get());
    currentLabeledVector.setLabels(key.getLabels());
    context.write(curPartition, currentLabeledVector);
  }
}