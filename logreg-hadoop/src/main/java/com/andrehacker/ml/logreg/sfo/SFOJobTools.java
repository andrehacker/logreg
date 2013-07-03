package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;

import com.google.common.io.Closeables;

public class SFOJobTools {
  
  /**
   * Makes the current base model available to mappers/reducers via hdfs
   * 
   * We use hdfs (and not distributed cache) because the base model remains
   * the same for train and test job.
   * 
   * TODO Improvement: Appending only every single added dimension would be nice
   * to reduce startup costs
   */
  static void writeBaseModel(IncrementalModel baseModel) throws IOException {
    Configuration conf = new Configuration();
    conf.addResource(new Path(GlobalJobSettings.CONFIG_FILE_PATH));
    
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer writer = null;
    try {
      writer = SequenceFile.createWriter(fs, conf, new Path(GlobalJobSettings.BASE_MODEL_PATH),
          IncrementalModelWritable.class, NullWritable.class);
      writer.append(new IncrementalModelWritable(baseModel), NullWritable.get());
    } finally {
      Closeables.close(writer, true);
    }
  }
  
  static IncrementalModel readBaseModel(Configuration conf) throws IOException {
    IncrementalModel baseModel = null;
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(GlobalJobSettings.BASE_MODEL_PATH), conf);
    try {
      IncrementalModelWritable baseModelWritable = new IncrementalModelWritable();
      // model is stored in key
      reader.next(baseModelWritable);
      baseModel = baseModelWritable.getModel();
    } finally {
      Closeables.close(reader, true);
    }
    
    return baseModel;
  }

}
