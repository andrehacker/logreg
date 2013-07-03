package com.andrehacker.ml.logreg.sfo;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;

import com.google.common.io.Closeables;

public class SFOJobTools {
  
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
