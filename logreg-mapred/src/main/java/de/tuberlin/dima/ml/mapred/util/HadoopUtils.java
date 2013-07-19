package de.tuberlin.dima.ml.mapred.util;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.VectorWritable;

public class HadoopUtils {

  public static boolean detectLocalMode(Configuration conf) {
    String jobTracker = conf.get("mapred.job.tracker");
    if (jobTracker == null)
      return true;
    return conf.get("mapred.job.tracker").equals("local");
  }
  
  public static void writeVectorToDistCache(Configuration conf, VectorWritable vector, Path cachePath) throws IOException {
    // Write sequence file to local filesystem
    SequenceFile.Writer writer = SequenceFile.createWriter(FileSystem.getLocal(conf), conf,
        cachePath, NullWritable.class, VectorWritable.class);
    
    writer.append(NullWritable.get(), vector);
    writer.close();

    // Add local file to distributed cache
    DistributedCache.addCacheFile(cachePath.toUri(), conf);
  }
  
}
