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
    // TODO This doesnt work for YARN
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

  /**
   * Create a Configuration object with all properties defined in the hadoop config folder. 
   */
  public static Configuration createConfigurationFromConfDir(String confDir) {
    Configuration conf = new Configuration();
    conf.addResource(new Path(confDir + "/core-site.xml"));
    conf.addResource(new Path(confDir + "/mapred-site.xml"));
    conf.addResource(new Path(confDir + "/hdfs-site.xml"));
    conf.addResource(new Path(confDir + "/yarn-site.xml"));
    System.out.println("Add resource: " + confDir + "/yarn-site.xml");
    return conf;
  }
  
  public static void addJarToConfiguration(Configuration configuration, String jarPath) {
    if (!"".equals(jarPath))
      configuration.set("mapred.jar", jarPath);
  }
  
  @Deprecated
  public static Configuration createConfiguration(String hdfsAddress, String jobtrackerAddress) {
    return createConfiguration(hdfsAddress, jobtrackerAddress, "");
  }
  
  @Deprecated
  public static Configuration createConfiguration(String hdfsAddress, String jobtrackerAddress, String jarPath) {
    // TODO: These properties no longer work for yarn (I guess)
    Configuration conf = new Configuration();
    if (!"".equals(jobtrackerAddress))
        conf.set("mapred.job.tracker", jobtrackerAddress);
    if (!"".equals(hdfsAddress))
      conf.set("fs.default.name", hdfsAddress);
    if (!"".equals(jarPath))
      conf.set("mapred.jar", jarPath);
    return conf;
  }
  
  
}
