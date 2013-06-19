package com.andrehacker.ml.sfo;

import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import com.andrehacker.ml.sfo.SFOJob;

public class SFOJobTest {

  @Test
  public void test() throws Exception {
//    String[] args = new String[] { inputPath, outputPath };
    ToolRunner.run(new SFOJob(), null);
    System.out.println("Done");
  }

}
