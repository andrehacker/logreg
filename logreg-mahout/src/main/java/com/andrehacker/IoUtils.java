package com.andrehacker;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;

public class IoUtils {
  
  /**
   * Try to open file for reading from Resources.
   * If there is no such resource, try to read from filesystem. 
   * 
   * @param inputFile
   * @return
   * @throws IOException
   */
  static BufferedReader open(String inputFile) throws IOException {
    InputStream in;
    try {
      in = Resources.getResource(inputFile).openStream();
    } catch (IllegalArgumentException e) {
      in = new FileInputStream(new File(inputFile));
    }
    return new BufferedReader(new InputStreamReader(in, Charsets.UTF_8));
  }

}
