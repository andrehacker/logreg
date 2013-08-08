package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.pact.logreg.sfo.udfs.MatchGainsAndCoefficients;
import eu.stratosphere.nephele.fs.FSDataInputStream;
import eu.stratosphere.nephele.fs.FileStatus;
import eu.stratosphere.nephele.fs.FileSystem;
import eu.stratosphere.nephele.fs.Path;

public final class SFOToolsPact {

  /**
   * 
   * @param outputPath Path with 
   * @param gains Empty list
   * @param coefficients List with numFeatures empty elements. Create with Arrays.asList(new Double[numFeatures])
   */
  public static List<FeatureGain> readGainsAndCoefficients(String outputPath) throws IOException, URISyntaxException {
    
    List<FeatureGain> gains = Lists.newArrayList();
    FileSystem fs = FileSystem.get(new URI(outputPath));
    
    FileStatus[] statusList = fs.listStatus(new Path(outputPath));
    for (FileStatus status : statusList) {
      System.out.println("Output: " + status.getPath());
      FSDataInputStream stream = fs.open(status.getPath());
      BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
      String line = null;
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.split(" ");
        gains.add(new FeatureGain(
            Integer.parseInt(tokens[MatchGainsAndCoefficients.IDX_OUT_DIMENSION]),
            Double.parseDouble(tokens[MatchGainsAndCoefficients.IDX_OUT_GAIN]),
            Double.parseDouble(tokens[MatchGainsAndCoefficients.IDX_OUT_COEFFICIENT])));
      }
    }
    
    return gains;
  }

}
