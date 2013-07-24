package de.tuberlin.dima.ml.pact.io;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.nephele.configuration.Configuration;
import eu.stratosphere.pact.common.io.DelimitedInputFormat;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;
import eu.stratosphere.pact.common.type.base.parser.DecimalTextIntParser;

public class WeightVectorInputFormat extends DelimitedInputFormat {

  // --------------------------------------- Config  ---------------------------------------------

  public static final String NUM_FEATURES = "libsvm.num_features";

  private static final int NUM_FEATURES_UNDEFINED = -1;

  private int numFeatures;

  // --------------------------------------- Output ---------------------------------------------

  private final PactVector weights = new PactVector();

  @Override
  public void configure(Configuration parameters) {
    super.configure(parameters);

    // num features
    this.numFeatures = parameters.getInteger(NUM_FEATURES, NUM_FEATURES_UNDEFINED);
    if (this.numFeatures == NUM_FEATURES_UNDEFINED) {
      throw new IllegalArgumentException("Please specify the number of features for the vector");
    }
  }

  @Override
  public boolean readRecord(PactRecord target, byte[] bytes, int offset,
      int numBytes) {

    final int limit = offset + numBytes;

    DecimalTextIntParser intParser = new DecimalTextIntParser();
    PactInteger initial = new PactInteger();

    intParser.parseField(bytes, offset, limit, ' ', initial);

    System.out.println("WeightVectorInputFormat: numFeatures="
        + this.numFeatures + " initialValue=" + initial.getValue());

    Vector vector = initial.getValue() == 0 ? new RandomAccessSparseVector(this.numFeatures)
        : new DenseVector(this.numFeatures).assign(initial.getValue());

    this.weights.setValue(vector);
    target.setField(0, this.weights);

    return true;
  }
}
