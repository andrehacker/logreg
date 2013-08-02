package de.tuberlin.dima.ml.pact.io;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.nephele.configuration.Configuration;
import eu.stratosphere.pact.common.io.DelimitedInputFormat;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactDouble;
import eu.stratosphere.pact.common.type.base.PactInteger;
import eu.stratosphere.pact.common.type.base.parser.DecimalTextDoubleParser;
import eu.stratosphere.pact.common.type.base.parser.DecimalTextIntParser;

/**
 * Read file in libsvm format.
 * The first numbers are a comma-separated list of the labels. After two spaces follows a sparse encoding of the feature
 * vector, e.g. '9,33,62,60,70  268:0.059 440:0.031 577:0.064'
 * If the labels is equal to the positive class label, the label will be set to 1, otherwise it will be 0.
 */
public class LibsvmBinaryInputFormat extends DelimitedInputFormat {

	// ------------------------------------- Config Keys ------------------------------------------

	public static final String CONF_KEY_POSITIVE_CLASS = "libsvm.positive_class";

	public static final String CONF_KEY_NUM_FEATURES = "libsvm.num_features";

	// --------------------------------------- Config ---------------------------------------------

	private static final char DELIMITER = ' ';

	private static final char DELIMITER_LABEL = ',';

	private static final char DELIMITER_FEATURE = ':';

	private static final int POSITIVE_CLASS_UNDEFINED = -1;

	private static final int NUM_FEATURES_UNDEFINED = -1;

	// --------------------------------------- Output ---------------------------------------------

	private final PactInteger label = new PactInteger();

	private final PactVector features = new PactVector();

	// -------------------------------------- Internal --------------------------------------------

	private PactInteger index = new PactInteger();

	private PactDouble weight = new PactDouble();

	private DecimalTextIntParser intParser = new DecimalTextIntParser();

	private DecimalTextDoubleParser doubleParser = new DecimalTextDoubleParser();

	private int positiveClass;
	
	private int numFeatures;

	// --------------------------------------------------------------------------------------------

	@Override
	public void configure(Configuration parameters) {
		super.configure(parameters);

		// target class
		this.positiveClass = parameters.getInteger(CONF_KEY_POSITIVE_CLASS, POSITIVE_CLASS_UNDEFINED);
		if (this.positiveClass == POSITIVE_CLASS_UNDEFINED) {
			throw new IllegalArgumentException("Please specify the positive class id");
		}

		// num features
		this.numFeatures = parameters.getInteger(CONF_KEY_NUM_FEATURES, NUM_FEATURES_UNDEFINED);
		if (this.numFeatures == NUM_FEATURES_UNDEFINED) {
			throw new IllegalArgumentException("Please specify the number of features for the vector");
		}
	}

	@Override
	public boolean readRecord(PactRecord target, byte[] bytes, int offset, int numBytes) {

		final int limit = offset + numBytes;
		int readPos = offset;

		// ------------------------------------
		// 1. parse labels
		// ------------------------------------

		// move behind labels
		while (bytes[readPos++] != DELIMITER) {
		}

		boolean isPositive = false;

		int currentOffset = offset;
		while (currentOffset < readPos - 1) {
			currentOffset = this.intParser.parseField(bytes, currentOffset, readPos - 1, DELIMITER_LABEL, this.label);

			if (this.label.getValue() == this.positiveClass) {
				isPositive = true;

				break;
			}
		}

		this.label.setValue(isPositive ? 1 : 0);
		target.setField(0, this.label);

		// ------------------------------------
		// 2. parse features
		// ------------------------------------
		Vector vector = new SequentialAccessSparseVector(this.numFeatures);

		currentOffset = readPos + 1;

		while (currentOffset < limit) {

			currentOffset = this.intParser.parseField(bytes, currentOffset, limit, DELIMITER_FEATURE, this.index);

			currentOffset = this.doubleParser.parseField(bytes, currentOffset, limit, DELIMITER, this.weight);

			vector.setQuick(this.index.getValue(), this.weight.getValue());
		}

		this.features.setValue(vector);
		target.setField(1, this.features);

		return true;
	}
}
