/***********************************************************************************************************************
 *
 * Copyright (C) 2013 by the Stratosphere project (http://stratosphere.eu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 **********************************************************************************************************************/
package de.tuberlin.dima.ml.pact.logreg.batchgd;

import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.logreg.LogRegMath;
import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.CrossStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;

/**
 * Computes the gradient for the logistic regression function for a given model
 * (weight vector) and a partition of the data.
 */
public class ComputeGradientPart extends CrossStub {
  
  private final PactRecord recordOut = new PactRecord();
  private final PactVector vectorOut = new PactVector();
  private static final PactInteger one = new PactInteger(1);

	@Override
	public void cross(PactRecord trainingVector, PactRecord model, Collector<PactRecord> out) throws Exception {

        int y = trainingVector.getField(0, PactInteger.class).getValue();
		Vector xTrain = trainingVector.getField(1, PactVector.class).getValue();
		Vector w = model.getField(0, PactVector.class).getValue();
		
//		System.out.println("Training vector: size=" + xTrain.size() + " non-zeros=" + xTrain.getNumNonZeroElements());
//        System.out.println("Model: size=" + w.size() + " non-zeros=" + w.getNumNonZeroElements());

        Vector gradient = LogRegMath.computePartialGradient(xTrain, w, y);
        vectorOut.setValue(gradient);
        recordOut.setField(BatchGDJob.ID_SUM_IN_KEY, one);
        recordOut.setField(BatchGDJob.ID_SUM_IN_GRADIENT_PART, vectorOut);
        out.collect(recordOut);
	}

}
