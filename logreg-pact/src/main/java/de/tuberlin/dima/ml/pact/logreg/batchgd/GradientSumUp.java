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

import java.util.Iterator;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import de.tuberlin.dima.ml.pact.types.PactVector;
import eu.stratosphere.pact.common.contract.ReduceContract.Combinable;
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactInteger;

/**
 * Sums up the gradients from subsets of the data to a global gradient. This is
 * possible because the gradient is in our case the sum of the gradients of the
 * individual data points.
 * 
 * The reduce method is combinable. However we don't implement combine because
 * the default behaviour is to use the reduce method as combine, which is fine
 * here.
 */
@Combinable
public class GradientSumUp extends ReduceStub {
  
  // Has to be similar because we want to use the reduce method as combiner 
  public static final int IDX_MODEL_KEY = ApplyGradient.IDX_INPUT2_MODEL_KEY;
  public static final int IDX_GRADIENT_PART = ApplyGradient.IDX_INPUT2_GRADIENT;

  @Override
  public void reduce(Iterator<PactRecord> gradientParts,
      Collector<PactRecord> out) throws Exception {
    PactRecord first = gradientParts.next();
    PactInteger modelKey = first.getField(IDX_MODEL_KEY, PactInteger.class);
    Vector gradient = first.getField(IDX_GRADIENT_PART, PactVector.class).getValue();

    while (gradientParts.hasNext()) {
      Vector gradientPart = gradientParts.next()
          .getField(IDX_GRADIENT_PART, PactVector.class).getValue();
      gradient.assign(gradientPart, Functions.PLUS);
    }
    
    PactRecord recordOut = new PactRecord();
    recordOut.setField(ApplyGradient.IDX_INPUT2_MODEL_KEY, modelKey);
    recordOut.setField(ApplyGradient.IDX_INPUT2_GRADIENT, new PactVector(gradient));
    out.collect(recordOut);
  }

}
