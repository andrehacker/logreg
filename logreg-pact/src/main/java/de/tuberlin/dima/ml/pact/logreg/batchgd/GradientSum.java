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
import eu.stratosphere.pact.common.stubs.Collector;
import eu.stratosphere.pact.common.stubs.ReduceStub;
import eu.stratosphere.pact.common.type.PactRecord;

/**
 * @author uce
 *
 */
public class GradientSum extends ReduceStub {

	/*
	 * (non-Javadoc)
	 * @see eu.stratosphere.pact.common.stubs.ReduceStub#reduce(java.util.Iterator, eu.stratosphere.pact.common.stubs.Collector)
	 */
	@Override
	public void reduce(Iterator<PactRecord> gradientParts, Collector<PactRecord> out) throws Exception {
	  
	  Vector gradient = gradientParts.next().getField(BatchGDJob.ID_SUM_IN_GRADIENT_PART, PactVector.class).getValue();
	  
	  while (gradientParts.hasNext()) {
	    Vector gradientPart = gradientParts.next().getField(BatchGDJob.ID_SUM_IN_GRADIENT_PART, PactVector.class).getValue();
	    gradient.assign(gradientPart, Functions.PLUS);
	  }
	  
	  PactRecord recordOut = new PactRecord(new PactVector(gradient));
	  out.collect(recordOut);
	}

}
