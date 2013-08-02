package de.tuberlin.dima.ml.pact.io;

import java.io.IOException;

import eu.stratosphere.nephele.configuration.Configuration;
import eu.stratosphere.nephele.template.GenericInputSplit;
import eu.stratosphere.pact.common.contract.GenericDataSource;
import eu.stratosphere.pact.common.io.statistics.BaseStatistics;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.generic.io.InputFormat;

/**
 * Base class for Input Formats that have only a single split and but instead
 * emit a sequence of records. The process of creating the records can be
 * made dynamic to a certain extend by overriding configure (i.e. by sending
 * simple parameters to the InputFormat).<br/>
 * 
 * Use {@link GenericDataSource} to get a Contract that can be used as an job
 * input. <br/>
 * 
 * It is currently not supported to send more complex objects (implementing
 * Value) during Job construction. This would, however, be a nice feature,
 * because sometimes we have the objects to use as input available at runtime
 * during job construction and we don't want to use the workaround to write to a
 * file and use this as input.
 * 
 */
public abstract class RecordSequenceInputFormat implements InputFormat<PactRecord, GenericInputSplit> {
  
  int recordsProcessed = 0;
  
  /**
   * The implementation must return the number of records.
   * This class will call getNextRecord() this many times.
   * 
   * @return The number of records this InputFormat will produce
   */
  abstract long getNumberRecords();
  
  abstract boolean fillNextRecord(PactRecord record);
  
  /**
   * Default implementation for configure where we don't get any parameters.
   */
  @Override
  public void configure(Configuration parameters) { }

  @Override
  public final BaseStatistics getStatistics(BaseStatistics cachedStatistics)
      throws IOException {
    // When this method is called, configure was called before
    // I guess this is called once only when the job gets compiled
    
    return new BaseStatistics() {
      @Override
      public long getTotalInputSize() {
        return BaseStatistics.SIZE_UNKNOWN;
      }
      @Override
      public long getNumberOfRecords() {
        return getNumberRecords();
      }
      @Override
      public float getAverageRecordWidth() {
        return BaseStatistics.AVG_RECORD_BYTES_UNKNOWN;
      }
    };
  }

  @Override
  public final GenericInputSplit[] createInputSplits(int minNumSplits)
      throws IOException {
    // return a single split
    GenericInputSplit[] splits = new GenericInputSplit[] {new GenericInputSplit(0)};
    return splits;
  }

  @Override
  public final Class<? extends GenericInputSplit> getInputSplitType() {
    return GenericInputSplit.class;
  }

  @Override
  public final void open(GenericInputSplit split) throws IOException {
    // Nothing to do here
  }
  
  @Override
  public final boolean nextRecord(PactRecord record) throws IOException {
    ++ recordsProcessed;
    return fillNextRecord(record);
  }
  
  @Override
  public final boolean reachedEnd() throws IOException {
    return (recordsProcessed >= getNumberRecords());
  }

  @Override
  public final void close() throws IOException {
    // Nothing to do here
  }

}
