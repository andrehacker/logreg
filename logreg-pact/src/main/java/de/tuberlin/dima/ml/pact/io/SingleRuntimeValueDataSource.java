package de.tuberlin.dima.ml.pact.io;

import java.io.DataOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import eu.stratosphere.nephele.fs.FSDataOutputStream;
import eu.stratosphere.nephele.fs.FileSystem;
import eu.stratosphere.nephele.fs.Path;
import eu.stratosphere.pact.common.contract.GenericDataSource;
import eu.stratosphere.pact.common.type.Value;

public class SingleRuntimeValueDataSource extends GenericDataSource<SingleRuntimeValueInputFormat> {

  /**
   * Using this DataSource you can use a single object (inheriting from
   * {@link Value}) as a DataSource. It will automatically serialize the object
   * to a file at any folder (that must be accessible from all nodes). The
   * related {@link SingleRuntimeValueInputFormat} will deserialize it and emit
   * a single record with just a single value. The contract is defined to have
   * only a single split.<br/>
   * 
   * TODO Improvement: It would be nice to just ask for the folder path and
   * automatically create a unique file name.
   * 
   * @param value The actual instance we want to use as input
   * @param uniqueTmpFilePath
   *          This file path must be accessible from JobManager and
   *          Taskmanagers. Everytime you use this contract you must define a
   *          unique path. E.g. hdfs://my-namenode/tmp/unique-name-for-my-object
   * @throws IOException
   */
  public SingleRuntimeValueDataSource(Value value, String uniqueTmpFilePath) throws IOException {
    
    super(SingleRuntimeValueInputFormat.class);

    // Send class name as parameter
    this.parameters.setString(SingleRuntimeValueInputFormat.CONF_KEY_VALUE_CLASS, value.getClass().getName());
    System.out.println("SingleRuntimeValueDataSource Class name: " + value.getClass().getName());
    
    // Serialize value to file (on the currently used filesystem)
    URI filePathUri = null;
    try {
      filePathUri = new URI(uniqueTmpFilePath);
    } catch (URISyntaxException e) {
      e.printStackTrace(); return;
    }
    
    FileSystem fs = FileSystem.get(filePathUri);
    FSDataOutputStream stream = fs.create(new Path(uniqueTmpFilePath), true);
    DataOutputStream inStream = new DataOutputStream(stream);
    
    value.write(inStream);
    
    stream.close();

    // Send location
    this.parameters.setString(SingleRuntimeValueInputFormat.CONF_KEY_FILE_PATH, uniqueTmpFilePath);
  }

}
