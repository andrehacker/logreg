package de.tuberlin.dima.ml.pact;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

import eu.stratosphere.nephele.configuration.ConfigConstants;
import eu.stratosphere.nephele.configuration.Configuration;
import eu.stratosphere.nephele.configuration.GlobalConfiguration;
import eu.stratosphere.pact.client.nephele.api.Client;
import eu.stratosphere.pact.client.nephele.api.ErrorInPlanAssemblerException;
import eu.stratosphere.pact.client.nephele.api.PactProgram;
import eu.stratosphere.pact.client.nephele.api.ProgramInvocationException;

/**
 * Runs a Stratosphere job (jar) using the pact.client api.
 * Allows the job to be
 * executed on a cluster without using the cli
 */
public class JobRunner {

  /**
   * Runs a Stratosphere job from a jar file using the pact client api
   * 
   * @param jarPath
   *          Path to the jar to execute
   * @param jobArgs
   *          Arguments for the job
   * @param configPath
   *          Optional: The path (or file) containing the cluster-configuration
   *          (e.g. jobmanager). If empty, defaults will be applied (localhost)
   */
  public static void run(String jarPath, String[] jobArgs, String configPath) {

    File jar = new File(jarPath);

    try {

      // PactProgram wraps plan related function
      // Pass (classname and) arguments here
      // PactProgram(File jarFile, String... args)
      // PactProgram(File jarFile, String className, String... args)
      PactProgram prog = new PactProgram(jar, jobArgs);

      // Client compiles and submits pact programs to nephele cluster
      // Configuration is the nephele configuration
      // - Only Jobmanager address and port are read from configuration file
      // - Reads all files in specified directory
      // - results in warning, because nephele-plugins.xml does not start with
      // configuration
      if (configPath != "") {
        GlobalConfiguration.loadConfiguration(configPath);
      }
      Configuration config = GlobalConfiguration.getConfiguration();
      if (configPath == "") {
        // Apply defaults (is there another way to get a default configuration?)
        config.setString(ConfigConstants.JOB_MANAGER_IPC_ADDRESS_KEY, "127.0.0.1");
        config.setInteger(ConfigConstants.JOB_MANAGER_IPC_PORT_KEY, ConfigConstants.DEFAULT_JOB_MANAGER_IPC_PORT);
      }
      Client client = new Client(config);

      // Client.run(...)
      //  - compiles the PactProgram to a OptimizedPlan using
      // PactCompiler.compile()
      //  - compiles OptimizedPlan to JogGraph using
      // JogGraphGenerator.compileJobGraph(plan)
      //  - Uses ...nephele.client.JobClient.submitJob()/submitJobAndWait()
      //    to submit Job to nephele
      // unfortunately jobDuration returned from submitJobAndWait() is not
      // returned
      // probably because it is only returned if we wait for completion
      
      // Here we get: cannot access OptimizedPlan class file for OptimizedPlan not found
      // PROBLEM: Looks for short name, not for full qualified name!
      // This cannot be compiled from maven / javac. Works only if Eclipse built the class file for this class before
      //  http://stackoverflow.com/questions/9693889/maven-class-file-for-not-found-compilation-error
      //  http://maven.40175.n5.nabble.com/Maven-project-can-t-access-class-file-td4599976.html
      client.run(prog, true);
      System.out.println("Job submitted succesfully");

    } catch (ProgramInvocationException e) {
      System.out.println(e.toString());
    } catch (ErrorInPlanAssemblerException e) {
      System.out.println(e.toString());
    }

  }

  public static void runFromPropertyFile(String propertyFile) {

    Properties prop = new Properties();
    try {
      prop.load(new FileInputStream(propertyFile));
    } catch (IOException e) {
      System.out.println(e.getMessage());
      System.exit(0);
    }
    int numArgs = Integer.parseInt(prop.getProperty("job.numargs", ""));
    System.out.println("Number of parameters: " + numArgs);
    String[] params = new String[numArgs];
    for (int i = 0; i < numArgs; ++i) {
      params[i] = prop.getProperty("job.arg" + (i + 1), "");
      System.out.println("param" + i + ": " + params[i]);
    }

    String jobJar = prop.getProperty("job.jar", "");

    String configPath = prop.getProperty("system.configpath", "");

    run(jobJar, params, configPath);
  }

}
