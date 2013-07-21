package de.tuberlin.dima.ml.preprocess.rcv1;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.MissingOptionException;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.celebihacker.ml.preprocess.rcv1.indexing.Indexer;
import com.celebihacker.ml.preprocess.rcv1.vectorization.Vectorizer;
import com.celebihacker.ml.preprocess.rcv1.vectorization.Vectorizer.SplitType;
import com.celebihacker.ml.preprocess.rcv1.vectorization.Vectorizer.Weighting;

public class CliFrontend {

  // actions
  private static final String GENERAL_OPTS = "general";
  private static final String ACTION_INDEX = "index";
  private static final String ACTION_VECTORIZE = "vectorize";

  // general options
  private static final Option INPUT_PATH_OPT = new Option("i", "input", true, "Path to input");
  private static final Option OUTPUT_PATH_OPT = new Option("o", "output", true, "Path to output");

  // vectorize options
  private static final Option VEC_MIN_DF_OPT = new Option("m", "mindf", true,
      "Minimum document frequency for terms, e.g. 3 (default)");

  private static final Option VEC_WEIGHTING_OPT = new Option("w", "weighting", true,
      "Weighting scheme to use\n" +
          "* term weighting: none, logarithm, augmented (default)\n" +
          "* doc weighting: none, idf (default)\n" +
          "* normalization: none, cosine (default)\n" +
          "Provide first letter of scheme in above order, e.g. aic for augmented, idf, cosine");

  private static final Option VEC_SPLIT_RATIO_OPT = new Option("sr", "splitratio", true,
      "Training split ratio, e.g. 0.8 (default)");

  private static final Option VEC_SPLIT_TYPE_OPT = new Option("st", "splittype", true,
      "Split type to use\n" +
          "* date (default): split chronologically\n" +
          "* random: split randomly");

  private CommandLineParser parser;
  private Map<String, Options> options;

  private String inputPath;
  private String outputPath;

  public CliFrontend() {

    this.parser = new GnuParser();

    this.options = new HashMap<String, Options>();
    this.options.put(GENERAL_OPTS, getGeneralOptions());
    this.options.put(ACTION_VECTORIZE, getVectorizeOptions());

  }

  private Options getGeneralOptions() {
    Options opts = new Options();

    INPUT_PATH_OPT.setRequired(true);
    OUTPUT_PATH_OPT.setRequired(true);

    opts.addOption(INPUT_PATH_OPT);
    opts.addOption(OUTPUT_PATH_OPT);

    return opts;
  }

  private Options getVectorizeOptions() {
    Options opts = new Options();

    VEC_MIN_DF_OPT.setRequired(false);
    VEC_WEIGHTING_OPT.setRequired(false);
    VEC_SPLIT_RATIO_OPT.setRequired(false);
    VEC_SPLIT_TYPE_OPT.setRequired(false);

    opts.addOption(VEC_MIN_DF_OPT);
    opts.addOption(VEC_WEIGHTING_OPT);
    opts.addOption(VEC_SPLIT_RATIO_OPT);
    opts.addOption(VEC_SPLIT_TYPE_OPT);

    return opts;
  }

  private void index() {

    System.out.println("Running index\n");
    
    try {
      Indexer.index(this.inputPath, this.outputPath);
    } catch (IOException e) {
      e.printStackTrace();
    }
    
    System.out.println("Finished.");
    System.out.println("Output in " + this.outputPath);
    
  }

  private void vectorize(String[] args) {
    // defaults
    int minDf = 3;
    Weighting weighting = Weighting.AIC;
    double trainingRatio = 0.8;
    SplitType splitBy = SplitType.DATE;

    // Parse command line options
    CommandLine line = null;
    try {
      line = this.parser.parse(this.options.get(ACTION_VECTORIZE), args, false);
    } catch (Exception e) {
      e.printStackTrace();
    }

    if (line == null)
      return;

    if (line.hasOption(VEC_MIN_DF_OPT.getOpt())) {
      String val = line.getOptionValue(VEC_MIN_DF_OPT.getOpt()).toUpperCase();

      try {
        minDf = Integer.parseInt(val);
      } catch (NumberFormatException e) {
        // keep default
      }
    }

    if (line.hasOption(VEC_WEIGHTING_OPT.getOpt())) {
      String val = line.getOptionValue(VEC_WEIGHTING_OPT.getOpt()).toUpperCase();

      try {
        weighting = Weighting.valueOf(val);
      } catch (Exception e) {
        // keep default
      }
    }

    if (line.hasOption(VEC_SPLIT_RATIO_OPT.getOpt())) {
      String val = line.getOptionValue(VEC_SPLIT_RATIO_OPT.getOpt()).toUpperCase();

      try {
        trainingRatio = Double.parseDouble(val);
      } catch (NumberFormatException e) {
        // keep default
      }
    }

    if (line.hasOption(VEC_SPLIT_TYPE_OPT.getOpt())) {
      String val = line.getOptionValue(VEC_SPLIT_TYPE_OPT.getOpt()).toUpperCase();

      try {
        splitBy = SplitType.valueOf(val);
      } catch (Exception e) {
        // keep default
      }
    }

    // vectorize
    System.out.println("Running vectorize with options: \n" +
        "* minDf: " + minDf + "\n" +
        "* weighting: " + weighting + "\n" +
        "* trainingRatio: " + trainingRatio + "\n" +
        "* splitBy: " + splitBy);

    try {
      Vectorizer vectorizer = new Vectorizer(this.inputPath, minDf, weighting);
      vectorizer.vectorize(this.outputPath, splitBy, trainingRatio);
    } catch (IOException e) {
      e.printStackTrace();
    }
    System.out.println("Finished.");
    System.out.println("Output in " + this.outputPath);
  }

  private void printHelp() {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setLeftPadding(5);

    System.out.println("./rcv1-cli [action] [general_options] [action_arguments]");

    formatter.setSyntaxPrefix("  general options:");
    formatter.printHelp(" ", this.options.get(GENERAL_OPTS));

    formatter.setSyntaxPrefix("  vectorize options:");
    formatter.printHelp(" ", this.options.get(ACTION_VECTORIZE));
  }

  private String[] parseGeneralOptions(String[] args) {

    try {

      CommandLine line = this.parser.parse(this.options.get(GENERAL_OPTS), args, true);

      this.inputPath = line.getOptionValue(INPUT_PATH_OPT.getOpt());
      this.outputPath = line.getOptionValue(OUTPUT_PATH_OPT.getOpt());

      return line.getArgs();

    } catch (MissingOptionException e) {

      System.err.println("Please provide missing required arguments for input/output!");

      printHelp();
      System.exit(1);

    } catch (ParseException e) {
      e.printStackTrace();
      System.exit(1);
    }

    return null;
  }

  private void parse(String[] args) {

    // check for action
    if (args.length < 1) {
      System.err.println("ERROR: Please specify an action.");
      printHelp();
      System.exit(1);
    }

    // get action
    String action = args[0];

    // remove action from parameters
    String[] params = new String[args.length - 1];
    System.arraycopy(args, 1, params, 0, params.length);

    params = parseGeneralOptions(params);

    if (action.equals(ACTION_INDEX)) {
      index();
    } else if (action.equals(ACTION_VECTORIZE)) {
      vectorize(params);
    } else {
      System.out.println("Invalid action!");
      printHelp();
      System.exit(1);
    }

  }

  public static void main(String[] args) {
    CliFrontend cli = new CliFrontend();
    cli.parse(args);
  }

}