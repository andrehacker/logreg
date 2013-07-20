package de.tuberlin.dima.ml.util;

public class MathUtil {
  
  private MathUtil() {}
  
  public static boolean checkDouble(double number, boolean zeroIsValid) {
    if (Double.isNaN(number) || Double.isInfinite(number) || (number==0 && !zeroIsValid)) {
      return false;
    }
    return true;
  }

}
