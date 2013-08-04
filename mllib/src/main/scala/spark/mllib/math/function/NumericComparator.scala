package spark.mllib.math.function

import spark.mllib.math.numeric.Numeric

/**
 * A comparison function which imposes a <i>total ordering</i> on some collection of elements.  Comparators can be
 * passed to a sort method (such as <tt>org.apache.mahout.math.Sorting.quickSort</tt>) to allow precise control over
 * the sort order.<p>
 *
 * Note: It is generally a good idea for comparators to implement <tt>java.io.Serializable</tt>, as they may be used as
 * ordering methods in serializable data structures.  In order for the data structure to serialize successfully, the
 * comparator (if provided) must implement <tt>Serializable</tt>.<p>
 *
 * @see java.util.Comparator
 * @see org.apache.mahout.math.Sorting
 */
abstract class NumericComparator[@specialized T: Numeric: ClassManifest] {

  /**
   * Compares its two arguments for order.  Returns a negative integer, zero, or a positive integer as the first
   * argument is less than, equal to, or greater than the second.<p>
   *
   * The implementor must ensure that <tt>sgn(compare(x, y)) == -sgn(compare(y, x))</tt> for all <tt>x</tt> and
   * <tt>y</tt>.  (This implies that <tt>compare(x, y)</tt> must throw an exception if and only if <tt>compare(y,
   * x)</tt> throws an exception.)<p>
   *
   * The implementor must also ensure that the relation is transitive: <tt>((compare(x, y)&gt;0) &amp;&amp; (compare(y,
   * z)&gt;0))</tt> implies <tt>compare(x, z)&gt;0</tt>.<p>
   *
   * Finally, the implementer must ensure that <tt>compare(x, y)==0</tt> implies that <tt>sgn(compare(x,
   * z))==sgn(compare(y, z))</tt> for all <tt>z</tt>.<p>
   *
   * @return a negative integer, zero, or a positive integer as the first argument is less than, equal to, or greater
   *         than the second.
   */
  def compare(o1: T, o2: T): Int
}