/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package spark.mllib.math.vector

import spark.mllib.math.function.{DoubleFunction, DoubleDoubleFunction}
import spark.mllib.math.collection.map.OrderedIntDoubleMapping

/**
 * The basic interface including numerous convenience functions.
 *
 * NOTE: All implementing classes must have a
 * constructor that takes an int as dimension and a no-arg constructor that can be used for marshalling the Writable
 * instance <p/> NOTE: Implementations may choose to reuse the Vector.Element in the Iterable methods
 */
trait Vector extends Serializable with Equals with Cloneable {
  /** Return the dimension of the vector. */
  def dimension: Int

  /** @return a formatted String suitable for output */
  override def toString: String

  /**
   * Get the value at the given index, without checking bounds
   *
   * @param index an int index
   * @return the double at the index
   */
  def apply(index: Int): Double

  /**
   * Set the value at the given index, without checking bounds.
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   */
  def update(index: Int, value: Double): Unit

  /**
   * Return the value at the given index.
   *
   * @param index an int index
   * @return the double at the index
   * @throws IndexException if the index is out of bounds
   */
  def get(index: Int): Double

  /**
   * Set the value at the given index.
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   * @throws IndexException if the index is out of bounds
   */
  def set(index: Int, value: Double): Unit

  /**
   * @return true iff this implementation should be considered dense -- that it explicitly
   *         represents every value
   */
  def isDense: Boolean

  /**
   * @return true iff this implementation should be considered to be iterable in index order in an efficient way.
   *         In particular this implies that `#all()` and `#nonZeroes()` return elements
   *         in ascending order by index.
   */
  def isSequentialAccess: Boolean

  /** Return a deep copy of the recipient. */
  override def clone(): Vector = super.clone().asInstanceOf[Vector]

  /** Return an empty vector of the same underlying class as the receiver, with specified dimension. */
  def like(_dimension: Int = dimension): Vector

  /** Return an vector of the same underlying class as the receiver, containing values of the array. */
  def like(array: Array[Double]): Vector

  /** Convert to a Double Array. */
  def toArray: Array[Double]

  /**
   * Assign the value to all elements of the receiver
   *
   * @param value a double value
   * @return the modified receiver
   */
  def assign(value: Double): Vector

  /**
   * Assign the values to the receiver
   *
   * @param values a double[] of values
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  def assign(values: Array[Double]): Vector

  /**
   * Assign the other vector values to the receiver
   *
   * @param other a Vector
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  def assign(other: Vector): Vector

  /**
   * Apply the function to each element of the receiver
   *
   * @param function a DoubleFunction to apply
   * @return the modified receiver
   */
  def assign(function: DoubleFunction): Vector

  /**
   * Apply the function to each element of the receiver and the corresponding element of the other argument
   *
   * @param other    a Vector containing the second arguments to the function
   * @param function a DoubleDoubleFunction to apply
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  def assign(other: Vector, function: DoubleDoubleFunction): Vector

  /**
   * Apply the function to each element of the receiver, using the y value as the second argument of the
   * DoubleDoubleFunction
   *
   * @param f a DoubleDoubleFunction to be applied
   * @param y a double value to be argument to the function
   * @return the modified receiver
   */
  def assign(f: DoubleDoubleFunction, y: Double): Vector

  def all: Iterable[Vector.Element]

  def nonZeroes: Iterable[Vector.Element]

  /**
   * Return an object of Vector.Element representing an element of this Vector. Useful when designing new iterator
   * types.
   *
   * @param index Index of the Vector.Element required
   * @return The Vector.Element Object
   */
  def getElement(index: Int): Vector.Element

  /**
   * Merge a set of (index, value) pairs into the vector.
   * @param updates an ordered mapping of indices to values to be merged in.
   */
  def mergeUpdates(updates: OrderedIntDoubleMapping): Unit

  /** Return a new vector containing the element by element sum of the recipient and the argument. */
  def +(that: Vector): Vector

  /** Return a new vector containing the sum of each element of the recipient and the argument. */
  def +(x: Double): Vector

  /** Return the original vector containing the element by element sum of the recipient and the argument. */
  def +=(that: Vector): Vector

  /** Return the original vector containing the sum of each element of the recipient and the argument. */
  def +=(x: Double): Vector

  /** Return a new vector containing the element by element difference of the recipient and the argument. */
  def -(that: Vector): Vector

  /** Return a new vector containing the element of the recipient each subtracted by the argument. */
  def -(x: Double): Vector

  /** Return the original vector containing the element by element difference of the recipient and the argument. */
  def -=(that: Vector): Vector

  /** Return the original vector containing the element of the recipient each subtracted by the argument. */
  def -=(x: Double): Vector

  /** Return the dot product of the recipient and the argument. */
  def *(that: Vector): Double

  /** Return a new vector containing the elements of the recipient multiplied by the argument. */
  def *(x: Double): Vector

  /** Return the original containing the elements of the recipient multiplied by the argument. */
  def *=(x: Double): Vector

  /**
   * Return a new vector containing the element-wise product of the recipient and the argument
   *
   * @param that a Vector argument
   * @return a new Vector
   * @throws DimensionException if the cardinalities differ
   */
  def times(that: Vector): Vector

  /** Return a new vector containing the elements of the recipient divided by the argument. */
  def /(x: Double): Vector

  /** Return the original vector containing the elements of the recipient divided by the argument. */
  def /=(x: Double): Vector

  /** Elementwise divide. */
  def /(that: Vector): Vector

  /** Elementwise divide(in place). */
  def /=(that: Vector): Vector

  /** Return the sum of all the elements of the vector. */
  def sum: Double

  /**
   * Return the sum of squares of all elements in the vector. Square root of this value is the length of the vector.
   */
  def lengthSquared: Double

  /** Get the square of the distance between this vector and the other vector. */
  def distanceSquared(that: Vector): Double

  /** Return a new vector containing the normalized (L_2 norm) values of the recipient
    *
    * @return a new Vector
    */
  def normalize(): Vector

  /**
   * Return a new Vector containing the normalized (L_power norm) values of the recipient. See
   * http://en.wikipedia.org/wiki/Lp_space <p/> Technically, when 0 < power < 1, we don't have a norm, just a metric,
   * but we'll overload this here. <p/> Also supports power == 0 (number of non-zero elements) and power =
   * `Double#POSITIVE_INFINITY` (max element). Again, see the Wikipedia page for more info
   *
   * @param power The power to use. Must be >= 0. May also be `Double#POSITIVE_INFINITY`. See the Wikipedia link
   *              for more on this.
   * @return a new Vector x such that norm(x, power) == 1
   */
  def normalize(power: Double): Vector

  /**
   * Return a new vector containing the log(1 + entry)/ L_2 norm  values of the recipient
   *
   * @return a new Vector
   */
  def logNormalize(): Vector

  /**
   * Return a new Vector with a normalized value calculated as log_power(1 + entry)/ L_power norm. <p/>
   *
   * @param power The power to use. Must be > 1. Cannot be `Double#POSITIVE_INFINITY`.
   * @return a new Vector
   */
  def logNormalize(power: Double): Vector

  /**
   * Return the p-norm of the vector. See http://en.wikipedia.org/wiki/Lp_space <p/> Technically, when 0 &gt; power
   * &lt; 1, we don't have a norm, just a metric, but we'll overload this here. Also supports power == 0 (number of
   * non-zero elements) and power = `Double#POSITIVE_INFINITY` (max element). Again, see the Wikipedia page for
   * more info.
   *
   * @param power The power to use.
   * @see #normalize(double)
   */
  def norm(power: Double): Double

  /** @return The minimum value in the Vector */
  def minValue: Double

  /** @return The index of the minimum value */
  def minValueIndex: Int

  /** @return The maximum value in the Vector */
  def maxValue: Double

  /** @return The index of the maximum value */
  def maxValueIndex: Int

  /**
   * Increment the value at the given index by the given value.
   *
   * @param index an int index into the receiver
   * @param increment sets the value at the given index to value + increment;
   */
  def incrementQuick(index: Int, increment: Double)

  /**
   * Return the number of values in the recipient which are not the default value.  For instance, for a
   * sparse vector, this would be the number of non-zero values. */
  def getNumNondefaultElements: Int

  /**
   * Return the number of non zero elements in the vector. */
  def getNumNonZeroElements: Int

  /**
   * Return a new vector containing the subset of the recipient
   *
   * @param offset an int offset into the receiver
   * @param length the cardinality of the desired result
   * @return a new Vector
   * @throws DimensionException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the receiver
   */
  def viewPart(offset: Int, length: Int): Vector

  /**
   * Examples speak louder than words:  aggregate(plus, pow(2)) is another way to say
   * lengthSquared(), aggregate(max, abs) is norm(Double.POSITIVE_INFINITY).  To sum all of the positive values,
   * aggregate(plus, max(0)).
   * @param aggregator used to combine the current value of the aggregation with the result of map.apply(nextValue)
   * @param map a function to apply to each element of the vector in turn before passing to the aggregator
   * @return the final aggregation
   */
  def aggregate(aggregator: DoubleDoubleFunction, map: DoubleFunction): Double

  /**
   * <p>Generalized inner product - take two vectors, iterate over them both, using the combiner to combine together
   * (and possibly map in some way) each pair of values, which are then aggregated with the previous accumulated
   * value in the combiner.</p>
   * <p>
   * Example: dot(other) could be expressed as aggregate(other, Plus, Times), and kernelized inner products (which
   * are symmetric on the indices) work similarly.
   * @param that a vector to aggregate in combination with
   * @param aggregator function we're aggregating with; fa
   * @param combiner function we're combining with; fc
   * @return the final aggregation; if r0 = fc(this[0], other[0]), ri = fa(r_{i-1}, fc(this[i], other[i]))
   *         for all i > 0
   */
  def aggregate(that: Vector, aggregator: DoubleDoubleFunction, combiner: DoubleDoubleFunction): Double

  /**
   * Gets an estimate of the cost (in number of operations) it takes to lookup a random element in this vector.
   */
  def getLookupCost: Double

  /**
   * Gets an estimate of the cost (in number of operations) it takes to advance an iterator through the nonzero
   * elements of this vector.
   */
  def getIteratorAdvanceCost: Double

  /**
   * @return true iff adding a new (nonzero) element takes constant time for this vector.
   */
  def isAddConstantTime: Boolean

  /** Return the extended Vector with intercept, (1.0, Vector). */
  def extend(intercept: Double = 1.0): Vector

  /** Return the original Vector. */
  def restore(): (Double, Vector)
}

object Vector {

  /**
   * A holder for information about a specific item in the Vector. <p/> When using with an Iterator, the implementation
   * may choose to reuse this element, so you may need to make a copy if you want to keep it
   */
  abstract class Element {

    /** @return the value of this vector element. */
    def value: Double

    /** @return the index of this vector element. */
    def index: Int

    /** @param value Set the current element to value. */
    def set(value: Double): Unit
  }

}