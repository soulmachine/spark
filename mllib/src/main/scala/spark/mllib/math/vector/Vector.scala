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

/**
 * The basic interface including numerous convenience functions.
 * 
 * NOTE: All implementing classes must have a
 * constructor that takes an int as dimension and a no-arg constructor that can be used for marshalling the Writable
 * instance <p/> NOTE: Implementations may choose to reuse the Vector.Element in the Iterable methods
 * 
 */
abstract class Vector extends Serializable with Equals {
  /**
   * Get the dimension of the vector.
   */
  def dimension: Int
  /** Get the i-th element of the vector. */
  def apply(i: Int): Double
  /** Set the i-th element of the vector. */
  def update(i: Int, value: Double): Unit
  
  /** Return an empty vector of the same underlying class as the receiver. */
  def like(): Vector
  /** Return an vector of the same underlying class as the receiver, containing values of the array. */
  def like(array: Array[Double]): Vector
  
  /** Convert to a Double Array. */
  def toArray(): Array[Double]
  
  /** Return a new vector containing the element by element sum of the recipient and the argument. */
  def + (that: Vector): Vector
  /** Return a new vector containing the sum of each element of the recipient and the argument. */
  def + (x: Double): Vector
  /** Return the original vector containing the element by element sum of the recipient and the argument. */
  def += (that: Vector): Vector
  /** Return the original vector containing the sum of each element of the recipient and the argument. */
  def += (x: Double): Vector
  
  /** Return a new vector containing the element by element difference of the recipient and the argument. */
  def - (that: Vector): Vector
  /** Return a new vector containing the element of the recipient each subtracted by the argument. */
  def - (x: Double): Vector
  /** Return the original vector containing the element by element difference of the recipient and the argument. */
  def -= (that: Vector): Vector
  /** Return the original vector containing the element of the recipient each subtracted by the argument. */
  def -= (x: Double): Vector
  
  /** Return a new vector containing the elements of the recipient divided by the argument. */
  def / (x: Double): Vector
  /** Return the original vector containing the elements of the recipient divided by the argument. */
  def /= (x: Double): Vector

  /** Return the dot product of the recipient and the argument. */
  def * (that: Vector): Double
  /** Return a new vector containing the elements of the recipient multiplied by the argument. */
  def * (x: Double): Vector
  /** Return the original containing the elements of the recipient multiplied by the argument. */
  def *= (x: Double): Vector
  
  /** Return the sum of all the elements of the vector. */
  def sum():Double
  
  /** Return the sum of squares of all elements in the vector. Square root of this value is the length of the vector. */
  def getLengthSquared(): Double
  /** Get the square of the distance between this vector and the other vector. */
  def getDistanceSquared(that: Vector): Double
  
  /**
   * Return a new vector containing the normalized (L_2 norm) values of the recipient
   *
   * @return a new Vector
   */
  def normalize(): Vector

  /**
   * Return a new Vector containing the normalized (L_power norm) values of the recipient. See
   * http://en.wikipedia.org/wiki/Lp_space <p/> Technically, when 0 < power < 1, we don't have a norm, just a metric,
   * but we'll overload this here. <p/> Also supports power == 0 (number of non-zero elements) and power = {@link
   * Double#POSITIVE_INFINITY} (max element). Again, see the Wikipedia page for more info
   *
   * @param power The power to use. Must be >= 0. May also be {@link Double#POSITIVE_INFINITY}. See the Wikipedia link
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
   * @param power The power to use. Must be > 1. Cannot be {@link Double#POSITIVE_INFINITY}.
   * @return a new Vector
   */
  def logNormalize(power: Double): Vector

  /**
   * Return the p-norm of the vector. See http://en.wikipedia.org/wiki/Lp_space <p/> Technically, when 0 &gt; power
   * &lt; 1, we don't have a norm, just a metric, but we'll overload this here. Also supports power == 0 (number of
   * non-zero elements) and power = {@link Double#POSITIVE_INFINITY} (max element). Again, see the Wikipedia page for
   * more info.
   *
   * @param power The power to use.
   * @see #normalize(double)
   */
  def norm(power: Double): Double
}
