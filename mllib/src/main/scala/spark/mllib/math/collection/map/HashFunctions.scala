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

package spark.mllib.math.collection.map

/** Provides various hash functions. */
private[math] final object HashFunctions {
  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Char): Int = value

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Double): Int = {
    val bits = java.lang.Double.doubleToLongBits(value)
    (bits ^ (bits >>> 32)).toInt

    //return (int) Double.doubleToLongBits(value*663608941.737);
    // this avoids excessive hashCollisions in the case values are of the form (1.0, 2.0, 3.0, ...)
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Float): Int = {
    java.lang.Float.floatToIntBits(value * 663608941.737f)
    // this avoids excessive hashCollisions in the case values are of the form (1.0, 2.0, 3.0, ...)
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Int): Int = value

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Long): Int = (value ^ (value >> 32)).toInt

  /**
   * Returns a hashcode for the specified object.
   *
   * @return a hash code value for the specified object.
   */
  def hash(obj: AnyRef): Int = if (obj == null) 0 else obj.hashCode()

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Short): Int = value

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  def hash(value: Boolean): Int = if (value) 1231 else 1237
}