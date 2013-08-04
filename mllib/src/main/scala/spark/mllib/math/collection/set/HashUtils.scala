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

package spark.mllib.math.collection.set

/**
 * Computes hashes of primitive values.  Providing these as statics allows the templated code
 * to compute hashes of sets.
 */
private[math] final object HashUtils {

  def hash(x: Byte): Int = x

  def hash(x: Short): Int = x

  def hash(x: Char): Int = x

  def hash(x: Int): Int = x

  def hash(x: Float): Int = {
    java.lang.Float.floatToIntBits(x) >>> 3 +
    java.lang.Float.floatToIntBits((math.Pi * x).toFloat)
  }

  def hash(x: Double): Int = {
    hash(17 * java.lang.Double.doubleToLongBits(x))
  }

  def hash(x: Long): Int = {
    ((x * 11) >>> 32 ^ x).toInt
  }
}