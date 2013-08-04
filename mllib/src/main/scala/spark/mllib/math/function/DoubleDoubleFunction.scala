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

package spark.mllib.math.function

/**
 * A function that takes two arguments and returns a single value.
 **/
abstract class DoubleDoubleFunction extends ((Double, Double) => Double) {

  /**
   * @return true iff f(x, 0) = x for any x
   */
  def isLikeRightPlus: Boolean = false

  /**
   * @return true iff f(0, y) = 0 for any y
   */
  def isLikeLeftMult: Boolean = false

  /**
   * @return true iff f(x, 0) = 0 for any x
   */
  def isLikeRightMult: Boolean = false

  /**
   * @return true iff f(x, 0) = f(0, y) = 0 for any x, y
   */
  def isLikeMult: Boolean = {
    isLikeLeftMult && isLikeRightMult
  }

  /**
   * @return true iff f(x, y) = f(y, x) for any x, y
   */
  def isCommutative: Boolean = false

  /**
   * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
   */
  def isAssociative: Boolean = false

  /**
   * @return true iff f(x, y) = f(y, x) for any x, y AND f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
   */
  def isAssociativeAndCommutative: Boolean = {
    isAssociative && isCommutative
  }

  /**
   * @return true iff f(0, 0) != 0
   */
  def isDensifying: Boolean = apply(0.0, 0.0) != 0.0
}
