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


import spark.mllib.math.Constants

/**
 * Only for performance tuning of compute intensive linear algebraic computations.
 * Constructs functions that return one of
 * <ul>
 * <li><tt>a + b*constant</tt>
 * <li><tt>a - b*constant</tt>
 * <li><tt>a + b/constant</tt>
 * <li><tt>a - b/constant</tt>
 * </ul> 
 * <tt>a</tt> and <tt>b</tt> are variables, <tt>constant</tt> is fixed, but for performance reasons publicly accessible.
 * Intended to be passed to <tt>matrix.assign(otherMatrix,function)</tt> methods.
 */

final class PlusMult(var multiplicator: Double) extends DoubleDoubleFunction {

  /** Returns the result of the function evaluation. */
  override def apply(a: Double, b: Double): Double = {
    a + b * multiplicator
  }

  /**
   * x + 0 * c = x
   * @return true iff f(x, 0) = x for any x
   */
  override def isLikeRightPlus(): Boolean = true

  /**
   * 0 + y * c = y * c != 0
   * @return true iff f(0, y) = 0 for any y
   */
  override def isLikeLeftMult(): Boolean = false

  /**
   * x + 0 * c = x != 0
   * @return true iff f(x, 0) = 0 for any x
   */
  override def isLikeRightMult(): Boolean = false

  /**
   * x + y * c = y + x * c iff c = 1
   * @return true iff f(x, y) = f(y, x) for any x, y
   */
  override def isCommutative(): Boolean = {
    math.abs(multiplicator - 1.0) < Constants.EPSILON
  }

  /**
   * f(x, f(y, z)) = x + c * (y + c * z) = x + c * y + c^2  * z
   * f(f(x, y), z) = (x + c * y) + c * z = x + c * y + c * z
   * true only for c = 0 or c = 1
   * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
   */
  override def isAssociative(): Boolean = {
    math.abs(multiplicator - 0.0) < Constants.EPSILON || 
    math.abs(multiplicator - 1.0) < Constants.EPSILON
  }
}

final object PlusMult {
  /** <tt>a - b*constant</tt>. */
  def minusMult(constant: Double): PlusMult = new PlusMult(-constant)

  /** <tt>a + b*constant</tt>. */
  def plusMult(constant: Double): PlusMult = new PlusMult(constant)
}