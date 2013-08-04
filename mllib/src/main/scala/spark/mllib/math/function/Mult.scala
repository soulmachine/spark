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
 * Only for performance tuning of compute intensive linear algebraic computations.
 * Constructs functions that return one of
 * <ul>
 * <li><tt>a * constant</tt>
 * <li><tt>a / constant</tt>
 * </ul> 
 * <tt>a</tt> is variable, <tt>constant</tt> is fixed, but for performance reasons publicly accessible.
 * Intended to be passed to <tt>matrix.assign(function)</tt> methods.
 */

final class Mult(var multiplicator: Double = 0.0) extends DoubleFunction {

  /** Returns the result of the function evaluation. */
  override def apply(a: Double): Double = a * multiplicator
}

final object Mult {
   /** <tt>a / constant</tt>. */
  def div(constant: Double): Mult = mult(1 / constant)

  /** <tt>a * constant</tt>. */
  def mult(constant: Double): Mult = new Mult(constant)
}