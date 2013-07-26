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

package spark.mllib.optimization

import org.jblas.DoubleMatrix
import spark.mllib.math.vector.Vector

abstract class Gradient extends Serializable {
  /**
   * Compute the gradient for a given row of data.
   *
   * @param data - One row of data. Row matrix of size 1xn where n is the number of features.
   * @param label - Label for this data item.
   * @param weights - Column matrix containing weights for every feature.
   */
  def compute(data: Vector, label: Double, weights: Vector): 
      (Vector, Double)
}
