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

package spark.mllib.math.vector.distance

import spark.mllib.math.vector.{Vector, DimensionException}

class CosineDistanceMeasure extends DistanceMeasure {

  override def distance(v1: Vector, v2: Vector): Double = {
    if(v1.dimension != v2.dimension) throw new DimensionException(v1.dimension, v2.dimension)

    val lengthSquaredv1 = v1.lengthSquared
    val lengthSquaredv2 = v2.lengthSquared

    val dotProduct = v2 * v1
    val denominatorTemp = scala.math.sqrt(lengthSquaredv1) * scala.math.sqrt(lengthSquaredv2)

    // correct for floating-point rounding errors
    val denominator = if (denominatorTemp < dotProduct) dotProduct else denominatorTemp

    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0) 0
    else 1.0 - dotProduct / denominator
  }

  override def distance(centroidLengthSquare: Double, centroid: Vector, v: Vector): Double = {
    val lengthSquaredv = v.lengthSquared

    val dotProduct = v * centroid
    val denominatorTemp = scala.math.sqrt(centroidLengthSquare) * scala.math.sqrt(lengthSquaredv)

    // correct for floating-point rounding errors
    val denominator = if (denominatorTemp < dotProduct) dotProduct else denominatorTemp

    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0) 0
    else 1.0 - dotProduct / denominator
  }
}