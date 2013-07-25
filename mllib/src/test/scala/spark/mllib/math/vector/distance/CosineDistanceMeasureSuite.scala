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

import org.scalatest.FunSuite
import spark.mllib.math.vector.DenseVector

class CosineDistanceMeasureSuite extends FunSuite {
  /** "Close enough" value for floating-point comparisons. */
  val EPSILON = 0.000001

  test("distance") {
    val distanceMeasure = new CosineDistanceMeasure()

    val vectors = Array(
      DenseVector(1, 0, 0, 0, 0, 0),
      DenseVector(1, 1, 1, 0, 0, 0),
      DenseVector(1, 1, 1, 1, 1, 1))

    val distanceMatrix = Array.fill(3, 3)(0.0)

    for (a <- 0 until 3) {
      for (b <- 0 until 3) {
        distanceMatrix(a)(b) = distanceMeasure.distance(vectors(a), vectors(b))
      }
    }

    assert(distanceMatrix(0)(0) < EPSILON)
    assert(distanceMatrix(0)(0) < distanceMatrix(0)(1))
    assert(distanceMatrix(0)(1) < distanceMatrix(0)(2))

    assert(distanceMatrix(1)(1) < EPSILON)
    assert(distanceMatrix(1)(0) > distanceMatrix(1)(1))
    assert(distanceMatrix(1)(2) < distanceMatrix(1)(0))

    assert(distanceMatrix(2)(2) < EPSILON)
    assert(distanceMatrix(2)(0) > distanceMatrix(2)(1))
    assert(distanceMatrix(2)(1) > distanceMatrix(2)(2))
  }
}
