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
import spark.mllib.math.vector.{DenseVector, SparseVector}

class EuclideanDistanceMeasureSuite extends FunSuite {
  test("distance") {
    val distanceMeasure = new EuclideanDistanceMeasure()

    val v1 = DenseVector(3.0, 0.0)
    val v2 = DenseVector(0.0, 4.0)
    val v3 = SparseVector(2, (0, 3.0))
    val v4 = SparseVector(2, (1, 4.0))
    
    assert(distanceMeasure.distance(v1, v2) == 5.0)
    assert(distanceMeasure.distance(v3, v4) == 5.0)
    assert(distanceMeasure.distance(v1, v4) == 5.0)
    assert(distanceMeasure.distance(v2, v3) == 5.0)
  }
}
