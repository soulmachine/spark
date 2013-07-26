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

import org.scalatest.FunSuite

class SparseVectorSuite extends FunSuite {
  test("add") {
    val v1 = SparseVector(3, (0, 1.0), (1, 2.0))
    val v2 = SparseVector(3, (1, 2.0), (2, 3.0))
    val v3 = new SparseVector(3)
    val v4 = SparseVector(3, (0, -1.0), (1, -2.0))
    
    assert(v1 + v2 == DenseVector(1.0, 4.0, 3.0))
    assert(v1 + v2 != DenseVector(1.0, 3.99, 3.0))
    assert(v1 + v3 == v1)
    assert(v1 + v4 == v3)
    
    assert(DenseVector(4.0, 6.0, 3.0) == SparseVector(3, (0, 4.0), (1, 6.0), (2, 3.0)))
  }
  
  test("sub") {
    val v1 = SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0))
    val v2 = SparseVector(3, (0, 4.0), (1, 5.0), (2, 6.0))
    val v3 = new SparseVector(3)
    val v4 = SparseVector(3, (0, -1.0), (1, -2.0), (2, -3.0))
    val v5 = SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0))
    val v6 = DenseVector(3.0, 4.0, 0.0)
    
    assert(v2 - v1 == DenseVector(3.0, 3.0, 3.0))
    assert(v1 - v4 == DenseVector(2.0, 4.0, 6.0))
    assert(v1 - v5 == v3)
    assert(v6 - v1 == DenseVector(2.0, 2.0, -3.0))
  }
  
  test("mul") {
    val v1 = SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0))
    
    assert(v1 * 2.0 == DenseVector(2.0, 4.0, 6.0))
    assert(v1 * 0.0 == new DenseVector(3))
  }
  
  test("divide") {
    val v1 = SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0))
    
    assert(v1 / 0.5 == DenseVector(2.0, 4.0, 6.0))
  }
  
  test("dot") {
    val v1 = SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0))
    val v2 = SparseVector(3, (0, 4.0), (1, 5.0), (2, 6.0))
    val v3 = new SparseVector(3)
    val v4 = DenseVector(3.0, 4.0, 0.0)
    
    assert(v1 * v2 == 32.0)
    assert(v1 * v3 == 0.0)
    assert(v1 * v1 == 14.0)
    assert(v1 * v4 == 11.0)
  }
  
  test("dimension") {
    assert(3 == SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0)).dimension)
  }
  
  test("squared lengh") {
    assert(14.0 == SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0)).getLengthSquared)
  }
  
  test("squared distance") {
    val v1 = SparseVector(3, (0, 1.0), (1, 2.0), (2, 3.0))
    val v2 = SparseVector(3, (0, 4.0), (1, 5.0), (2, 6.0))
    assert(v1.getDistanceSquared(v2) == 27.0)
  }
}
