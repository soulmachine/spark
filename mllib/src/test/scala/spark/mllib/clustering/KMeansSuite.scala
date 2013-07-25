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

package spark.mllib.clustering

import scala.util.Random

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite

import spark.{SparkContext, RDD}
import spark.SparkContext._

import spark.mllib.math.vector.{Vector, DenseVector, SparseVector}

import org.jblas._


class KMeansSuite extends FunSuite with BeforeAndAfterAll {
  val sc = new SparkContext("local", "test")

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  val EPSILON = 1e-4

  import KMeans.{RANDOM, K_MEANS_PARALLEL}

  def prettyPrint(point: Vector): String = point.toString()

  def prettyPrint(points: Array[Vector]): String = {
    points.map(prettyPrint).mkString("(", "; ", ")")
  }

  // L1 distance between two points
  def distance1(v1: Vector, v2: Vector): Double = (v1 - v2).norm(1)

  // Assert that two vectors are equal within tolerance EPSILON
  def assertEqual(v1: Vector, v2: Vector) {
    def errorMessage = prettyPrint(v1) + " did not equal " + prettyPrint(v2)
    assert(v1.dimension == v2.dimension, errorMessage)
    assert(v1.getDistanceSquared(v2) <= EPSILON, errorMessage)
  }

  // Assert that two sets of points are equal, within EPSILON tolerance
  def assertSetsEqual(set1: Array[Vector], set2: Array[Vector]) {
    def errorMessage = prettyPrint(set1) + " did not equal " + prettyPrint(set2)
    assert(set1.length == set2.length, errorMessage)
    for (v <- set1) {
      val closestDistance = set2.map(w => distance1(v, w)).min
      if (closestDistance > EPSILON) {
        fail(errorMessage)
      }
    }
    for (v <- set2) {
      val closestDistance = set1.map(w => distance1(v, w)).min
      if (closestDistance > EPSILON) {
        fail(errorMessage)
      }
    }
  }

  test("single cluster") {
    val data = sc.parallelize(Array(
      DenseVector(1.0, 2.0, 6.0),
      DenseVector(1.0, 3.0, 0.0),
      DenseVector(1.0, 4.0, 6.0)
    )).asInstanceOf[RDD[Vector]]

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    var model = KMeans.train(data, k=1, maxIterations=1)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=2)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=5)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=5)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=5)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=1, initializationMode=RANDOM)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(
      data, k=1, maxIterations=1, runs=1, initializationMode=K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))
  }

  test("single cluster with big dataset") {
    val smallData = Array(
      DenseVector(1.0, 2.0, 6.0),
      DenseVector(1.0, 3.0, 0.0),
      DenseVector(1.0, 4.0, 6.0)
    )
    val data = sc.parallelize((1 to 100).flatMap(_ => smallData), 4).asInstanceOf[RDD[Vector]]

    // No matter how many runs or iterations we use, we should get one cluster,
    // centered at the mean of the points

    var model = KMeans.train(data, k=1, maxIterations=1)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=2)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=5)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=5)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=5)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=1, initializationMode=RANDOM)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))

    model = KMeans.train(data, k=1, maxIterations=1, runs=1, initializationMode=K_MEANS_PARALLEL)
    assertSetsEqual(model.clusterCenters, Array(DenseVector(1.0, 3.0, 4.0)))
  }

  test("k-means|| initialization") {
    val points = Array(
      DenseVector(1.0, 2.0, 6.0),
      DenseVector(1.0, 3.0, 0.0),
      DenseVector(1.0, 4.0, 6.0),
      DenseVector(1.0, 0.0, 1.0),
      DenseVector(1.0, 1.0, 1.0)
    ).asInstanceOf[Array[Vector]]
    val rdd = sc.parallelize(points).asInstanceOf[RDD[Vector]]

    // K-means|| initialization should place all clusters into distinct centers because
    // it will make at least five passes, and it will give non-zero probability to each
    // unselected point as long as it hasn't yet selected all of them

    var model = KMeans.train(rdd, k=5, maxIterations=1)
    assertSetsEqual(model.clusterCenters, points)

    // Iterations of Lloyd's should not change the answer either
    model = KMeans.train(rdd, k=5, maxIterations=10)
    assertSetsEqual(model.clusterCenters, points)

    // Neither should more runs
    model = KMeans.train(rdd, k=5, maxIterations=10, runs=5)
    assertSetsEqual(model.clusterCenters, points)
  }
}
