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

package spark.mllib.regression

import scala.util.Random

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite

import spark.SparkContext
import spark.mllib.math.vector.{Vector, DenseVector}


class LogisticRegressionSuite extends FunSuite with BeforeAndAfterAll {
  val sc = new SparkContext("local", "test")

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  // Generate input of the form Y = logistic(offset + scale*X)
  def generateLogisticInput(
      offset: Double,
      scale: Double,
      nPoints: Int,
      seed: Int): Seq[(Double, Vector)]  = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    // NOTE: if U is uniform[0, 1] then ln(u) - ln(1-u) is Logistic(0,1)
    val unifRand = new scala.util.Random(45)
    val rLogis = (0 until nPoints).map { i =>
      val u = unifRand.nextDouble()
      math.log(u) - math.log(1.0-u)
    }

    // y <- A + B*x + rLogis()
    // y <- as.numeric(y > 0)
    val y: Seq[Double] = (0 until nPoints).map { i =>
      val yVal = offset + scale * x1(i) + rLogis(i)
      if (yVal > 0) 1.0 else 0.0
    }

    val testData = (0 until nPoints).map(i => (y(i), new DenseVector(Array(x1(i)))))
    testData
  }

  def validatePrediction(predictions: Seq[Double], input: Seq[(Double, Vector)]) {
    val numOffPredictions = predictions.zip(input).filter { case (prediction, (expected, _)) =>
      // A prediction is off if the prediction is more than 0.5 away from expected value.
      math.abs(prediction - expected) > 0.5
    }.size
    // At least 80% of the predictions should be on.
    assert(numOffPredictions < input.length / 5)
  }

  // Test if we can correctly learn A, B where Y = logistic(A + B*X)
  test("logistic regression") {
    val nPoints = 10000
    val A = 2.0
    val B = -1.5

    val testData = generateLogisticInput(A, B, nPoints, 42)

    val testRDD = sc.parallelize(testData, 2)
    testRDD.cache()
    val lr = new LogisticRegression().setStepSize(10.0).setNumIterations(20)

    val model = lr.train(testRDD)

    // Test the weights
    val weight0 = model.weights(0)
    assert(weight0 >= -1.60 && weight0 <= -1.40, weight0 + " not in [-1.6, -1.4]")
    assert(model.intercept >= 1.9 && model.intercept <= 2.1, model.intercept + " not in [1.9, 2.1]")

    val validationData = generateLogisticInput(A, B, nPoints, 17)
    val validationRDD = sc.parallelize(validationData, 2)
    // Test prediction on RDD.
    validatePrediction(model.predict(validationRDD.map(_._2)).collect(), validationData)

    // Test prediction on Array.
    validatePrediction(validationData.map(row => model.predict(row._2)), validationData)
  }

  test("logistic regression with initial weights") {
    val nPoints = 10000
    val A = 2.0
    val B = -1.5

    val testData = generateLogisticInput(A, B, nPoints, 42)

    val initialB = -1.0
    val initialWeights = new DenseVector(Array(initialB))

    val testRDD = sc.parallelize(testData, 2)
    testRDD.cache()

    // Use half as many iterations as the previous test.
    val lr = new LogisticRegression().setStepSize(10.0).setNumIterations(10)

    val model = lr.train(testRDD, initialWeights)

    val weight0 = model.weights(0)
    assert(weight0 >= -1.60 && weight0 <= -1.40, weight0 + " not in [-1.6, -1.4]")
    assert(model.intercept >= 1.9 && model.intercept <= 2.1, model.intercept + " not in [1.9, 2.1]")

    val validationData = generateLogisticInput(A, B, nPoints, 17)
    val validationRDD = sc.parallelize(validationData, 2)
    // Test prediction on RDD.
    validatePrediction(model.predict(validationRDD.map(_._2)).collect(), validationData)

    // Test prediction on Array.
    validatePrediction(validationData.map(row => model.predict(row._2)), validationData)
  }
}
