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

import spark.{Logging, RDD, SparkContext}
import spark.mllib.util.MLUtils
import spark.mllib.math.vector.{Vector}

import org.jblas.DoubleMatrix
import org.jblas.Solve

import scala.annotation.tailrec
import scala.collection.mutable

/**
 * Ridge Regression from Joseph Gonzalez's implementation in MLBase
 */
class RidgeRegressionModel(
    val weights: Vector,
    val intercept: Double,
    val lambdaOpt: Double,
    val lambdas: Seq[(Double, Double, Vector)])
  extends RegressionModel {

  override def predict(testData: RDD[Vector]): RDD[Double] = {
    // A small optimization to avoid serializing the entire model.
    val localIntercept = this.intercept
    val localWeights = this.weights
    testData.map { x => x * localWeights + localIntercept }
  }

  override def predict(testData: Vector): Double = {
    testData * this.weights + this.intercept
  }
}

class RidgeRegression private (var lambdaLow: Double, var lambdaHigh: Double)
  extends Logging {

  def this() = this(0.0, 100.0)

  /**
   * Set the lower bound on binary search for lambda's. Default is 0.
   */
  def setLowLambda(low: Double) = {
    this.lambdaLow = low
    this
  }

  /**
   * Set the upper bound on binary search for lambda's. Default is 100.0.
   */
  def setHighLambda(hi: Double) = {
    this.lambdaHigh = hi
    this
  }

  def train(input: RDD[(Double, Vector)]): RidgeRegressionModel = {
    val oneSample = input.take(1)(0)._2
    val nfeatures: Int = oneSample.dimension
    val nexamples: Long = input.count()

    val (yMean, xColMean, xColSd) = MLUtils.computeStats(input, nfeatures, nexamples)

    val data = input.map { case(y, features) =>
      val yNormalized = y - yMean
      val xColMeanV = features.like(xColMean.toArray())
      val xColSdV = features.like(xColSd.toArray())

      val featuresMat = features.clone()
      val featuresNormalized = (featuresMat - xColMeanV) / xColSdV
      (yNormalized, featuresNormalized.toArray)
    }

    // Compute XtX - Size of XtX is nfeatures by nfeatures
    val XtX: DoubleMatrix = data.map { case (y, features) =>
      val x = new DoubleMatrix(1, features.length, features:_*)
      x.transpose().mmul(x)
    }.reduce(_.addi(_))

    // Compute Xt*y - Size of Xty is nfeatures by 1
    val Xty: DoubleMatrix = data.map { case (y, features) =>
      new DoubleMatrix(features.length, 1, features:_*).mul(y)
    }.reduce(_.addi(_))

    // Define a function to compute the leave one out cross validation error
    // for a single example
    def crossValidate(lambda: Double): (Double, Double, DoubleMatrix) = {
      // Compute the MLE ridge regression parameter value

      // Ridge Regression parameter = inv(XtX + \lambda*I) * Xty
      val XtXlambda = DoubleMatrix.eye(nfeatures).muli(lambda).addi(XtX)
      val w = Solve.solveSymmetric(XtXlambda, Xty)

      val invXtX = Solve.solveSymmetric(XtXlambda, DoubleMatrix.eye(nfeatures))

      // compute the generalized cross validation score
      val cverror = data.map {
        case (y, features) =>
          val x = new DoubleMatrix(features.length, 1, features:_*)
          val yhat = w.transpose().mmul(x).get(0)
          val H_ii = x.transpose().mmul(invXtX).mmul(x).get(0)
          val residual = (y - yhat) / (1.0 - H_ii)
          residual * residual
      }.reduce(_ + _) / nexamples

      (lambda, cverror, w)
    }

    // Binary search for the best assignment to lambda.
    def binSearch(low: Double, high: Double): Seq[(Double, Double, DoubleMatrix)] = {
      val buffer = mutable.ListBuffer.empty[(Double, Double, DoubleMatrix)]

      @tailrec
      def loop(low: Double, high: Double): Seq[(Double, Double, DoubleMatrix)] = {
        val mid = (high - low) / 2 + low
        val lowValue = crossValidate((mid - low) / 2 + low)
        val highValue = crossValidate((high - mid) / 2 + mid)
        val (newLow, newHigh) = if (lowValue._2 < highValue._2) {
          (low, mid + (high-low)/4)
        } else {
          (mid - (high-low)/4, high)
        }
        if (newHigh - newLow > 1.0E-7) {
          buffer += lowValue += highValue
          loop(newLow, newHigh)
        } else {
          buffer += lowValue += highValue
          buffer.result()
        }
      }

      loop(low, high)
    }

    // Actually compute the best lambda
    val lambdas = binSearch(lambdaLow, lambdaHigh).sortBy(_._1)
    val lambdasV = lambdas.map {
      case (low, high, matrix) => (low, high, oneSample.like(matrix.toArray()))
    }

    // Find the best parameter set by taking the lowest cverror.
    val (lambdaOpt, cverror, weights) = lambdas.reduce((a, b) => if (a._2 < b._2) a else b)

    // Return the model which contains the solution
    val weightsScaled = oneSample.like(weights.div(xColSd).toArray())
    val intercept = yMean - (weights.transpose().mmul(xColMean.div(xColSd)).get(0))
    val model = new RidgeRegressionModel(weightsScaled, intercept, lambdaOpt, lambdasV)

    logInfo("RidgeRegression: optimal lambda " + model.lambdaOpt)
    logInfo("RidgeRegression: optimal weights " + model.weights)
    logInfo("RidgeRegression: optimal intercept " + model.intercept)
    logInfo("RidgeRegression: cross-validation error " + cverror)

    model
  }
}

/**
 * Top-level methods for calling Ridge Regression.
 * NOTE(shivaram): We use multiple train methods instead of default arguments to support
 *                 Java programs.
 */
object RidgeRegression {

  /**
   * Train a ridge regression model given an RDD of (response, features) pairs.
   * We use the closed form solution to compute the cross-validation score for
   * a given lambda. The optimal lambda is computed by performing binary search
   * between the provided bounds of lambda.
   *
   * @param input RDD of (response, array of features) pairs.
   * @param lambdaLow lower bound used in binary search for lambda
   * @param lambdaHigh upper bound used in binary search for lambda
   */
  def train(
      input: RDD[(Double, Vector)],
      lambdaLow: Double,
      lambdaHigh: Double)
    : RidgeRegressionModel =
  {
    new RidgeRegression(lambdaLow, lambdaHigh).train(input)
  }

  /**
   * Train a ridge regression model given an RDD of (response, features) pairs.
   * We use the closed form solution to compute the cross-validation score for
   * a given lambda. The optimal lambda is computed by performing binary search
   * between lambda values of 0 and 100.
   *
   * @param input RDD of (response, array of features) pairs.
   */
  def train(input: RDD[(Double, Vector)]) : RidgeRegressionModel = {
    train(input, 0.0, 100.0)
  }

  def main(args: Array[String]) {
    if (args.length != 2) {
      println("Usage: RidgeRegression <master> <input_dir>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "RidgeRegression")
    val data = MLUtils.loadLabeledData(sc, args(1))
    val model = RidgeRegression.train(data, 0, 1000)
    sc.stop()
  }
}
