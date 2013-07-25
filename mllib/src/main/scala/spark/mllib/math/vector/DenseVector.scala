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

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions.{pow, logi, log}

import cern.colt.function.IntDoubleProcedure

/** Implements dense vector, based on DoubleMatrix of jblas(http://jblas.org). */
class DenseVector private (dimension: Int, array: Option[DoubleMatrix] = None) extends AbstractVector(dimension) {
  require(array match {
    case Some(arr) => dimension == arr.length
    case None => true
  })

  private val values = array match {
    case None => new DoubleMatrix(dimension)
    case Some(arr) => arr
  }

  private def this(_values: DoubleMatrix) = this(_values.length, Option(_values))
  
  def this(dimension: Int) = this(dimension, None)

  def this(array: Array[Double]) = this(array.length, Option(new DoubleMatrix(array)))

  def this(that: DenseVector) = this(that.values.dup())

  def this(that: SparseVector) = this(that.toArray)

  override def clone(): DenseVector = new DenseVector(this)

  def apply(i: Int): Double = values.get(i)

  def update(i: Int, value: Double): Unit = {
    values.put(i, value)
    invalidateCachedLength()
  }
  
  def like(): Vector = new DenseVector(dimension)
  
  def like(array: Array[Double]): Vector = new DenseVector(array)
  
  def toArray(): Array[Double] = this.values.toArray()

  def +(that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = DenseVector.getOrConvert(that)
    val result = new DoubleMatrix(dimension)
    values.addi(thatV.values, result)
    new DenseVector(result)
  }

  def +(x: Double): Vector = {
    val result = new DoubleMatrix(dimension)
    values.addi(x, result)
    new DenseVector(result)
  }

  def +=(that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = DenseVector.getOrConvert(that)
    values.addi(thatV.values)
    invalidateCachedLength()
    this
  }

  def +=(x: Double): Vector = {
    values.addi(x)
    invalidateCachedLength()
    this
  }

  def -(that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = DenseVector.getOrConvert(that)
    val result = new DoubleMatrix(dimension)
    values.subi(thatV.values, result)
    new DenseVector(result)
  }

  def -(x: Double): Vector = {
    val result = new DoubleMatrix(dimension)
    values.subi(x, result)
    new DenseVector(result)
  }

  def -=(that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = DenseVector.getOrConvert(that)
    values.subi(thatV.values)
    invalidateCachedLength()
    this
  }
  def -=(x: Double): Vector = {
    values.subi(x)
    invalidateCachedLength()
    this
  }

  def *(that: Vector): Double = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = DenseVector.getOrConvert(that)
    values.dot(thatV.values)
  }

  def *(x: Double): Vector = new DenseVector(this.values.mul(x))
  
  def *=(x: Double): Vector = {
    this.values.muli(x)
    invalidateCachedLength()
    this
  }

  def /(x: Double): Vector = new DenseVector(this.values.div(x))
  
  def /=(x: Double): Vector = {
    this.values.divi(x)
    invalidateCachedLength()
    this
  }

  def sum(): Double = this.values.sum()

  def getDistanceSquared(that: Vector): Double = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = DenseVector.getOrConvert(that)
    this.values.squaredDistance(thatV.values)
  }

  def norm(power: Double): Double = {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0")
    }
    // We can special case certain powers.
    if (power.isInfinite) values.normmax()
    else if (power == 2.0) scala.math.sqrt(getLengthSquared())
    else if (power == 1.0) values.norm1()
    else if (power == 0.0) values.findIndices().length
    else pow(pow(values, power).sum(), 1.0 / power)
  }
  
  protected def logNormalize(power: Double, norm: Double): Vector = {
    // we can special case certain powers
    if (power.isInfinite() || power <= 1.0) {
      throw new IllegalArgumentException("Power must be > 1 and < infinity");
    } else {
      val result = this.clone()
      result.values.addi(1.0)
      logi(result.values)
      result.values.divi(log(power) * norm)
      result
    }
  }
  
  protected def dotSelf(): Double = values.dot(values)

  override def equals(obj: Any): Boolean = obj match {
    case vector: Vector => {
      this.canEqual(vector) && vector.canEqual(this) && this.## == vector.## && {
        vector match {
          case v: SparseVector => this.compareTo(v)
          case v: DenseVector => values.equals(v.values)
          case _ => false
        }
      }
    }
    case _ => false
  }
  
  override def toString(): String = values.toString()
}

object DenseVector {
  def apply(values: Double*): DenseVector = new DenseVector(values.toArray)
  
  /**
   * If a given vector is SparseVector then convert it to a DenseVector, otherwise return itself.
   */
  private def getOrConvert(v: Vector): DenseVector = v match {
    case v: DenseVector => v
    case v: SparseVector => new DenseVector(v)
    case _ => throw new UnsupportedOperationException
  }
}
