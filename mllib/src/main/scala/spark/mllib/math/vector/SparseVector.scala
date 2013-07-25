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

import cern.colt.matrix.impl.SparseDoubleMatrix1D
import cern.colt.list.{IntArrayList, DoubleArrayList}
import cern.colt.function.{DoubleFunction, DoubleDoubleFunction, IntDoubleProcedure}
import cern.jet.math.Functions

/** Implements sparse vector, based on SparseDoubleMatrix1D  of colt(http://acs.lbl.gov/software/colt/). */
class SparseVector(dimension: Int) extends AbstractVector(dimension) {
  private val values: SparseDoubleMatrix1D = new SparseDoubleMatrix1D(dimension)

  def this(that: SparseVector) = {
    this(that.dimension)
    values.assign(that.values)
  }

  def this(array: Array[Double]) = {
    this(array.length)
    values.assign(array)
  }
  
  def this(that: DenseVector) = this(that.toArray)

  override def clone(): SparseVector = new SparseVector(this)
  
  def apply(i: Int): Double = values.getQuick(i)
  
  def update(i: Int, value: Double): Unit = {
    values.setQuick(i, value)
    invalidateCachedLength()
  }
  
  def like(): Vector = new SparseVector(dimension)
  
  def like(array: Array[Double]): Vector = new SparseVector(array)
  
  def toArray(): Array[Double] = this.values.toArray()
  
  def + (that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)
    
    val thatV = SparseVector.getOrConvert(that)
    val result = new SparseVector(this)
    result.assign(thatV, Functions.plus)
  }
  
  def + (x: Double): Vector = {
    val result = this.clone
    if (x != 0.0) {
      result.assign(Functions.plus(x))
    }
    result
  }
  
  def += (that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)
    
    val thatV = SparseVector.getOrConvert(that)
    this.assign(thatV, Functions.plus)
  }
  
  def += (x: Double): Vector = {
    if (x != 0.0) {
      this.assign(Functions.plus(x))
    }
    this
  }
  
  def - (that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)
    
    val thatV = SparseVector.getOrConvert(that)
    val result = this.clone
    result.assign(thatV, Functions.minus)
  }
  
  def - (x: Double): Vector = {
    val result = this.clone
    if (x != 0.0) {
      result.assign(Functions.minus(x))
    }
    result
  }
  
  def -= (that: Vector): Vector = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)
    
    val thatV = SparseVector.getOrConvert(that)
    this.assign(thatV, Functions.minus)
  }
  
  def -= (x: Double): Vector = {
    if (x != 0.0) {
      this.assign(Functions.minus(x))
    }
    this
  }
  
  def *(that: Vector): Double = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)
    
    val thatV = SparseVector.getOrConvert(that)
    this.values.zDotProduct(thatV.values)
  }

  def *(x: Double): Vector = {
    val result = this.clone
    result.assign(Functions.mult(x))
    result
  }
  
  def *=(x: Double) = {
    this.assign(Functions.mult(x))
    this
  }
  
  def /(x: Double): Vector = {
    val result = this.clone
    result.assign(Functions.div(x))
    result
  }
  
  def /=(x: Double): Vector = {
    this.assign(Functions.div(x))
    this
  }
  
  def sum(): Double = this.values.zSum()
  
  def getDistanceSquared(that: Vector): Double = {
    if(this.dimension != that.dimension) throw new DimensionException(dimension, that.dimension)

    val thatV = SparseVector.getOrConvert(that)
    val diff = (this - thatV).asInstanceOf[SparseVector]
    diff.dotSelf()
  }
  
  def norm(power: Double): Double = {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0");
    }
    // We can special case certain powers.
    if (power.isInfinite()) values.aggregate(Functions.max, Functions.abs)
    else if (power == 2.0) scala.math.sqrt(getLengthSquared())
    else if (power == 1.0) values.aggregate(Functions.plus, Functions.abs)
    else if (power == 0.0) values.cardinality()
    else scala.math.pow(values.aggregate(Functions.plus, Functions.pow(power)), 1.0 / power)
  }
  
  /**
   * Fill all cells with the value.
   *
   * @param value the value to be filled into the cells.
   * @return the modified receiver
   */
  private def fill(value: Double): SparseVector = {
    this.values.assign(value)
    invalidateCachedLength()
    this
  }
  
  /**
   * Apply the function to each element of the receiver.
   *
   * @param function a DoubleFunction to apply
   * @return the modified receiver
   */
  private def assign(function: DoubleFunction): SparseVector = {
    this.values.assign(function)
    invalidateCachedLength()
    this
  }
  
  /**
   * Apply the function to each element of the receiver and the corresponding element of the other argument
   *
   * @param that    a Vector containing the second arguments to the function
   * @param function a DoubleDoubleFunction to apply
   * @return the modified receiver
   */
  private def assign(that: SparseVector, function: DoubleDoubleFunction): SparseVector = {
    this.values.assign(that.values, function)
    invalidateCachedLength()
    this;
  }
  
  protected def logNormalize(power: Double, norm: Double): Vector = {
    // we can special case certain powers
    if (power.isInfinite() || power <= 1.0) {
      throw new IllegalArgumentException("Power must be > 1 and < infinity");
    } else {
      val result = this.clone()
      result.assign(Functions.chain(Functions.div(norm), Functions.chain(Functions.lg(power), Functions.plus(1.0))))
      result
    }
  }
  
  protected def dotSelf(): Double = values.zDotProduct(values)

  override def equals(obj: Any): Boolean = obj match {
    case vector: Vector => {
      this.canEqual(vector) && vector.canEqual(this) && this.## == vector.## && {
        vector match {
          case v: SparseVector => values.equals(v.values)
          case v: DenseVector => this.compareTo(v)
          case _ => false
        }
      }
    }
    case _ => false
  }
  
  override def toString(): String = values.toString()
}

object SparseVector {
  def apply(dimension: Int, pairs: (Int, Double)*): SparseVector = {
    val v = new SparseVector(dimension)
    pairs.map(p => v(p._1) = p._2)
    v
  }
  
  /**
   * If a given vector is DenseVector then convert it to a SparseVector, otherwise return itself.
   */
  private def getOrConvert(v: Vector): SparseVector = v match {
    case v: DenseVector => new SparseVector(v)
    case v: SparseVector => v
    case _ => throw new UnsupportedOperationException
  }
}
