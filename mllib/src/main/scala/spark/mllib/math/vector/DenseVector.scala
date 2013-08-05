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
import scala.collection.JavaConversions._
import spark.mllib.math.collection.map.OrderedIntDoubleMapping
import java.util

/** Implements dense vector, based on DoubleMatrix of jblas(http://jblas.org). */
class DenseVector private(private val values: DoubleMatrix) extends AbstractVector(values.length) {

  def this(dimension: Int) = this(new DoubleMatrix(dimension))

  /** For serialization purposes only */
  def this() = this(0)

  def this(array: Array[Double]) = this(new DoubleMatrix(array))

  /**
   * Copy-constructor (for use in turning a sparse vector into a dense one, for example)
   * @param vector The vector to copy
   */
  def this(vector: Vector) = {
    this(vector.dimension)
    for (e <- vector.nonZeroes) {
      values.put(e.index, e.value)
    }
  }

  override def apply(i: Int): Double = values.get(i)

  override def update(i: Int, value: Double): Unit = {
    values.put(i, value)
    invalidateCachedLength()
  }

  override def clone(): DenseVector = new DenseVector(this.values)

  override def like(_dimension: Int): Vector = new DenseVector(_dimension)

  override def like(array: Array[Double]): Vector = new DenseVector(array)

  override def toArray = this.values.toArray

  override def +(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super.+(that)
    }

    val thatV = that.asInstanceOf[DenseVector]
    val result = new DoubleMatrix(dimension)
    values.addi(thatV.values, result)
    new DenseVector(result)
  }

  override def +(x: Double): Vector = {
    val result = new DoubleMatrix(dimension)
    values.addi(x, result)
    new DenseVector(result)
  }

  override def +=(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super.+=(that)
    }

    val thatV = that.asInstanceOf[DenseVector]
    values.addi(thatV.values)
    invalidateCachedLength()
    this
  }

  override def +=(x: Double): Vector = {
    values.addi(x)
    invalidateCachedLength()
    this
  }

  override def -(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super.-(that)
    }

    val thatV = that.asInstanceOf[DenseVector]
    val result = new DoubleMatrix(dimension)
    values.subi(thatV.values, result)
    new DenseVector(result)
  }

  override def -(x: Double): Vector = {
    val result = new DoubleMatrix(dimension)
    values.subi(x, result)
    new DenseVector(result)
  }

  override def -=(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super.-=(that)
    }

    val thatV = that.asInstanceOf[DenseVector]
    values.subi(thatV.values)
    invalidateCachedLength()
    this
  }

  override def -=(x: Double): Vector = {
    values.subi(x)
    invalidateCachedLength()
    this
  }

  override def *(that: Vector): Double = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super.*(that)
    }

    val thatV = that.asInstanceOf[DenseVector]
    values.dot(thatV.values)
  }

  override def *(x: Double): Vector = new DenseVector(this.values.mul(x))

  override def *=(x: Double): Vector = {
    this.values.muli(x)
    invalidateCachedLength()
    this
  }

  override def /(x: Double): Vector = new DenseVector(this.values.div(x))

  override def /=(x: Double): Vector = {
    this.values.divi(x)
    invalidateCachedLength()
    this
  }

  override def /(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super./(that)
    }
    val result = this.clone()
    val thatV = that.asInstanceOf[DenseVector]
    result.values.divi(thatV.values)
    result
  }

  override def /=(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super./=(that)
    }
    val thatV = that.asInstanceOf[DenseVector]
    this.values.divi(thatV.values)
    this
  }


  override def sum: Double = this.values.sum()

  override def distanceSquared(that: Vector): Double = {
    AbstractVector.checkDimension(this, that)

    if (!DenseVector.isBothDense(this, that)) {
      return super.distanceSquared(that)
    }

    val thatV = that.asInstanceOf[DenseVector]
    this.values.squaredDistance(thatV.values)
  }

  override def norm(power: Double): Double = {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0")
    }
    // We can special case certain powers.
    if (power.isInfinite) values.normmax()
    else if (power == 2.0) scala.math.sqrt(lengthSquared)
    else if (power == 1.0) values.norm1()
    else if (power == 0.0) values.findIndices().length
    else pow(pow(values, power).sum(), 1.0 / power)
  }

  override def logNormalize(power: Double, norm: Double): Vector = {
    require(!power.isInfinite && power > 1.0, "Power must be > 1 and < infinity")

    val result = this.clone()
    result.values.addi(1.0)
    logi(result.values)
    result.values.divi(log(power) * norm)
    result
  }

  protected override def dotSelf(): Double = values.dot(values)

  override def equals(obj: Any): Boolean = {
    obj match {
      case that: DenseVector => values.equals(that.values)
      case _ => super.equals(obj)
    }
  }

  def addAll(v: Vector) {
    AbstractVector.checkDimension(this, v)

    for (e <- v.nonZeroes) {
      values.put(e.index, e.value)
    }
  }

  private final class NonDefaultIterator extends util.Iterator[Vector.Element] {
    private val element = new DenseElement()
    private var index = -1
    private var lookAheadIndex = -1

    override def hasNext: Boolean = {
      if (lookAheadIndex == index) {
        // User calls hasNext() after a next()
        lookAhead()
      } // else user called hasNext() repeatedly.
      lookAheadIndex < dimension
    }

    private def lookAhead() {
      lookAheadIndex += 1
      while (lookAheadIndex < dimension && values.get(lookAheadIndex) == 0.0) {
        lookAheadIndex += 1
      }
    }

    override def next(): Vector.Element = {
      if (lookAheadIndex == index) {
        // If user called next() without checking hasNext().
        lookAhead()
      }

      assert(lookAheadIndex > index)
      index = lookAheadIndex

      if (index >= dimension) {
        // If the end is reached.
        throw new NoSuchElementException()
      }

      element._index = index
      element
    }

    def remove() {
      throw new UnsupportedOperationException()
    }
  }

  private final class AllIterator extends util.Iterator[Vector.Element] {
    private val element = new DenseElement()

    override def hasNext: Boolean = element.index + 1 < dimension

    override def next(): Vector.Element = {
      if (element.index + 1 >= dimension) {
        // If the end is reached.
        throw new NoSuchElementException()
      }
      element._index += 1
      element
    }

    def remove() {
      throw new UnsupportedOperationException()
    }
  }

  private final class DenseElement(private[DenseVector] var _index: Int = -1) extends Vector.Element {

    override def value: Double = values.get(index)

    override def index: Int = _index

    override def set(value: Double) {
      invalidateCachedLength()
      values.put(index, value)
    }
  }

  def isDense: Boolean = true

  def isSequentialAccess: Boolean = true

  def mergeUpdates(updates: OrderedIntDoubleMapping) {
    val numUpdates = updates.getNumMappings
    val indices = updates.getIndices()
    val values = updates.getValues()

    for (i <- 0 until numUpdates) {
      values(indices(i)) = values(i)
    }
  }

  def getNumNondefaultElements: Int = values.length

  def getLookupCost: Double = 1.0

  def getIteratorAdvanceCost: Double = 1.0

  def isAddConstantTime: Boolean = true

  protected def iterator: Iterator[Vector.Element] = new AllIterator()

  protected def nonZeroIterator: Iterator[Vector.Element] = new NonDefaultIterator()
}

object DenseVector {
  def apply(values: Double*): DenseVector = new DenseVector(values.toArray)

  private def isBothDense(v1: Vector, v2: Vector): Boolean = {
    v1.isDense && v2.isDense
  }
}
