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

import spark.mllib.math.function.{DoubleFunction, DoubleDoubleFunction, Functions}
import spark.mllib.math.collection.map.OrderedIntDoubleMapping
import spark.mllib.math.collection.set.HashUtils

/** Implementations of generic capabilities like lengthSquared and normalize.
  * And Implementations of operations between two different kinds of vectors.
  */
abstract class AbstractVector(val dimension: Int) extends Vector with LengthCachingVector {
  protected var _lengthSquared: Option[Double] = None

  override def all: Iterable[Vector.Element] = new Iterable[Vector.Element]() {
    override def iterator: Iterator[Vector.Element] = {
      AbstractVector.this.iterator
    }
  }

  override def nonZeroes: Iterable[Vector.Element] = new Iterable[Vector.Element]() {
    override def iterator: Iterator[Vector.Element] = nonZeroIterator
  }

  /**
   * Iterates over all elements
   */
  protected def iterator: Iterator[Vector.Element]

  /**
   * Iterates over all non-zero elements.
   */
  protected def nonZeroIterator: Iterator[Vector.Element]

  /**
   * Aggregates a vector by applying a mapping function fm(x) to every component and aggregating
   * the results with an aggregating function fa(x, y).
   *
   * @param aggregator used to combine the current value of the aggregation with the result of map.apply(nextValue)
   * @param map a function to apply to each element of the vector in turn before passing to the aggregator
   * @return the result of the aggregation
   */
  override def aggregate(aggregator: DoubleDoubleFunction, map: DoubleFunction): Double = {
    if (dimension == 0) {
      return 0
    }

    // If the aggregator is associative and commutative and it's likeLeftMult (fa(0, y) = 0), and there is
    // at least one zero in the vector (dimension > getNumNondefaultElements) and applying fm(0) = 0, the result
    // gets cascaded through the aggregation and the final result will be 0.
    if (aggregator.isAssociativeAndCommutative && aggregator.isLikeLeftMult
      && dimension > getNumNondefaultElements && !map.isDensifying) {
      return 0
    }

    var result: Double = 0.0
    if (isSequentialAccess || aggregator.isAssociativeAndCommutative) {
      var iter: Iterator[Vector.Element] = null
      // If fm(0) = 0 and fa(x, 0) = x, we can skip all zero values.
      if (!map.isDensifying && aggregator.isLikeRightPlus) {
        iter = nonZeroIterator
        if (!iter.hasNext) {
          return 0
        }
      } else {
        iter = iterator
      }
      var element = iter.next()
      result = map(element.value)
      while (iter.hasNext) {
        element = iter.next()
        result = aggregator(result, map(element.value))
      }
    } else {
      result = map(this(0))
      for (i <- 1 until dimension) {
        result = aggregator(result, map(this(i)))
      }
    }

    result
  }

  override def aggregate(other: Vector, aggregator: DoubleDoubleFunction, combiner: DoubleDoubleFunction): Double = {
    require(dimension == other.dimension, new DimensionException(dimension, other.dimension))
    if (dimension == 0)
      0
    else
      VectorBinaryAggregate.aggregateBest(this, other, aggregator, combiner)
  }

  /**
   * Subclasses must override to return an appropriately sparse or dense result
   */
  //protected abstract Matrix matrixLike(int rows, int columns);

  override def viewPart(offset: Int, length: Int): Vector = {
    if (offset < 0) {
      throw new IndexException(offset, dimension)
    }
    if (offset + length > dimension) {
      throw new IndexException(offset + length, dimension)
    }
    new VectorView(this, offset, length)
  }

  override def clone(): Vector = {
    try {
      val r = super.clone().asInstanceOf[AbstractVector]
      r._lengthSquared = _lengthSquared
      r
    } catch {
      case e: CloneNotSupportedException => throw new IllegalStateException("Can't happen")
    }
  }

  protected def dotSelf(): Double = {
    aggregate(Functions.PLUS, Functions.pow(2))
  }

  override def get(index: Int): Double = {
    if (index < 0 || index >= dimension) {
      throw new IndexException(index, dimension)
    }
    this(index)
  }

  override def set(index: Int, value: Double) {
    if (index < 0 || index >= dimension) {
      throw new IndexException(index, dimension)
    }
    this(index) = value
  }

  override def getElement(index: Int): Vector.Element = new LocalElement(index)

  def normalize(): Vector = this / scala.math.sqrt(lengthSquared)

  def normalize(power: Double): Vector = this / norm(power)

  def logNormalize(): Vector = logNormalize(2.0, scala.math.sqrt(lengthSquared))

  def logNormalize(power: Double): Vector = logNormalize(power, norm(power))

  def logNormalize(power: Double, norm: Double): Vector = {
    // we can special case certain powers
    if (power.isInfinity || power <= 1.0) {
      throw new IllegalArgumentException("Power must be > 1 and < infinity")
    } else {
      val denominator = norm * math.log(power)
      val result = createOptimizedCopy()
      for (element <- result.nonZeroes) {
        element.set(math.log1p(element.value) / denominator)
      }
      result
    }
  }

  override def norm(power: Double): Double = {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0")
    }
    // We can special case certain powers.
    if (power.isInfinity) {
      aggregate(Functions.MAX, Functions.ABS)
    } else if (power == 2.0) {
      math.sqrt(lengthSquared)
    } else if (power == 1.0) {
      var result = 0.0
      val iterator = this.nonZeroIterator
      while (iterator.hasNext) {
        result += math.abs(iterator.next().value)
      }
      result
      // TODO: this should ideally be used, but it's slower.
      // return aggregate(Functions.PLUS, Functions.ABS);
    } else if (power == 0.0) {
      getNumNonZeroElements
    } else {
      math.pow(aggregate(Functions.PLUS, Functions.pow(power)), 1.0 / power)
    }
  }

  def lengthSquared: Double = _lengthSquared match {
    case None =>
      _lengthSquared = Some(dotSelf())
      _lengthSquared.get
    case Some(length) =>
      length
  }

  def invalidateCachedLength() {
    _lengthSquared = None
  }

  override def distanceSquared(that: Vector): Double = {
    require(dimension == that.dimension, new DimensionException(dimension, that.dimension))

    val distanceEstimate = lengthSquared + that.lengthSquared - 2 * (this * that)

    if (distanceEstimate > 1.0e-3 * (lengthSquared + that.lengthSquared))
      // The vectors are far enough from each other that the formula is accurate.
      math.max(distanceEstimate, 0)
    else
      aggregate(that, Functions.PLUS, Functions.MINUS_SQUARED)
  }

  override def maxValue: Double =
    if (dimension == 0)
      Double.NegativeInfinity
    else
      aggregate(Functions.MAX, Functions.IDENTITY)

  override def maxValueIndex: Int = {
    var result = -1
    var max = Double.NegativeInfinity
    var nonZeroElements = 0
    val iter = this.nonZeroIterator
    while (iter.hasNext) {
      nonZeroElements += 1
      val element = iter.next()
      val tmp = element.value
      if (tmp > max) {
        max = tmp
        result = element.index
      }
    }
    // if the maxElement is negative and the vector is sparse then any
    // unfilled element(0.0) could be the maxValue hence we need to
    // find one of those elements
    if (nonZeroElements < dimension && max < 0.0) {
      for (element <- all) {
        if (element.value == 0.0) {
          return element.index
        }
      }
    }
    result
  }

  override def minValue: Double = {
    if (dimension == 0) {
      Double.PositiveInfinity
    } else {
      aggregate(Functions.MIN, Functions.IDENTITY)
    }
  }

  def minValueIndex: Int = {
    var result = -1
    var min = Double.PositiveInfinity
    var nonZeroElements = 0
    val iter = this.nonZeroIterator
    while (iter.hasNext) {
      nonZeroElements += 1
      val element = iter.next()
      val tmp = element.value
      if (tmp < min) {
        min = tmp
        result = element.index
      }
    }
    // if the maxElement is positive and the vector is sparse then any
    // unfilled element(0.0) could be the maxValue hence we need to
    // find one of those elements
    if (nonZeroElements < dimension && min > 0.0) {
      for (element <- all) {
        if (element.value == 0.0) {
          return element.index
        }
      }
    }
    result
  }

  override def incrementQuick(index: Int, increment: Double) {
    this(index) = this(index) + increment
  }

  /**
   * Copy the current vector in the most optimum fashion. Used by immutable methods like plus(), minus().
   * Use this instead of vector.like().assign(vector). Sub-class can choose to override this method.
   *
   * @return a copy of the current vector.
   */
  protected def createOptimizedCopy(): Vector = {
    AbstractVector.createOptimizedCopy(this)
  }

  override def sum: Double = aggregate(Functions.PLUS, Functions.IDENTITY)

  override def getNumNonZeroElements: Int = {
    var count = 0
    val it = nonZeroIterator
    while (it.hasNext) {
      if (it.next().value != 0.0) {
        count += 1
      }
    }
    count
  }

  override def assign(value: Double): Vector = {
    if (value == 0.0) {
      // Make all the non-zero values 0.
      val it = nonZeroIterator
      while (it.hasNext) {
        it.next().set(value)
      }
    } else {
      if (isSequentialAccess && !isAddConstantTime) {
        // Update all the non-zero values and queue the updates for the zero values.
        // The vector will become dense.
        val it = iterator
        val updates = new OrderedIntDoubleMapping()
        while (it.hasNext) {
          val element = it.next()
          if (element.value == 0.0) {
            updates(element.index) = value
          } else {
            element.set(value)
          }
        }
        mergeUpdates(updates)
      } else {
        for (i <- 0 until dimension) {
          this(i) = value
        }
      }
    }
    invalidateCachedLength()
    this
  }

  override def assign(values: Array[Double]): Vector = {
    if (dimension != values.length) {
      throw new DimensionException(dimension, values.length)
    }
    if (isSequentialAccess && !isAddConstantTime) {
      val updates = new OrderedIntDoubleMapping()
      val it = iterator
      while (it.hasNext) {
        val element = it.next()
        val index = element.index
        if (element.value == 0.0) {
          updates(index) = values(index)
        } else {
          element.set(values(index))
        }
      }
      mergeUpdates(updates)
    } else {
      for (i <- 0 until dimension) {
        this(i) = values(i)
      }
    }
    invalidateCachedLength()
    this
  }

  override def assign(other: Vector): Vector = assign(other, Functions.SECOND)

  override def assign(f: DoubleFunction): Vector = {
    val iter = if (!f.isDensifying) nonZeroIterator else iterator
    while (iter.hasNext) {
      val element = iter.next()
      element.set(f(element.value))
    }
    invalidateCachedLength()
    this
  }

  override def assign(other: Vector, function: DoubleDoubleFunction): Vector = {
    if (dimension != other.dimension) {
      throw new DimensionException(dimension, other.dimension)
    }
    VectorBinaryAssign.assignBest(this, other, function)
    invalidateCachedLength()
    this
  }

  override def assign(f: DoubleDoubleFunction, y: Double): Vector = {
    val iter = if (f(0, y) == 0) nonZeroIterator else iterator
    while (iter.hasNext) {
      val element = iter.next()
      element.set(f(element.value, y))
    }
    invalidateCachedLength()
    this
  }

  //  @Override
  //  public Matrix cross(Vector other) {
  //    Matrix result = matrixLike(size, other.size());
  //    Iterator<Vector.Element> it = iterateNonZero();
  //    while (it.hasNext()) {
  //      Vector.Element e = it.next();
  //      int row = e.index();
  //      result.assignRow(row, other.times(getQuick(row)));
  //    }
  //    return result;
  //  }

  override def hashCode(): Int = {
    var result = dimension
    val iter = nonZeroIterator
    while (iter.hasNext) {
      val ele = iter.next()
      //TODO: RandomUtils
      result += ele.index * HashUtils.hash(ele.value)
    }
    result
  }

  /**
   * Determines whether this [[spark.mllib.math.vector.Vector]] represents the same logical vector as another
   * object. Two [[spark.mllib.math.vector.Vector]]s are equal (regardless of implementation) if the value at
   * each index is the same, and the cardinalities are the same.
   */
  override def equals(other: Any): Boolean = {
    if (!this.canEqual(other)) return false
    if (this.## != other.##) return false

    val that = other.asInstanceOf[Vector]

    if (this eq that) return true

    dimension == that.dimension && aggregate(that, Functions.PLUS, Functions.MINUS_ABS) == 0.0
  }

  override def canEqual(that: Any): Boolean =
    that.isInstanceOf[Vector]

  override def toString: String = toString(null)

  def toString(dictionary: Array[String]): String = {
    val result = new StringBuilder()
    result.append('{')
    for (index <- 0 until dimension) {
      val value = this(index)
      if (value != 0.0) {
        result.append(if (dictionary != null && dictionary.length > index) dictionary(index) else index)
        result.append(':')
        result.append(value)
        result.append(',')
      }
    }
    if (result.length > 1) {
      result.setCharAt(result.length - 1, '}')
    } else {
      result.append('}')
    }
    result.toString()
  }

  /**
   * toString() implementation for sparse vectors via nonZeroes() method
   *
   * @return String representation of the vector
   */
  def sparseVectorToString(): String = {
    val iter = nonZeroIterator
    if (!iter.hasNext) {
      "{}"
    } else {
      val result = new StringBuilder()
      result.append('{')
      while (iter.hasNext) {
        val e = iter.next()
        result.append(e.index)
        result.append(':')
        result.append(e.value)
        result.append(',')
      }
      result.setCharAt(result.length - 1, '}')
      result.toString()
    }
  }

  override def extend(intercept: Double = 1.0): Vector = {
    val ext = like(dimension + 1)
    ext(0) = intercept
    for (i <- 0 until dimension) {
      ext(i + 1) = this(i)
    }
    ext
  }

  override def restore(): (Double, Vector) = {
    val v = like(dimension - 1)
    for (i <- 1 until dimension) {
      v(i - 1) = this(i)
    }
    (this(0), v)
  }

  protected final class LocalElement(private[AbstractVector] val _index: Int) extends Vector.Element {

    //TODO: how to this(index)
    override def value: Double = apply(index)

    override def index: Int = _index

    override def set(value: Double) {
      update(index, value)
    }
  }

  override def +(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)
    createOptimizedCopy().assign(that, Functions.PLUS)
  }

  override def +(x: Double): Vector = {
    val result = createOptimizedCopy()
    if (x == 0.0) {
      result
    } else {
      result.assign(Functions.plus(x))
    }
  }

  def +=(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)
    this.assign(that, Functions.PLUS)
  }

  def +=(x: Double): Vector = {
    if (x == 0.0) {
      this
    } else {
      this.assign(Functions.plus(x))
    }
  }

  override def -(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)
    createOptimizedCopy().assign(that, Functions.MINUS)
  }

  def -(x: Double): Vector = {
    val result = createOptimizedCopy()
    if (x == 0.0) {
      result
    } else {
      result.assign(Functions.minus(x))
    }
  }

  def -=(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)
    this.assign(that, Functions.MINUS)
  }

  def -=(x: Double): Vector = {
    if (x == 0.0) {
      this
    } else {
      this.assign(Functions.minus(x))
    }
  }

  override def *(that: Vector): Double = {
    AbstractVector.checkDimension(this, that)

    if (this eq that) {
      lengthSquared
    } else {
      aggregate(that, Functions.PLUS, Functions.MULT)
    }
  }

  override def *(x: Double): Vector = {
    if (x == 0.0) {
      this.like()
    } else {
      createOptimizedCopy().assign(Functions.mult(x))
    }
  }

  def *=(x: Double): Vector = {
    if (x == 0.0) {
      this
    } else {
      this.assign(Functions.mult(x))
    }
  }

  override def times(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    if (this.getNumNondefaultElements <= that.getNumNondefaultElements) {
      AbstractVector.createOptimizedCopy(this).assign(that, Functions.MULT)
    } else {
      AbstractVector.createOptimizedCopy(that).assign(this, Functions.MULT)
    }
  }

  override def /(x: Double): Vector = {
    if (x == 1.0) {
      return this.clone()
    }
    val result = createOptimizedCopy()
    for (element <- result.nonZeroes) {
      element.set(element.value / x)
    }
    result
  }

  override def /=(x: Double): Vector = {
    if (x == 1.0) {
      return this
    }
    for (element <- this.nonZeroes) {
      element.set(element.value / x)
    }
    this
  }

  def /(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    this.createOptimizedCopy().assign(that, Functions.DIV)
  }

  def /=(that: Vector): Vector = {
    AbstractVector.checkDimension(this, that)

    this.assign(that, Functions.DIV)
  }

}


object AbstractVector {

  private[vector] def checkDimension(v1: Vector, v2: Vector) {
    if (v1.dimension != v2.dimension) {
      throw new DimensionException(v1.dimension, v2.dimension)
    }
  }

  private[vector] def createOptimizedCopy(vector: Vector): Vector = {
    if (vector.isDense) {
      vector.like().assign(vector, Functions.SECOND_LEFT_ZERO)
    } else {
      vector.clone()
    }
  }
}
