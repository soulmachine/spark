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

import com.google.common.collect.AbstractIterator

import scala.collection.JavaConversions._
import spark.mllib.math.collection.map.OrderedIntDoubleMapping

/** Implements subset view of a Vector */
class VectorView(private val vector: Vector, private val offset: Int, dimension: Int) extends AbstractVector(dimension) {

  /** For serialization purposes only */
  def this() = this(null, 0, 0)

  //  protected override matrixLike(rows: Int, columns: Int): Matrix = {
  //    ((AbstractVector) vector).matrixLike(rows, columns)
  //  }

  override def clone(): Vector = new VectorView(this.vector.clone(), this.offset, this.dimension)

  override def isDense: Boolean = vector.isDense

  override def isSequentialAccess: Boolean = vector.isSequentialAccess

  def like(_dimension: Int = dimension): VectorView =
    new VectorView(vector.like(), offset, _dimension)

  /** Return an vector of the same underlying class as the receiver, containing values of the array. */
  def like(array: Array[Double]) =
    new VectorView(vector.like(array), offset, dimension)

  def apply(index: Int): Double = vector(offset + index)

  def update(index: Int, value: Double) {
    vector(offset + index) = value
  }

  override def getNumNondefaultElements: Int = dimension

  override def viewPart(offset: Int, length: Int): Vector = {
    if (offset < 0) {
      throw new IndexException(offset, dimension)
    }
    if (offset + length > dimension) {
      throw new IndexException(offset + length, dimension)
    }
    new VectorView(vector, offset + this.offset, length)
  }

  /** @return true if index is a valid index in the underlying Vector */
  private def isInView(index: Int): Boolean = {
    index >= offset && index < offset + dimension
  }

  protected def nonZeroIterator: Iterator[Vector.Element] = new NonZeroIterator()

  protected def iterator: Iterator[Vector.Element] = new AllIterator()

  final class NonZeroIterator extends AbstractIterator[Vector.Element] {

    private val iter = vector.nonZeroes().iterator

    protected override def computeNext(): Vector.Element = {
      while (iter.hasNext) {
        val e = iter.next()
        if (isInView(e.index) && e.get() != 0) {
          val decorated = vector.getElement(e.index)
          return new DecoratorElement(decorated)
        }
      }
      endOfData()
    }

  }

  final class AllIterator extends AbstractIterator[Vector.Element] {

    private val iter = vector.all().iterator

    protected override def computeNext(): Vector.Element = {
      while (iter.hasNext) {
        val e = iter.next()
        if (isInView(e.index)) {
          val decorated = vector.getElement(e.index)
          return new DecoratorElement(decorated)
        }
      }
      endOfData() // No element was found
    }

  }

  private final class DecoratorElement(private val decorated: Vector.Element) extends Vector.Element {

    override def get(): Double = decorated.get()

    override def index: Int = decorated.index - offset

    override def set(value: Double) {
      decorated.set(value)
    }
  }

  override def getLengthSquared: Double = {
    var result = 0.0
    for (i <- 0 until dimension) {
      val value = this(i)
      result += value * value
    }
    result
  }

  override def getDistanceSquared(other: Vector): Double = {
    var result = 0.0
    for (i <- 0 until dimension) {
      val delta = this(i) - other(i)
      result += delta * delta
    }
    result
  }

  override def getLookupCost: Double = vector.getLookupCost

  override def getIteratorAdvanceCost: Double = {
    // TODO: remove the 2x after fixing the Element iterator
    2 * vector.getIteratorAdvanceCost
  }

  override def isAddConstantTime: Boolean = vector.isAddConstantTime

  /**
   * Used internally by assign() to update multiple indices and values at once.
   * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
   * <p/>
   * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
   *
   * @param updates a mapping of indices to values to merge in the vector.
   */
  override def mergeUpdates(updates: OrderedIntDoubleMapping) {
    for (i <- 0 until updates.getNumMappings) {
      updates.setIndexAt(i, updates.indexAt(i) + offset)
    }
    vector.mergeUpdates(updates)
  }

  /** Convert to a Double Array. */
  def toArray: Array[Double] = vector.toArray.slice(offset, offset + dimension)
}
