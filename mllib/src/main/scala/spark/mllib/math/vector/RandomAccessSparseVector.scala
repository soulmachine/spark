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

import scala.collection.JavaConversions._
import spark.mllib.math.collection.map.{OpenIntDoubleHashMap, OrderedIntDoubleMapping}
import spark.mllib.math.collection.set.AbstractSet
import java.util

/** Implements sparse vector, based on SparseDoubleMatrix1D  of colt(http://acs.lbl.gov/software/colt/). */
class RandomAccessSparseVector private(
  _dimension: Int,
  private val values: OpenIntDoubleHashMap = new OpenIntDoubleHashMap(RandomAccessSparseVector.INITIAL_CAPACITY)
) extends AbstractVector(_dimension) {

  def this(dimension: Int, initialCapacity: Int) = {
    this(dimension, new OpenIntDoubleHashMap(initialCapacity))
  }

  def this(_dimension: Int) = {
    this(_dimension, math.min(_dimension, RandomAccessSparseVector.INITIAL_CAPACITY))
  }

  /** For serialization purposes only. */
  def this() = this(0)

  def this(other: Vector) = {
    this(other.dimension, other.getNumNondefaultElements)
    for (e <- other.nonZeroes) {
      values(e.index) = e.value
    }
  }

  def this(other: RandomAccessSparseVector, shallowCopy: Boolean) = {
    this(other.dimension, if (shallowCopy) other.values else other.values.clone())
  }

  def this(array: Array[Double]) = {
    this(array.length)
    for (i <- 0 until array.length) {
      this(i) = array(i)
    }
  }

  override def clone(): RandomAccessSparseVector = {
    new RandomAccessSparseVector(this, false)
  }

  def like(_diminsion: Int): Vector = new RandomAccessSparseVector(_dimension)

  def like(array: Array[Double]): Vector = new RandomAccessSparseVector(array)

  override def toString: String = sparseVectorToString()

  def toArray: Array[Double] = {
    val result = new Array[Double](dimension)
    for (e <- nonZeroes) {
      result(e.index) = e.value
    }
    result
  }

  //TODO:
  //  protected override matrixLike(rows: Int, columns: Int): Matrix = {
  //    new SparseRowMatrix(rows, columns)
  //  }

  override def assign(other: Vector): Vector = {
    if (dimension != other.dimension) {
      throw new DimensionException(dimension, other.dimension)
    }
    values.clear()
    for (e <- other.nonZeroes) {
      this(e.index) = e.value
    }
    this
  }

  override def mergeUpdates(updates: OrderedIntDoubleMapping) {
    val indices = updates.getIndices()
    val values = updates.getValues()
    for (i <- 0 until updates.getNumMappings) {
      values(indices(i)) = values(i)
    }
  }

  override def isDense: Boolean = false

  override def isSequentialAccess: Boolean = false

  def apply(i: Int): Double = values(i)

  override def update(index: Int, value: Double) {
    invalidateCachedLength()
    if (value == 0.0) {
      values.removeKey(index)
    } else {
      values(index) = value
    }
  }

  override def incrementQuick(index: Int, increment: Double) {
    invalidateCachedLength()
    values.adjustOrPutValue(index, increment, increment)
  }

  override def getNumNondefaultElements: Int = values.size

  override def getLookupCost: Double = 1.0

  override def getIteratorAdvanceCost: Double = {
    1 + (AbstractSet.DEFAULT_MAX_LOAD_FACTOR + AbstractSet.DEFAULT_MIN_LOAD_FACTOR) / 2
  }

  /**
   * This is "sort of" constant, but really it might resize the array.
   */
  override def isAddConstantTime: Boolean = true

  /*
  @Override
  public Element getElement(int index) {
    // TODO: this should return a MapElement so as to avoid hashing for both getQuick and setQuick.
    return super.getElement(index);
  }
   */

  /**
   * NOTE: this implementation reuses the Vector.Element instance for each call of next(). If you need to preserve the
   * instance, you need to make a copy of it
   *
   * @return an { @link Iterator} over the Elements.
   * @see #getElement(int)
   */
  override def nonZeroIterator: Iterator[Vector.Element] = {
    new NonDefaultIterator()
  }

  override def iterator: Iterator[Vector.Element] = {
    new AllIterator()
  }

  private final class NonDefaultIterator extends util.Iterator[Vector.Element] {

    private final class NonDefaultElement extends Vector.Element {

      override def value: Double = mapElement.get()

      override def index = mapElement.index

      override def set(value: Double) {
        invalidateCachedLength()
        mapElement.set(value)
      }
    }

    private var mapElement: values.MapElement = _
    private val element = new NonDefaultElement()
    private val iter = values.iterator

    override def hasNext: Boolean = iter.hasNext

    override def next(): Vector.Element = {
      mapElement = iter.next() // This will throw an exception at the end of enumeration.
      element
    }

    override def remove() {
      throw new UnsupportedOperationException()
    }
  }

  private final class AllIterator extends util.Iterator[Vector.Element] {
    private val element = new RandomAccessElement()

    override def hasNext: Boolean = element.index + 1 < dimension

    override def next(): Vector.Element = {
      if (!hasNext) {
        throw new NoSuchElementException()
      } else {
        element.index += 1
        element
      }
    }

    override def remove() {
      throw new UnsupportedOperationException()
    }
  }

  private final class RandomAccessElement(var index: Int = -1) extends Vector.Element {

    override def value: Double = values(index)

    override def set(value: Double) {
      invalidateCachedLength()
      if (value == 0.0) {
        values.removeKey(index)
      } else {
        values(index) = value
      }
    }
  }

}

object RandomAccessSparseVector {
  val INITIAL_CAPACITY = 11

  def apply(dimension: Int, pairs: (Int, Double)*): RandomAccessSparseVector = {
    val v = new RandomAccessSparseVector(dimension)
    pairs.map(p => v(p._1) = p._2)
    v
  }
}
