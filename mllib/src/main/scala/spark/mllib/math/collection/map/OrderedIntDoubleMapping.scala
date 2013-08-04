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

package spark.mllib.math.collection.map

import java.io.Serializable

// If noDefault = true, doesn't allow DEFAULT_VALUEs in the mapping (adding a zero discards it). Otherwise, a DEFAULT_VALUE is
// treated like any other value.
private[math] final class OrderedIntDoubleMapping(
    private var indices: Array[Int],
    private var values: Array[Double],
    private var numMappings: Int) extends Serializable with Cloneable {

  // If true, doesn't allow DEFAULT_VALUEs in the mapping (adding a zero discards it). Otherwise, a DEFAULT_VALUE is
  // treated like any other value.
  private var noDefault = true

  def this(capacity: Int) = this(new Array[Int](capacity), new Array[Double](capacity), 0)

  // no-arg constructor for deserializer
  def this()  = this(11)

  def this(noDefault: Boolean) = {
    this()
    this.noDefault = noDefault
  }

  def getIndices(): Array[Int] = indices

  def indexAt(offset: Int): Int = indices(offset)

  def setIndexAt(offset: Int, index: Int) {
    indices(offset) = index
  }

  def getValues(): Array[Double] = values

  def setValueAt(offset: Int, value: Double) {
    values(offset) = value
  }

  def getNumMappings: Int = numMappings

  private def growTo(newCapacity: Int) {
    if (newCapacity > indices.length) {
      val newIndices = new Array[Int](newCapacity)
      Array.copy(indices, 0, newIndices, 0, numMappings)
      indices = newIndices
      val newValues = new Array[Double](newCapacity)
      Array.copy(values, 0, newValues, 0, numMappings)
      values = newValues
    }
  }

  // binary search
  private def find(index: Int): Int = {
    var low = 0
    var high = numMappings - 1
    while (low <= high) {
      val mid = low + (high - low >>> 1)
      val midVal = indices(mid)
      if (midVal < index) {
        low = mid + 1
      } else if (midVal > index) {
        high = mid - 1
      } else {
        return mid
      }
    }
    return -(low + 1)
  }

  def apply(index: Int): Double = {
    val offset = find(index)
    if (offset >= 0 ) values(offset) else OrderedIntDoubleMapping.DEFAULT_VALUE
  }

  def update(index: Int, value: Double) {
    if (numMappings == 0 || index > indices(numMappings - 1)) {
      if (!noDefault || value != OrderedIntDoubleMapping.DEFAULT_VALUE) {
        if (numMappings >= indices.length) {
          growTo(math.max((OrderedIntDoubleMapping.GROW_FACTOR * numMappings).toInt, numMappings + 1))
        }
        indices(numMappings) = index
        values(numMappings) = value
        numMappings += 1
      }
    } else {
      val offset = find(index)
      if (offset >= 0) {
        removeOrUpdateValueIfPresent(offset, value)
      } else {
        insertValueIfNotDefault(index, offset, value)
      }
    }
  }

  /**
   * Merges the updates in linear time by allocating new arrays and iterating through the existing indices and values
   * and the updates' indices and values at the same time while selecting the minimum index to set at each step.
   * @param updates another list of mappings to be merged in.
   */
  def merge(updates: OrderedIntDoubleMapping) {
    val updateIndices = updates.getIndices()
    val updateValues = updates.getValues()

    val newNumMappings = numMappings + updates.getNumMappings
    val newCapacity = math.max((OrderedIntDoubleMapping.GROW_FACTOR * newNumMappings).toInt, newNumMappings + 1)
    val newIndices = new Array[Int](newCapacity)
    val newValues = new Array[Double](newCapacity)

    var k = 0
    var i = 0
    var j = 0
    while (i < numMappings && j < updates.getNumMappings) {
      if (indices(i) < updateIndices(j)) {
        newIndices(k) = indices(i)
        newValues(k) = values(i)
        i += 1
      } else if (indices(i) > updateIndices(j)) {
        newIndices(k) = updateIndices(j)
        newValues(k) = updateValues(j)
        j += 1
      } else {
        newIndices(k) = updateIndices(j)
        newValues(k) = updateValues(j)
        i += 1
        j += 1
      }
      k += 1
    }

    while (i < numMappings) {
      newIndices(k) = indices(i)
      newValues(k) = values(i)
      i += 1
      k += 1
    }
    while (j < updates.getNumMappings) {
      newIndices(k) = updateIndices(j)
      newValues(k) = updateValues(j)
      j += 1
      k += 1
    }

    indices = newIndices
    values = newValues
    numMappings = k
  }

  override def hashCode(): Int = {
    var result = 0
    for (i <- 0 until numMappings) {
      result = 31 * result + indices(i)
      result = 31 * result + java.lang.Double.doubleToRawLongBits(values(i)).toInt
    }
    result
  }

  override def equals(o: Any): Boolean = {
    if (o.isInstanceOf[OrderedIntDoubleMapping]) {
      val other = o.asInstanceOf[OrderedIntDoubleMapping]
      if (numMappings == other.numMappings) {
        for (i <- 0 until numMappings) {
          if (indices(i) != other.indices(i) || values(i) != other.values(i)) {
            return false
          }
        }
        return true
      }
    }
    return false
  }

  override def toString: String = {
    val result = new StringBuilder(10 * numMappings)
    for (i <- 0 until numMappings) {
      result.append('(')
      result.append(indices(i))
      result.append(',')
      result.append(values(i))
      result.append(')')
    }
    result.toString()
  }

  //@SuppressWarnings("CloneDoesntCallSuperClone")
  override def clone(): OrderedIntDoubleMapping = {
    new OrderedIntDoubleMapping(indices.clone(), values.clone(), numMappings)
  }

  def increment(index: Int, increment: Double) {
    val offset = find(index)
    if (offset >= 0) {
      val newValue = values(offset) + increment
      removeOrUpdateValueIfPresent(offset, newValue)
    } else {
      insertValueIfNotDefault(index, offset, increment)
    }
  }

  private def insertValueIfNotDefault(index: Int, offset: Int, value: Double) {
    if (!noDefault || value != OrderedIntDoubleMapping.DEFAULT_VALUE) {
      if (numMappings >= indices.length) {
        growTo(math.max((OrderedIntDoubleMapping.GROW_FACTOR * numMappings).toInt, numMappings + 1))
      }
      val at = -offset - 1
      if (numMappings > at) {
        var i = numMappings - 1
        var j = numMappings
        while (i >= at) {
          indices(j) = indices(i)
          values(j) = values(i)
          i -= 1
          j -= 1
        }
      }
      indices(at) = index
      values(at) = value
      numMappings += 1
    }
  }

  private def removeOrUpdateValueIfPresent(offset: Int, newValue: Double) {
    if (noDefault && newValue == OrderedIntDoubleMapping.DEFAULT_VALUE) {
      var i = offset + 1
      var j = offset
      while (i < numMappings) {
        indices(j) = indices(i)
        values(j) = values(i)
        i += 1
        j += 1
      }
      numMappings -= 1
    } else {
      values(offset) = newValue
    }
  }
}

final object OrderedIntDoubleMapping {
  val DEFAULT_VALUE = 0.0
  private val GROW_FACTOR = 1.2
}
