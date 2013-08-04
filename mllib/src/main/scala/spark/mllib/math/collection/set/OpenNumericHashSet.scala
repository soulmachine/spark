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

package spark.mllib.math.collection.set

import java.nio.IntBuffer

import spark.mllib.math.numeric.Numeric
import spark.mllib.math.numeric.FastImplicits._

import spark.mllib.math.function.NumericProcedure
import spark.mllib.math.collection.list.NumericArrayList
import spark.mllib.math.collection.map.{HashFunctions, PrimeFinder}

/** Open hash set of numeric(Int, Long, Float, Double) items. */
class OpenNumericHashSet[T: Numeric : ClassManifest](
  initialCapacity: Int = AbstractSet.DEFAULT_CAPACITY,
  _minLoadFactor: Double = AbstractSet.DEFAULT_MIN_LOAD_FACTOR,
  _maxLoadFactor: Double = AbstractSet.DEFAULT_MAX_LOAD_FACTOR
) extends AbstractSet {
  setUp(initialCapacity, _minLoadFactor, _maxLoadFactor)

  /** The hash table keys. */
  protected var table: Array[T] = _

  /** The state of each hash table entry (FREE, FULL, REMOVED). */
  protected var state: Array[Byte] = _

  /** The number of table entries in state==FREE. */
  protected var freeEntries: Int = _

  override def clear() {
    java.util.Arrays.fill(this.state, OpenNumericHashSet.FREE)
    distinct = 0
    freeEntries = table.length // delta
    trimToSize()
  }

  override def clone(): OpenNumericHashSet[T] = {
    val copy = super.clone().asInstanceOf[OpenNumericHashSet[T]]
    copy.table = this.table.clone()
    copy.state = this.state.clone()
    copy
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  def contains(key: T): Boolean = indexOfKey(key) >= 0

  override def ensureCapacity(minCapacity: Int) {
    if (table.length < minCapacity) {
      val newCapacity = nextPrime(minCapacity)
      rehash(newCapacity)
    }
  }

  override def equals(obj: Any): Boolean = {
    if (!obj.isInstanceOf[OpenNumericHashSet[T]]) {
      return false
    }
    val other = obj.asInstanceOf[OpenNumericHashSet[T]]

    if (this eq other) {
      return true
    }

    if (other.size != size) {
      return false
    }

    forEachKey(new NumericProcedure[T]() {
      override def apply(key: T): Boolean = {
        other.contains(key)
      }
    })
  }

  override def hashCode(): Int = {
    val buf = new Array[Int](size)
    forEachKey(new NumericProcedure[T]() {
      var i = 0

      override def apply(iterKey: T): Boolean = {
        //TODO:
        buf(i) = HashUtils.hash(iterKey.toInt)
        i += 1
        true
      }
    })
    java.util.Arrays.sort(buf)
    IntBuffer.wrap(buf).hashCode()
  }

  /**
   * Applies a procedure to each key of the receiver, if any.
   */
  def forEachKey(procedure: NumericProcedure[T]): Boolean = {
    for (i <- 0 until table.length) {
      if (state(i) == OpenNumericHashSet.FULL) {
        if (!procedure(table(i))) {
          return false
        }
      }
    }
    true
  }

  /**
   * @param key the key to be added to the receiver.
   * @return the index where the key would need to be inserted, if it is not already contained. Returns -index-1 if the
   *         key is already contained at slot index. Therefore, if the returned index < 0, then it is already contained
   *         at slot -index-1. If the returned index >= 0, then it is NOT already contained and should be inserted at
   *         slot index.
   */
  private def indexOfInsertion(key: T): Int = {
    val length = table.length

    //TODO: Double 和 Float 的 Hash
    val hash = HashFunctions.hash(key.toInt) & 0x7FFFFFFF
    var i = hash % length
    val tmp = hash % (length - 2) // double hashing, see http://www.eece.unm.edu/faculty/heileman/hash/node4.html
    //int decrement = (hash / length) % length;
    val decrement = if (tmp == 0) 1 else tmp

    // stop if we find a removed or free slot, or if we find the key itself
    // do NOT skip over removed slots (yes, open addressing is like that...)
    while (state(i) == OpenNumericHashSet.FULL && table(i) != key) {
      i -= decrement
      //hashCollisions++;
      if (i < 0) {
        i += length
      }
    }

    if (state(i) == OpenNumericHashSet.REMOVED) {
      // stop if we find a free slot, or if we find the key itself.
      // do skip over removed slots (yes, open addressing is like that...)
      // assertion: there is at least one FREE slot.
      val j = i
      while (state(i) != OpenNumericHashSet.FREE && (state(i) == OpenNumericHashSet.REMOVED || table(i) != key)) {
        i -= decrement
        //hashCollisions++;
        if (i < 0) {
          i += length
        }
      }
      if (state(i) == OpenNumericHashSet.FREE) {
        i = j
      }
    }


    if (state(i) == OpenNumericHashSet.FULL) {
      // key already contained at slot i.
      // return a negative number identifying the slot.
      -i - 1
    } else {
      // not already contained, should be inserted at slot i.
      // return a number >= 0 identifying the slot.
      i
    }
  }

  /**
   * @param key the key to be searched in the receiver.
   * @return the index where the key is contained in the receiver, returns -1 if the key was not found.
   */
  private def indexOfKey(key: T): Int = {
    val length = table.length

    //TODO:Double and Float
    val hash = HashFunctions.hash(key.toInt) & 0x7FFFFFFF
    var i = hash % length
    val tmp = hash % (length - 2) // double hashing, see http://www.eece.unm.edu/faculty/heileman/hash/node4.html
    //int decrement = (hash / length) % length;
    val decrement = if (tmp == 0) 1 else tmp

    // stop if we find a free slot, or if we find the key itself.
    // do skip over removed slots (yes, open addressing is like that...)
    while (state(i) != OpenNumericHashSet.FREE && (state(i) == OpenNumericHashSet.REMOVED || table(i) != key)) {
      i -= decrement
      //hashCollisions++;
      if (i < 0) {
        i += length
      }
    }

    if (state(i) == OpenNumericHashSet.FREE) {
      -1 // not found
    } else {
      i //found, return index where key is contained
    }
  }

  /**
   * Returns a list filled with all keys contained in the receiver. The returned list has a size that equals
   * <tt>this.size()</tt>. Iteration order is guaranteed to be <i>identical</i> to the order used by method
   * `#forEachKey(DoubleProcedure)`. <p> This method can be used to iterate over the keys of the receiver.
   *
   * @return the keys.
   */
  def keys(): NumericArrayList[T] = {
    val list = new NumericArrayList[T](size)
    keys(list)
    list
  }

  /**
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method `#forEachKey(DoubleProcedure)`.
   * <p> This method can be used to
   * iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  def keys(list: NumericArrayList[T]) {
    list.setSize(distinct)
    val elements = list.getElements

    var j = 0
    for (i <- 0 until table.length) {
      if (state(i) == OpenNumericHashSet.FULL) {
        elements(j) = table(i)
        j += 1
      }
    }
  }

  /**
   * Associates the given key with the given value. Replaces any old <tt>(key,someOtherValue)</tt> association, if
   * existing.
   *
   * @param key   the key the value shall be associated with.
   * @return <tt>true</tt> if the receiver did not already contain such a key; <tt>false</tt> if the receiver did
   *         already contain such a key - the new value has now replaced the formerly associated value.
   */
  def add(key: T): Boolean = {
    val i = indexOfInsertion(key)
    if (i < 0) {
      //already contained
      //i = -i - 1;
      return false
    }

    if (this.distinct > this.highWaterMark) {
      val newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor)
      rehash(newCapacity)
      return add(key)
    }

    this.table(i) = key
    if (this.state(i) == OpenNumericHashSet.FREE) {
      this.freeEntries -= 1
    }
    this.state(i) = OpenNumericHashSet.FULL
    this.distinct += 1

    if (this.freeEntries < 1) {
      //delta
      val newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor)
      rehash(newCapacity)
    }

    true
  }

  /**
   * Rehashes the contents of the receiver into a new table with a smaller or larger capacity. This method is called
   * automatically when the number of keys in the receiver exceeds the high water mark or falls below the low water
   * mark.
   */
  private def rehash(newCapacity: Int) {
    val oldCapacity = table.length
    //if (oldCapacity == newCapacity) return;

    val oldTable = table
    val oldState = state

    this.table = new Array[T](newCapacity)
    this.state = new Array[Byte](newCapacity)

    this.lowWaterMark = chooseLowWaterMark(newCapacity, this.minLoadFactor)
    this.highWaterMark = chooseHighWaterMark(newCapacity, this.maxLoadFactor)

    this.freeEntries = newCapacity - this.distinct // delta

    for (i <- 0 until oldCapacity) {
      if (oldState(i) == OpenNumericHashSet.FULL) {
        val element = oldTable(i)
        val index = indexOfInsertion(element)
        this.table(index) = element
        this.state(index) = OpenNumericHashSet.FULL
      }
    }
  }

  /**
   * Removes the given key with its associated element from the receiver, if present.
   *
   * @param key the key to be removed from the receiver.
   * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
   */
  def remove(key: T): Boolean = {
    val i = indexOfKey(key)
    if (i < 0) {
      return false
    } // key not contained

    this.state(i) = OpenNumericHashSet.REMOVED
    this.distinct -= 1

    if (this.distinct < this.lowWaterMark) {
      val newCapacity = chooseShrinkCapacity(this.distinct, this.minLoadFactor, this.maxLoadFactor)
      rehash(newCapacity)
    }

    true
  }

  final protected override def setUp(initialCapacity: Int, minLoadFactor: Double, maxLoadFactor: Double) {
    var capacity = initialCapacity
    super.setUp(capacity, minLoadFactor, maxLoadFactor)
    capacity = nextPrime(capacity)
    if (capacity == 0) {
      capacity = 1
    } // open addressing needs at least one FREE slot at any time.

    this.table = new Array[T](capacity)
    this.state = new Array[Byte](capacity)

    // memory will be exhausted long before this pathological case happens, anyway.
    this.minLoadFactor = minLoadFactor
    if (capacity == PrimeFinder.LARGEST_PRIME) {
      this.maxLoadFactor = 1.0
    } else {
      this.maxLoadFactor = maxLoadFactor
    }

    this.distinct = 0
    this.freeEntries = capacity // delta

    // lowWaterMark will be established upon first expansion.
    // establishing it now (upon instance construction) would immediately make the table shrink upon first put(...).
    // After all the idea of an "initialCapacity" implies violating lowWaterMarks when an object is young.
    // See ensureCapacity(...)
    this.lowWaterMark = 0
    this.highWaterMark = chooseHighWaterMark(capacity, this.maxLoadFactor)
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. Releases any superfluous internal memory. An
   * application can use this operation to minimize the storage of the receiver.
   */
  override def trimToSize() {
    // * 1.2 because open addressing's performance exponentially degrades beyond that point
    // so that even rehashing the table can take very long
    val newCapacity = nextPrime((1 + 1.2 * size).toInt)
    if (table.length > newCapacity) {
      rehash(newCapacity)
    }
  }

  /**
   * Returns a string representation of the receiver, containing the String representation of each key-value pair,
   * sorted ascending by key.
   */
  override def toString: String = {
    val theKeys = keys()
    //theKeys.sort();

    val buf = new StringBuilder()
    buf.append('[')
    for (i <- 0 until theKeys.size) {
      val key = theKeys(i)
      buf.append(key.toString)
      if (i < (theKeys.size - 1)) {
        buf.append(", ")
      }
    }
    buf.append(']')
    buf.toString()
  }

  /**
   * Access for unit tests.
   *
   * @return (capacity: Int, minLoadFactor: Double, maxLoadFactor: Double)
   */
  protected def getInternalFactors: (Int, Double, Double) = {
    (table.length, this.minLoadFactor, this.maxLoadFactor)
  }
}

object OpenNumericHashSet {
  val FREE = 0.toByte
  val FULL = 1.toByte
  val REMOVED = 2.toByte
  val NO_KEY_VALUE = Double.NaN
}
