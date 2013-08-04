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

import java.util.NoSuchElementException

import spark.mllib.math.function.{IntProcedure, IntDoubleProcedure}
import spark.mllib.math.collection.set.AbstractSet

import spark.mllib.math.collection.list.{IntArrayList, DoubleArrayList}
import java.util


/**
 * Open hash map from int keys to double values.
 **/
class OpenIntDoubleHashMap(initialCapacity: Int = AbstractSet.DEFAULT_CAPACITY,
                           _minLoadFactor: Double = AbstractSet.DEFAULT_MIN_LOAD_FACTOR,
                           _maxLoadFactor: Double = AbstractSet.DEFAULT_MAX_LOAD_FACTOR)
  extends AbstractIntDoubleMap {

  setUp(initialCapacity, _minLoadFactor, _maxLoadFactor)

  /** The hash table keys. */
  private var table: Array[Int] = _

  /** The hash table values. */
  private var values: Array[Double] = _

  /** The state of each hash table entry (FREE, FULL, REMOVED). */
  private var state: Array[Byte] = _

  /** The number of table entries in state==FREE. */
  private var freeEntries: Int = _

  /** Removes all (key,value) associations from the receiver. Implicitly calls <tt>trimToSize()</tt>. */
  override def clear() {
    util.Arrays.fill(this.state, OpenIntDoubleHashMap.FREE)
    distinct = 0
    freeEntries = table.length // delta
    trimToSize()
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  override def clone(): OpenIntDoubleHashMap = {
    val copy = super.clone().asInstanceOf[OpenIntDoubleHashMap]
    copy.table = copy.table.clone()
    copy.values = copy.values.clone()
    copy.state = copy.state.clone()
    copy
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  override def containsKey(key: Int): Boolean = indexOfKey(key) >= 0

  /**
   * Returns <tt>true</tt> if the receiver contains the specified value.
   *
   * @return <tt>true</tt> if the receiver contains the specified value.
   */
  override def containsValue(value: Double): Boolean = indexOfValue(value) >= 0

  /**
   * Ensures that the receiver can hold at least the specified number of associations without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver. <p> This
   * method never need be called; it is for performance tuning only. Calling this method before <tt>put()</tt>ing a
   * large number of associations boosts performance, because the receiver will grow only once instead of potentially
   * many times and hash collisions get less probable.
   *
   * @param minCapacity the desired minimum capacity.
   */
  override def ensureCapacity(minCapacity: Int) {
    if (table.length < minCapacity) {
      val newCapacity = nextPrime(minCapacity)
      rehash(newCapacity)
    }
  }

  /**
   * Applies a procedure to each key of the receiver, if any. Note: Iterates over the keys in no particular order.
   * Subclasses can define a particular order, for example, "sorted by key". All methods which <i>can</i> be expressed
   * in terms of this method (most methods can) <i>must guarantee</i> to use the <i>same</i> order defined by this
   * method, even if it is no particular order. This is necessary so that, for example, methods <tt>keys</tt> and
   * <tt>values</tt> will yield association pairs, not two uncorrelated lists.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  override def forEachKey(procedure: IntProcedure): Boolean = {
    for (i <- 0 until table.length) {
      if (state(i) == OpenIntDoubleHashMap.FULL && !procedure(table(i))) {
        return false
      }
    }
    true
  }

  /**
   * Applies a procedure to each (key,value) pair of the receiver, if any. Iteration order is guaranteed to be
   * <i>identical</i> to the order used by method `#forEachKey(IntProcedure)`.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  override def forEachPair(procedure: IntDoubleProcedure): Boolean = {
    for (i <- 0 until table.length) {
      if (state(i) == OpenIntDoubleHashMap.FULL && !procedure(table(i), values(i))) {
        return false
      }
    }
    true
  }

  /**
   * Returns the value associated with the specified key. It is often a good idea to first check with
   * containsKey(int) whether the given key has a value associated or not, i.e. whether there exists an association
   * for the given key or not.
   *
   * @param key the key to be searched for.
   * @return the value associated with the specified key; <tt>0</tt> if no such key is present.
   */
  override def apply(key: Int): Double = {
    val i = indexOfKey(key)
    if (i < 0) 0 else values(i)
  }

  /**
   * @param key the key to be added to the receiver.
   * @return the index where the key would need to be inserted, if it is not already contained. Returns -index-1 if the
   *         key is already contained at slot index. Therefore, if the returned index < 0, then it is already contained
   *         at slot -index-1. If the returned index >= 0, then it is NOT already contained and should be inserted at
   *         slot index.
   */
  protected def indexOfInsertion(key: Int): Int = {
    val length = table.length

    val hash = HashFunctions.hash(key) & 0x7FFFFFFF
    var i = hash % length

    val tmp = hash % (length - 2) // double hashing, see http://www.eece.unm.edu/faculty/heileman/hash/node4.html
    //val decrement = (hash / length) % length;
    val decrement = if (tmp == 0) 1 else tmp

    // stop if we find a removed or free slot, or if we find the key itself
    // do NOT skip over removed slots (yes, open addressing is like that...)
    while (state(i) == OpenIntDoubleHashMap.FULL && table(i) != key) {
      i -= decrement
      //hashCollisions += 1
      if (i < 0) {
        i += length
      }
    }

    if (state(i) == OpenIntDoubleHashMap.REMOVED) {
      // stop if we find a free slot, or if we find the key itself.
      // do skip over removed slots (yes, open addressing is like that...)
      // assertion: there is at least one FREE slot.
      val j = i
      while (state(i) != OpenIntDoubleHashMap.FREE && (state(i) == OpenIntDoubleHashMap.REMOVED || table(i) != key)) {
        i -= decrement
        //hashCollisions += 1
        if (i < 0) {
          i += length
        }
      }
      if (state(i) == OpenIntDoubleHashMap.FREE) {
        i = j
      }
    }

    if (state(i) == OpenIntDoubleHashMap.FULL) {
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
  protected def indexOfKey(key: Int): Int = {
    val length = table.length

    val hash = HashFunctions.hash(key) & 0x7FFFFFFF
    var i = hash % length

    val tmp = hash % (length - 2) // double hashing, see http://www.eece.unm.edu/faculty/heileman/hash/node4.html
    //val decrement = (hash / length) % length;
    val decrement = if (tmp == 0) 1 else tmp

    // stop if we find a free slot, or if we find the key itself.
    // do skip over removed slots (yes, open addressing is like that...)
    while (state(i) != OpenIntDoubleHashMap.FREE && (state(i) == OpenIntDoubleHashMap.REMOVED || table(i) != key)) {
      i -= decrement
      //hashCollisions += 1
      if (i < 0) {
        i += length
      }
    }

    if (state(i) == OpenIntDoubleHashMap.FREE) {
      -1 // not found
    } else {
      i //found, return index where key is contained
    }
  }

  /**
   * @param value the value to be searched in the receiver.
   * @return the index where the value is contained in the receiver, returns -1 if the value was not found.
   */
  protected def indexOfValue(value: Double): Int = {
    for (i <- 0 until state.length) {
      if (state(i) == OpenIntDoubleHashMap.FULL && values(i) == value) {
        return i
      }
    }

    -1 // not found
  }

  /**
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method `#forEachKey(IntProcedure)`.
   * <p> This method can be used
   * to iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  override def getKeys(list: IntArrayList) {
    list.setSize(distinct)
    val elements = list.getElements

    var j = 0
    for (i <- 0 until table.length) {
      if (state(i) == OpenIntDoubleHashMap.FULL) {
        elements(j) = table(i)
        j += 1
      }
    }
  }

  def iterator: util.Iterator[MapElement] = new MapIterator()

  final class MapElement {
    /** offset of the array. */
    private var offset = -1
    /** number of valid pairs that have met. */
    private[OpenIntDoubleHashMap] var seen = 0

    def advanceOffset(): Boolean = {
      offset += 1
      while (offset < state.length && state(offset) != OpenIntDoubleHashMap.FULL) {
        offset += 1
      }
      if (offset < state.length) {
        seen += 1
      }
      offset < state.length
    }

    def get(): Double = values(offset)

    def index: Int = table(offset)

    def set(value: Double) {
      values(offset) = value
    }
  }

  final class MapIterator extends util.Iterator[MapElement] {
    private val element = new MapElement()

    //private MapIterator() { }

    override def hasNext: Boolean = element.seen < distinct

    override def next(): MapElement = {
      if (element.advanceOffset()) element else throw new NoSuchElementException()
    }

    override def remove() {
      throw new UnsupportedOperationException()
    }
  }

  /**
   * Fills all pairs satisfying a given condition into the specified lists. Fills into the lists, starting at index 0.
   * After this call returns the specified lists both have a new size, the number of pairs satisfying the condition.
   * Iteration order is guaranteed to be <i>identical</i> to the order used by method
   * `#forEachKey(IntProcedure)`. <p> <b>Example:</b> <br>
   * <pre>
   * IntDoubleProcedure condition = new IntDoubleProcedure() { // match even values only
   * public boolean apply(int key, double value) { return value%2==0; }
   * }
   * keys = (8,7,6), values = (1,2,2) --> keyList = (6,8), valueList = (2,1)</tt>
   * </pre>
   *
   * @param condition the condition to be matched. Takes the current key as first and the current value as second
   *                  argument.
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  override def pairsMatching(condition: IntDoubleProcedure,
                             keyList: IntArrayList,
                             valueList: DoubleArrayList) {
    keyList.clear()
    valueList.clear()

    for (i <- 0 until table.length) {
      if (state(i) == OpenIntDoubleHashMap.FULL && condition(table(i), values(i))) {
        keyList.add(table(i))
        valueList.add(values(i))
      }
    }
  }

  /**
   * Associates the given key with the given value. Replaces any old <tt>(key,someOtherValue)</tt> association, if
   * existing.
   *
   * @param key   the key the value shall be associated with.
   * @param value the value to be associated.
   * @return <tt>true</tt> if the receiver did not already contain such a key; <tt>false</tt> if the receiver did
   *         already contain such a key - the new value has now replaced the formerly associated value.
   */
  override def update(key: Int, value: Double) {
    var i = indexOfInsertion(key)
    if (i < 0) {
      // already contained
      i = -i - 1
      this.values(i) = value
      return
    }

    if (this.distinct > this.highWaterMark) {
      val newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor)
      rehash(newCapacity)
      this(key) = value
      return
    }

    this.table(i) = key
    this.values(i) = value
    if (this.state(i) == OpenIntDoubleHashMap.FREE) {
      this.freeEntries -= 1
    }
    this.state(i) = OpenIntDoubleHashMap.FULL
    this.distinct += 1

    if (this.freeEntries < 1) {
      //delta
      val newCapacity = chooseGrowCapacity(this.distinct + 1,
        this.minLoadFactor, this.maxLoadFactor)
      rehash(newCapacity)
    }
  }

  override def adjustOrPutValue(key: Int, newValue: Double, incrValue: Double): Double = {
    var i = indexOfInsertion(key)
    if (i < 0) {
      //already contained
      i = -i - 1
      this.values(i) += incrValue
      this.values(i)
    } else {
      this.values(key) = newValue
      newValue
    }
  }

  /**
   * Rehashes the contents of the receiver into a new table with a smaller or larger capacity. This method is called
   * automatically when the number of keys in the receiver exceeds the high water mark or falls below the low water
   * mark.
   */
  protected def rehash(newCapacity: Int) {
    val oldCapacity = table.length
    //if (oldCapacity == newCapacity) return;

    val oldTable = table
    val oldValues = values
    val oldState = state

    this.table = new Array[Int](newCapacity)
    this.values = new Array[Double](newCapacity)
    this.state = new Array[Byte](newCapacity)

    this.lowWaterMark = chooseLowWaterMark(newCapacity, this.minLoadFactor)
    this.highWaterMark = chooseHighWaterMark(newCapacity, this.maxLoadFactor)

    this.freeEntries = newCapacity - this.distinct // delta

    for (i <- 0 until oldCapacity) {
      if (oldState(i) == OpenIntDoubleHashMap.FULL) {
        val element = oldTable(i)
        val index = indexOfInsertion(element)
        this.table(index) = element
        this.values(index) = oldValues(i)
        this.state(index) = OpenIntDoubleHashMap.FULL
      }
    }
  }

  /**
   * Removes the given key with its associated element from the receiver, if present.
   *
   * @param key the key to be removed from the receiver.
   * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
   */
  override def removeKey(key: Int): Boolean = {
    val i = indexOfKey(key)
    if (i < 0) {
      return false
    } // key not contained

    this.state(i) = OpenIntDoubleHashMap.REMOVED
    //this.values(i)=0; // delta
    this.distinct -= 1

    if (this.distinct < this.lowWaterMark) {
      val newCapacity = chooseShrinkCapacity(this.distinct, this.minLoadFactor, this.maxLoadFactor)
      rehash(newCapacity)
    }

    true
  }

  /**
   * Initializes the receiver.
   *
   * @param initialCapacity the initial capacity of the receiver.
   * @param minLoadFactor   the minLoadFactor of the receiver.
   * @param maxLoadFactor   the maxLoadFactor of the receiver.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   */
  final protected override def setUp(initialCapacity: Int,
                                     minLoadFactor: Double, maxLoadFactor: Double) {
    var capacity = initialCapacity
    super.setUp(capacity, minLoadFactor, maxLoadFactor)
    capacity = nextPrime(capacity)
    if (capacity == 0) {
      capacity = 1
    } // open addressing needs at least one FREE slot at any time.

    this.table = new Array[Int](capacity)
    this.values = new Array[Double](capacity)
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
   * Fills all values contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method `#forEachKey(IntProcedure)`.
   * <p> This method can be used
   * to iterate over the values of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  override def getValues(list: DoubleArrayList) {
    list.setSize(distinct)
    val elements = list.getElements

    var j = 0
    for (i <- 0 until state.length) {
      if (state(i) == OpenIntDoubleHashMap.FULL) {
        elements(j) = values(i)
        j += 1
      }
    }
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

object OpenIntDoubleHashMap {
  protected val FREE = 0.toByte
  protected val FULL = 1.toByte
  protected val REMOVED = 2.toByte
  protected val NO_KEY_VALUE = 0.toByte
}