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

import java.nio.IntBuffer

import spark.mllib.math.Sorting
import spark.mllib.math.Swapper
import spark.mllib.math.collection.set.HashUtils

import spark.mllib.math.function.IntProcedure
import spark.mllib.math.function.IntDoubleProcedure

import spark.mllib.math.function.NumericComparator
import spark.mllib.math.function.DoubleFunction

import spark.mllib.math.collection.list.IntArrayList
import spark.mllib.math.collection.list.DoubleArrayList

import spark.mllib.math.collection.set.AbstractSet

abstract class AbstractIntDoubleMap extends AbstractSet {

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  def containsKey(key: Int): Boolean = {
    !forEachKey((iterKey:Int)=> key != iterKey)
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified value.
   *
   * @return <tt>true</tt> if the receiver contains the specified value.
   */
  def containsValue(value: Double): Boolean =  {
    !forEachPair((iterKey: Int, iterValue: Double) => value != iterValue)
  }

  /**
   * Compares the specified object with this map for equality.  Returns <tt>true</tt> if the given object is also a map
   * and the two maps represent the same mappings.  More formally, two maps <tt>m1</tt> and <tt>m2</tt> represent the
   * same mappings iff
   * <pre>
   * m1.forEachPair(
   *    new IntDoubleProcedure() {
   *      public boolean apply(int key, double value) {
   *        return m2.containsKey(key) && m2.get(key) == value;
   *      }
   *    }
   *  )
   * &&
   * m2.forEachPair(
   *    new IntDoubleProcedure() {
   *      public boolean apply(int key, double value) {
   *        return m1.containsKey(key) && m1.get(key) == value;
   *      }
   *    }
   *  );
   * </pre>
   *
   * This implementation first checks if the specified object is this map; if so it returns <tt>true</tt>.  Then, it
   * checks if the specified object is a map whose size is identical to the size of this set; if not, it it returns
   * <tt>false</tt>.  If so, it applies the iteration as described above.
   *
   * @param obj object to be compared for equality with this map.
   * @return <tt>true</tt> if the specified object is equal to this map.
   */
  override def equals(obj: Any): Boolean = {
    if (!obj.isInstanceOf[AbstractIntDoubleMap]) {
      return false
    }
    val other = obj.asInstanceOf[AbstractIntDoubleMap]

    if (this eq other) {
      return true
    }

    if (other.size != size) {
      return false
    }

    forEachPair(
      (key: Int, value: Double) => {
         other.containsKey(key) && other(key) == value
       }) &&
       other.forEachPair(
         (key: Int, value: Double) => {
           containsKey(key) && this(key) == value
         })
  }

  override def hashCode(): Int = {
    val buf = new Array[Int](size)
    forEachPair(
      (key: Int, value: Double) => {
        var i = 0
        buf(i) = HashUtils.hash(key) ^ HashUtils.hash(value)
        i += 1
        true
      })

    java.util.Arrays.sort(buf)
    IntBuffer.wrap(buf).hashCode()
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
  def forEachKey(procedure: IntProcedure): Boolean

  /**
   * Applies a procedure to each (key,value) pair of the receiver, if any. Iteration order is guaranteed to be
   * <i>identical</i> to the order used by method `#forEachKey(IntProcedure)`.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  def forEachPair(procedure: IntDoubleProcedure): Boolean = {
    forEachKey((iterKey: Int) => {
      procedure(iterKey, this(iterKey))
    })
  }

  /**
   * Returns the value associated with the specified key. It is often a good idea to first check with
   * `#containsKey(int)` whether the given key has a value associated or not, i.e. whether there exists an association
   * for the given key or not.
   *
   * @param key the key to be searched for.
   * @return the value associated with the specified key; <tt>0</tt> if no such key is present.
   */
  def apply(key: Int): Double

  /**
   * Returns a list filled with all keys contained in the receiver. The returned list has a size that equals
   * <tt>this.size()</tt>. Iteration order is guaranteed to be <i>identical</i> to the order used by method
   * `#forEachKey(IntProcedure)`. <p> This method can be used to iterate over the keys of the receiver.
   *
   * @return the keys.
   */
  def getKeys: IntArrayList = {
    val list = new IntArrayList(size)
    getKeys(list)
    list
  }

  /**
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method `forEachKey(IntProcedure)`. <p> This method can be used to
   * iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  def getKeys(list: IntArrayList) {
    list.clear()
    forEachKey(
      (key: Int) => {
        list.add(key)
        true
      })
  }

  /**
   * Fills all keys <i>sorted ascending by their associated value</i> into the specified list. Fills into the list,
   * starting at index 0. After this call returns the specified list has a new size that equals <tt>this.size()</tt>.
   * Primary sort criterium is "value", secondary sort criterium is "key". This means that if any two values are equal,
   * the smaller key comes first. <p> <b>Example:</b> <br> <tt>keys = (8,7,6), values = (1,2,2) --> keyList =
   * (8,6,7)</tt>
   *
   * @param keyList the list to be filled, can have any size.
   */
  def keysSortedByValue(keyList: IntArrayList) {
    pairsSortedByValue(keyList, new DoubleArrayList(size))
  }

  /**
   * Fills all pairs satisfying a given condition into the specified lists. Fills into the lists, starting at index 0.
   * After this call returns the specified lists both have a new size, the number of pairs satisfying the condition.
   * Iteration order is guaranteed to be <i>identical</i> to the order used by method
   * `#forEachKey(IntProcedure)`.
   * <p> <b>Example:</b> <br>
   * <pre>
   * IntIntProcedure condition = new IntIntProcedure() { // match even keys only
   * public boolean apply(int key, int value) { return key%2==0; }
   * }
   * keys = (8,7,6), values = (1,2,2) --> keyList = (6,8), valueList = (2,1)</tt>
   * </pre>
   *
   * @param condition the condition to be matched. Takes the current key as first and the current value as second
   *                  argument.
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  def pairsMatching(condition: IntDoubleProcedure,
      keyList: IntArrayList,
      valueList: DoubleArrayList) {
    keyList.clear()
    valueList.clear()

    forEachPair((key: Int, value: Double) => {
      if (condition(key, value)) {
        keyList.add(key)
        valueList.add(value)
      }
      true
    })
  }

  /**
   * Fills all keys and values <i>sorted ascending by key</i> into the specified lists. Fills into the lists, starting
   * at index 0. After this call returns the specified lists both have a new size that equals <tt>this.size()</tt>. <p>
   * <b>Example:</b> <br> <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (6,7,8), valueList = (2,2,1)</tt>
   *
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  def pairsSortedByKey(keyList: IntArrayList, valueList: DoubleArrayList): Unit = {
    getKeys(keyList)
    keyList.sort()
    valueList.setSize(keyList.size)
    for (i <- 0 until keyList.size) {
      valueList(i) = this(keyList(i))
    }
  }

  /**
   * Fills all keys and values <i>sorted ascending by value</i> into the specified lists. Fills into the lists, starting
   * at index 0. After this call returns the specified lists both have a new size that equals <tt>this.size()</tt>.
   * Primary sort criterium is "value", secondary sort criterium is "key". This means that if any two values are equal,
   * the smaller key comes first. <p> <b>Example:</b> <br> <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (8,6,7),
   * valueList = (1,2,2)</tt>
   *
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  def pairsSortedByValue(keyList: IntArrayList, valueList: DoubleArrayList) {
    getKeys(keyList)
    getValues(valueList)

    val k = keyList.getElements
    val v = valueList.getElements
    val swapper = new Swapper() {
      override def swap(a: Int, b: Int) {
        val t1 = v(a)
        v(a) = v(b)
        v(b) = t1
        val t2 = k(a)
        k(a) = k(b)
        k(b) = t2
      }
    }

    val comp = new NumericComparator[Int]() {
      override def compare(a: Int, b: Int): Int = {
        if (v(a) < v(b)) -1
        else {
          if (v(a) > v(b)) 1
          else {
            if (k(a) < k(b)) -1
            else {
              if (k(a) == k(b)) 0
              else 1
            }
          }
        }
      }
    }

    Sorting.quickSort(0, keyList.size, comp, swapper)
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
  def update(key: Int, value: Double): Unit

  /**
   * Removes the given key with its associated element from the receiver, if present.
   *
   * @param key the key to be removed from the receiver.
   * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
   */
  def removeKey(key: Int): Boolean

  /**
   * Returns a string representation of the receiver, containing the String representation of each key-value pair,
   * sorted ascending by key.
   */
  override def toString: String = {
    val theKeys = getKeys
    //theKeys.sort()

    val buf = new StringBuilder()
    buf.append('[')
    val maxIndex = theKeys.size - 1
    for (i <- 0 to maxIndex) {
      val key = theKeys.get(i)
      buf.append(String.valueOf(key))
      buf.append("->")
      buf.append(String.valueOf(this(key)))
      if (i < maxIndex) {
        buf.append(", ")
      }
    }
    buf.append(']')
    buf.toString()
  }

  /**
   * Returns a string representation of the receiver, containing the String representation of each key-value pair,
   * sorted ascending by value.
   */
  def toStringByValue: String = {
    val theKeys = new IntArrayList()
    keysSortedByValue(theKeys)

    val buf = new StringBuilder()
    buf.append('[')
    val maxIndex = theKeys.size - 1
    for (i <- 0 to maxIndex) {
      val key = theKeys.get(i)
      buf.append(key.toString)
      buf.append("->")
      buf.append(this(key).toString)
      if (i < maxIndex) {
        buf.append(", ")
      }
    }
    buf.append(']')
    buf.toString()
  }

  /**
   * Returns a list filled with all values contained in the receiver. The returned list has a size that equals
   * <tt>this.size()</tt>. Iteration order is guaranteed to be <i>identical</i> to the order used by method
   * `#forEachKey(IntProcedure)`. <p> This method can be used to iterate over the values of the receiver.
   *
   * @return the values.
   */
  def getValues: DoubleArrayList = {
    val list = new DoubleArrayList(size)
    getValues(list)
    list
  }

  /**
   * Fills all values contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method `#forEachKey(IntProcedure)`.
   * <p> This method can be used to
   * iterate over the values of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  def getValues(list: DoubleArrayList) {
    list.clear()
    forEachKey((key: Int) => {
      list.add(this(key))
      true
    })
  }

    /**
   * Assigns the result of a function to each value; <tt>v[i] = function(v[i])</tt>.
   *
   * @param function a function object taking as argument the current association's value.
   */
  def map(function: DoubleFunction) {
    this.forEachPair((key: Int, value: Double) => {
      this(key) = function(value)
      true
    })
  }

  /**
   * Clears the receiver, then adds all (key,value) pairs of <tt>other</tt>values to it.
   *
   * @param other the other map to be copied into the receiver.
   */
  def assign(other: AbstractIntDoubleMap) {
    clear()
    other.forEachPair((key: Int, value: Double) => {
      this(key) = value
      true
    })
  }

  /**
    * Check the map for a key. If present, add an increment to the value. If absent,
    * store a specified value.
    * @param key the key.
    * @param newValue the value to store if the key is not currently in the map.
    * @param incrValue the value to be added to the current value in the map.
   **/
  def adjustOrPutValue(key: Int, newValue: Double, incrValue: Double): Double = {
      val present = containsKey(key)
      if (present) {
        this(key) += incrValue
      } else {
        this(key) = newValue
      }
      newValue
  }
}
