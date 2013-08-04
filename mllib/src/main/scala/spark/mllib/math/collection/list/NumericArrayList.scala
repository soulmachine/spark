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

package spark.mllib.math.collection.list

import spark.mllib.math.numeric.Numeric

import spark.mllib.math.Sorting
import spark.mllib.math.function.NumericProcedure
import spark.mllib.math.function.NumericComparator

import scala.collection.mutable.ListBuffer

/** Resizable list holding primitive numeric type(`Byte`, `Char`, `Short`, `Int`, `Long`, `Float`,
  * `Double`) elements.
  *
  * implemented with arrays.
  *
  * @tparam T the type of elements in the list
  */
class NumericArrayList[T: Numeric: ClassManifest] (
    protected var elements: Array[T]) extends AbstractList {

  /**
   * The size of the list.
   */
  protected var _size: Int = elements.length

  /** Constructs an empty list. */
  def this() = this(new Array[T](10))

  /**
   * Constructs an empty list with the specified initial capacity.
   *
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding
   *                        itself by allocating new internal memory.
   */
  def this(initialCapacity: Int) = this(new Array[T](initialCapacity))

  /**
   * Appends the specified element to the end of this list.
   *
   * @param element element to be appended to this list.
   */
  def add(element: T) {
    // overridden for performance only.
    if (size == elements.length) {
      ensureCapacity(size + 1)
    }
    elements(size) = element
    _size += 1
  }


  /**
   * Appends all elements of the specified list to the end of the receiver.
   *
   * @param other the list to be appended.
   */
  def addAllOf(other: NumericArrayList[T]) {
    addAllOfFromTo(other, 0, other.size - 1)
  }

  /**
   * Appends the part of the specified list between `from` (inclusive) and `to`
   * (inclusive) to the receiver.
   *
   * @param other the list to be added to the receiver.
   * @param from  the index of the first element to be appended (inclusive).
   * @param to    the index of the last element to be appended (inclusive).
   * @throws IndexOutOfBoundsException index is out of range (<tt>other.size() < 0 &&
   *                                   (from &lt; 0 || from &gt; to || to &gt;= other.size())</tt>).
   */
  def addAllOfFromTo(other: NumericArrayList[T], from: Int, to: Int) {
    beforeInsertAllOfFromTo(size, other, from, to)
  }

  /**
   * Inserts the specified element before the specified position into the receiver. Shifts the
   * element currently at that position (if any) and any subsequent elements to the right.
   *
   * @param index   index before which the specified element is to be inserted (must be in [0,size]).
   * @param element element to be inserted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt; size()</tt>).
   */
  def beforeInsert(index: Int, element: T) {
    if (size == index) {
      add(element)
      return
    }
    if (index > size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size)
    }
    ensureCapacity(size + 1)
    Array.copy(elements, index, elements, index + 1, size - index)
    elements(index) = element
    _size += 1
  }

  /**
   * Inserts the part of the specified list between <code>otherFrom</code> (inclusive) and
   * <code>otherTo</code> (inclusive) before the specified position into the receiver. Shifts the
   * element currently at that position (if any) and any subsequent elements to the right.
   *
   * @param index index before which to insert first element from the specified list (must be in
   *              [0,size])..
   * @param other list of which a part is to be inserted into the receiver.
   * @param from  the index of the first element to be inserted (inclusive).
   * @param to    the index of the last element to be inserted (inclusive).
   * @throws IndexOutOfBoundsException index is out of range (<tt>other.size()&gt;0 &amp;&amp;
   *         (from&lt;0 || from&gt;to || to&gt;=other.size())</tt>).
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt;
   *         size()</tt>).
   */
  def beforeInsertAllOfFromTo(index: Int, other: NumericArrayList[T], from: Int, to: Int) {
    val length = to - from + 1
    this.beforeInsertDummies(index, length)
    this.replaceFromToWithFrom(index, index + length - 1, other, from)
  }

  /**
   * Inserts <tt>length</tt> dummy elements before the specified position into the receiver. Shifts the element
   * currently at that position (if any) and any subsequent elements to the right. <b>This method must set the new size
   * to be <tt>size()+length</tt>.
   *
   * @param index  index before which to insert dummy elements (must be in [0,size])..
   * @param length number of dummy elements to be inserted.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt; size()</tt>.
   */
  override protected def beforeInsertDummies(index: Int, length: Int) {
    if (index > size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size)
    }
    if (length > 0) {
      ensureCapacity(size + length)
      _size = size + length
      replaceFromToWithFrom(index + length, size - 1, this, index)
    }
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm.
   *
   * The receiver must <strong>must</strong> be sorted (as by the sort method) prior to making this
   * call.  If it is not sorted, the results are undefined: in particular, the call may enter an
   * infinite loop.  If the receiver contains multiple elements equal to the specified object, there
   * is no guarantee which instance will be found.
   *
   * @param key the value to be searched for.
   * @return index of the search key, if it is contained in the receiver; otherwise,
   *         <tt>(-(<i>insertion point</i>) - 1)</tt>.  The <i>insertion point</i> is defined as the
   *         the point at which the value would be inserted into the receiver: the index of the first
   *         element greater than the key, or <tt>receiver.size()</tt>, if all elements in the
   *         receiver are less than the specified key.  Note that this guarantees that the return
   *         value will be &gt;= 0 if and only if the key is found.
   * @see java.util.Arrays
   */
  def binarySearch(key: T): Int = {
    this.binarySearchFromTo(key, 0, size - 1)
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm.
   *
   * The receiver must <strong>must</strong> be sorted (as by the sort method) prior to making this
   * call.  If it is not sorted, the results are undefined: in particular, the call may enter an
   * infinite loop.  If the receiver contains multiple elements equal to the specified object, there
   * is no guarantee which instance will be found.
   *
   * @param key  the value to be searched for.
   * @param from the leftmost search position, inclusive.
   * @param to   the rightmost search position, inclusive.
   * @return index of the search key, if it is contained in the receiver; otherwise,
   *         <tt>(-(<i>insertion point</i>) - 1)</tt>.  The <i>insertion point</i> is defined as the
   *         the point at which the value would be inserted into the receiver: the index of the first
   *         element greater than the key, or <tt>receiver.size()</tt>, if all elements in the
   *         receiver are less than the specified key.  Note that this guarantees that the return
   *         value will be &gt;= 0 if and only if the key is found.
   * @see java.util.Arrays
   */
  def binarySearchFromTo(key: T, from: Int, to: Int): Int = {
    spark.mllib.math.BinarySearch.binarySearchFromTo(elements, key, from, to)
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  override def clone(): NumericArrayList[T] = new NumericArrayList[T](elements.clone())

  /**
   * Returns true if the receiver contains the specified element.
   *
   * @param elem element whose presence in the receiver is to be tested.
   */
  def contains(elem: T): Boolean = indexOfFromTo(elem, 0, size - 1) >= 0

  /**
   * Deletes the first element from the receiver that is identical to the specified element. Does nothing, if no such
   * matching element is contained.
   *
   * @param element the element to be deleted.
   */
  def delete(element: T) {
    val index = indexOfFromTo(element, 0, size - 1)
    if (index >= 0) {
      remove(index)
    }
  }

  /**
   * Returns the elements currently stored, possibly including invalid elements between size and capacity.
   *
   * <b>WARNING:</b> For efficiency reasons and to keep memory usage low, this method may decide <b>not to copy the
   * array</b>. So if subsequently you modify the returned array directly via the [] operator, be sure you know what
   * you're doing.
   *
   * @return the elements currently stored.
   */
  def getElements: Array[T] = elements

  /**
   * Sets the receiver's elements to be the specified array. The size and capacity of the list is the length of the
   * array. <b>WARNING:</b> For efficiency reasons and to keep memory usage low, this method may decide <b>not to copy
   * the array</b>. So if subsequently you modify the returned array directly via the [] operator, be sure you know what
   * you're doing.
   *
   * @param elements the new elements to be stored.
   * @return the receiver itself.
   */
  def setElements(elements: Array[T]): NumericArrayList[T] = {
    this.elements = elements
    this._size = elements.length
    this
  }

  /**
   * Ensures that the receiver can hold at least the specified number of elements without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver.
   *
   * @param minCapacity the desired minimum capacity.
   */
  def ensureCapacity(minCapacity: Int): Unit = {
    elements = spark.mllib.math.Arrays.ensureCapacity(elements, minCapacity)
  }

  /**
   * Compares the specified Object with the receiver. Returns true if and only if the specified Object is also an
   * ArrayList of the same type, both Lists have the same size, and all corresponding pairs of elements in the two Lists
   * are identical. In other words, two Lists are defined to be equal if they contain the same elements in the same
   * order.
   *
   * @param otherObj the Object to be compared for equality with the receiver.
   * @return true if the specified Object is equal to the receiver.
   */
  override def equals(otherObj: Any): Boolean = {
    if (otherObj == null) {
      return false
    }

    if (!otherObj.isInstanceOf[NumericArrayList[_]]) {
      return false
    }

    val other = otherObj.asInstanceOf[NumericArrayList[_]]

    if (this eq other) {
      return true
    }

    if (size != other.size) {
      return false
    }

    val otherElements = other.getElements
    for (i <- (0 to size).reverse) {
      if (elements(i) != otherElements(i)) {
        return false
      }
    }

    true
  }

  /**
   * Sets the specified range of elements in the specified array to the specified value.
   *
   * @param from   the index of the first element (inclusive) to be filled with the specified value.
   * @param to     the index of the last element (inclusive) to be filled with the specified value.
   * @param value  the value to be stored in the specified elements of the receiver.
   */
  def fillFromToWith(from: Int, to: Int, value: T) {
    AbstractList.checkRangeFromTo(from, to, this.size)
    for (i <- from to to) {
      this(i) = value
    }
  }

  /**
   * Applies a procedure to each element of the receiver, if any. Starts at index 0, moving rightwards.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   */
  def forEach(procedure: NumericProcedure[T]): Boolean = {
    // overridden for performance only.
    for (i <- 0 until _size) {
      if (!procedure(elements(i))) {
        return false
      }
    }
    true
  }

  /**
   * Returns the element at the specified position in the receiver.
   *
   * @param index index of element to return.
   * @throws IndexOutOfBoundsException index is out of range (index &lt; 0 || index &gt;= size()).
   */
  def get(index: Int): T = {
    // overridden for performance only.
    if (index >= size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size)
    }

    elements(index)
  }

  /**
   * Returns the element at the specified position in the receiver; <b>WARNING:</b> Does not check preconditions.
   * Provided with invalid parameters this method may return invalid elements without throwing any exception! <b>You
   * should only use this method when you are absolutely sure that the index is within bounds.</b> Precondition
   * (unchecked): <tt>index &gt;= 0 && index &lt; size()</tt>.
   *
   * This method is normally only used internally in large loops where bounds are explicitly checked before the loop and
   * need no be rechecked within the loop. However, when desperately, you can give this method <tt>public</tt>
   * visibility in subclasses.
   *
   * @param index index of element to return.
   */
  def apply(index: Int): T = elements(index)

  /**
   * Returns the index of the first occurrence of the specified element. Returns <code>-1</code> if the receiver does
   * not contain this element.
   *
   * @param element the element to be searched for.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   */
  def indexOf(element: T): Int = indexOfFromTo(element, 0, size - 1)

  /**
   * Returns the index of the first occurrence of the specified element. Returns <code>-1</code> if the receiver does
   * not contain this element. Searches between <code>from</code>, inclusive and <code>to</code>, inclusive. Tests for
   * identity.
   *
   * @param element element to search for.
   * @param from    the leftmost search position, inclusive.
   * @param to      the rightmost search position, inclusive.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  def indexOfFromTo(element: T, from: Int, to: Int): Int = {
    // overridden for performance only.
    if (_size == 0) {
      return -1
    }
    AbstractList.checkRangeFromTo(from, to, _size)

    for (i <- from to to) {
      if (element == elements(i)) {
        return i
      } //found
    }

    -1 //not found
  }

  /**
   * Returns the index of the last occurrence of the specified element. Returns <code>-1</code> if the receiver does not
   * contain this element.
   *
   * @param element the element to be searched for.
   * @return the index of the last occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   */
  def lastIndexOf(element: T): Int = lastIndexOfFromTo(element, 0, size - 1)

  /**
   * Returns the index of the last occurrence of the specified element. Returns <code>-1</code> if the receiver does not
   * contain this element. Searches beginning at <code>to</code>, inclusive until <code>from</code>, inclusive. Tests
   * for identity.
   *
   * @param element element to search for.
   * @param from    the leftmost search position, inclusive.
   * @param to      the rightmost search position, inclusive.
   * @return the index of the last occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  def lastIndexOfFromTo(element: T, from: Int, to: Int): Int = {
    // overridden for performance only.
    if (_size == 0) {
      return -1
    }
    AbstractList.checkRangeFromTo(from, to, size)

    for (i <- (from to to).reverse) {
      if (element == elements(i)) {
        return i
      } //found
    }

    -1 //not found
  }

  /**
   * Sorts the specified range of the receiver into ascending order.
   *
   * The sorting algorithm is a modified mergesort (in which the merge is omitted if the highest element in the low
   * sublist is less than the lowest element in the high sublist).  This algorithm offers guaranteed n*log(n)
   * performance, and can approach linear performance on nearly sorted lists.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting algorithm is the right one
   * for your data set.</b> It is generally better to call <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because
   * those methods automatically choose the best sorting algorithm.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  override def mergeSortFromTo(from: Int, to: Int) {
    AbstractList.checkRangeFromTo(from, to, _size)
    Sorting.mergeSort(elements, from, to + 1)
  }

  /**
   * Sorts the receiver according to the order induced by the specified comparator.  All elements in the range must be
   * <i>mutually comparable</i> by the specified comparator (that is, <tt>c.compare(e1, e2)</tt> must not throw a
   * <tt>ClassCastException</tt> for any elements <tt>e1</tt> and <tt>e2</tt> in the range).<p>
   *
   * This sort is guaranteed to be <i>stable</i>:  equal elements will not be reordered as a result of the sort.<p>
   *
   * The sorting algorithm is a modified mergesort (in which the merge is omitted if the highest element in the low
   * sublist is less than the lowest element in the high sublist).  This algorithm offers guaranteed n*log(n)
   * performance, and can approach linear performance on nearly sorted lists.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @param c    the comparator to determine the order of the receiver.
   * @throws ClassCastException             if the array contains elements that are not <i>mutually comparable</i> using
   *                                        the specified comparator.
   * @throws IllegalArgumentException       if <tt>fromIndex &gt; toIndex</tt>
   * @throws ArrayIndexOutOfBoundsException if <tt>fromIndex &lt; 0</tt> or <tt>toIndex &gt; a.length</tt>
   * @throws IndexOutOfBoundsException      index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                        to&gt;=size())</tt>).
   */
  def mergeSortFromTo(from: Int, to: Int, c: NumericComparator[T]) {
    AbstractList.checkRangeFromTo(from, to, _size)
    Sorting.mergeSort(elements, from, to + 1, c)
  }

  /**
   * Returns a new list of the part of the receiver between <code>from</code>, inclusive, and <code>to</code>,
   * inclusive.
   *
   * @param from the index of the first element (inclusive).
   * @param to   the index of the last element (inclusive).
   * @return a new list
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  def partFromTo(from: Int, to: Int): NumericArrayList[T] = {
    if (_size == 0) {
      return new NumericArrayList(0)
    }

    AbstractList.checkRangeFromTo(from, to, size)

    val part = new Array[T](to - from + 1)
    Array.copy(elements, from, part, 0, to - from + 1)
    new NumericArrayList(part)
  }

  /**
   * Sorts the specified range of the receiver into ascending numerical order.  The sorting
   * algorithm is a tuned quicksort, adapted from Jon L. Bentley and M. Douglas McIlroy's
   * "Engineering a Sort Function", Software-Practice and Experience, Vol. 23(11) P. 1249-1265
   * (November 1993).  This algorithm offers n*log(n) performance on many data sets that cause other
   * quicksorts to degrade to quadratic performance.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting
   * algorithm is the right one for your data set.</b> It is generally better to call
   * <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because those methods automatically choose
   * the best sorting algorithm.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 ||
   *                                   from&gt;to || to&gt;=size())</tt>).
   */
  def quickSortFromTo(from: Int, to: Int) {
    AbstractList.checkRangeFromTo(from, to, _size)
    Sorting.quickSort(elements, from, to + 1)
  }

  /**
   * Sorts the specified range of the receiver into ascending numerical order.  The sorting
   * algorithm is a tuned quicksort, adapted from Jon L. Bentley and M. Douglas McIlroy's
   * "Engineering a Sort Function", Software-Practice and Experience, Vol. 23(11) P. 1249-1265
   * (November 1993).  This algorithm offers n*log(n) performance on many data sets that cause other
   * quicksorts to degrade to quadratic performance.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting
   * algorithm is the right one for your data set.</b> It is generally better to call
   * <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because those methods automatically choose
   * the best sorting algorithm.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @param c    the comparator to determine the order of the receiver.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 ||
   *                                   from&gt;to || to&gt;=size())</tt>).
   */
  def quickSortFromTo(from: Int, to: Int, c: NumericComparator[T]) {
    AbstractList.checkRangeFromTo(from, to, _size)
    Sorting.quickSort(elements, from, to + 1, c)
  }

  /**
   * Removes from the receiver all elements that are contained in the specified list. Tests for identity.
   *
   * @param other the other list.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  def removeAll(other: NumericArrayList[T]): Boolean = {
    if (!other.isInstanceOf[NumericArrayList[_]]) {
      return false
    }

    /* There are two possibilities to do the thing
       a) use other.indexOf(...)
       b) sort other, then use other.binarySearch(...)

       Let's try to figure out which one is faster. Let M=size, N=other.size, then
       a) takes O(M*N) steps
       b) takes O(N*logN + M*logN) steps (sorting is O(N*logN) and binarySearch is O(logN))

       Hence, if N*logN + M*logN < M*N, we use b) otherwise we use a).
    */
    if (other.isEmpty) {
      return false
    } //nothing to do
    val limit = other.size - 1
    var j = 0

    val N = other.size
    val M = _size

    if ((N + M) * spark.mllib.math.Arithmetic.log2(N) < M * N) {
      // it is faster to sort other before searching in it
      val sortedList = other.clone()
      sortedList.quickSort()

      for (i <- 0 until _size) {
        if (sortedList.binarySearchFromTo(elements(i), 0, limit) < 0) {
          elements(j) = elements(i)
          j += 1
        }
      }
    } else {
      // it is faster to search in other without sorting
      for (i <- 0 until _size) {
        if (other.indexOfFromTo(elements(i), 0, limit) < 0) {
          elements(j) = elements(i)
          j += 1
        }
      }
    }

    val modified = j != _size
    setSize(j)
    modified
  }

  /**
   * Removes from the receiver all elements whose index is between <code>from</code>, inclusive and <code>to</code>,
   * inclusive.  Shifts any succeeding elements to the left (reduces their index). This call shortens the list by
   * <tt>(to - from + 1)</tt> elements.
   *
   * @param from index of first element to be removed.
   * @param to   index of last element to be removed.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  override def removeFromTo(from: Int, to: Int) {
    AbstractList.checkRangeFromTo(from, to, _size)
    val numMoved = _size - to - 1
    if (numMoved > 0) {
      replaceFromToWithFrom(from, from - 1 + numMoved, this, to + 1)
      //fillFromToWith(from+numMoved, size-1, 0.0f); //delta
    }
    val width = to - from + 1
    if (width > 0) {
      _size = _size - width
    }
  }

  /**
   * Replaces a number of elements in the receiver with the same number of elements of another list. Replaces elements
   * in the receiver, between <code>from</code> (inclusive) and <code>to</code> (inclusive), with elements of
   * <code>other</code>, starting from <code>otherFrom</code> (inclusive).
   *
   * @param from      the position of the first element to be replaced in the receiver
   * @param to        the position of the last element to be replaced in the receiver
   * @param other     list holding elements to be copied into the receiver.
   * @param otherFrom position of first element within other list to be copied.
   */
  def replaceFromToWithFrom(from: Int, to: Int, other: NumericArrayList[T], otherFrom: Int) {
    val length = to - from + 1
    if (length > 0) {
      AbstractList.checkRangeFromTo(from, to, _size)
      AbstractList.checkRangeFromTo(otherFrom, otherFrom + length - 1, other.size)
      Array.copy(other.elements, otherFrom, elements, from, length)
    }
  }

  /**
   * Replaces the part between <code>from</code> (inclusive) and <code>to</code> (inclusive) with the other list's part
   * between <code>otherFrom</code> and <code>otherTo</code>. Powerful (and tricky) method! Both parts need not be of
   * the same size (part A can both be smaller or larger than part B). Parts may overlap. Receiver and other list may
   * (but most not) be identical. If <code>from &gt; to</code>, then inserts other part before <code>from</code>.
   *
   * @param from      the first element of the receiver (inclusive)
   * @param to        the last element of the receiver (inclusive)
   * @param other     the other list (may be identical with receiver)
   * @param otherFrom the first element of the other list (inclusive)
   * @param otherTo   the last element of the other list (inclusive)
   *
   *                  <p><b>Examples:</b><pre>
   *                  a=[0, 1, 2, 3, 4, 5, 6, 7]
   *                  b=[50, 60, 70, 80, 90]
   *                  a.R(...)=a.replaceFromToWithFromTo(...)
   *
   *                  a.R(3,5,b,0,4)-->[0, 1, 2, 50, 60, 70, 80, 90,
   *                  6, 7]
   *                  a.R(1,6,b,0,4)-->[0, 50, 60, 70, 80, 90, 7]
   *                  a.R(0,6,b,0,4)-->[50, 60, 70, 80, 90, 7]
   *                  a.R(3,5,b,1,2)-->[0, 1, 2, 60, 70, 6, 7]
   *                  a.R(1,6,b,1,2)-->[0, 60, 70, 7]
   *                  a.R(0,6,b,1,2)-->[60, 70, 7]
   *                  a.R(5,3,b,0,4)-->[0, 1, 2, 3, 4, 50, 60, 70,
   *                  80, 90, 5, 6, 7]
   *                  a.R(5,0,b,0,4)-->[0, 1, 2, 3, 4, 50, 60, 70,
   *                  80, 90, 5, 6, 7]
   *                  a.R(5,3,b,1,2)-->[0, 1, 2, 3, 4, 60, 70, 5, 6,
   *                  7]
   *                  a.R(5,0,b,1,2)-->[0, 1, 2, 3, 4, 60, 70, 5, 6,
   *                  7]
   *
   *                  Extreme cases:
   *                  a.R(5,3,b,0,0)-->[0, 1, 2, 3, 4, 50, 5, 6, 7]
   *                  a.R(5,3,b,4,4)-->[0, 1, 2, 3, 4, 90, 5, 6, 7]
   *                  a.R(3,5,a,0,1)-->[0, 1, 2, 0, 1, 6, 7]
   *                  a.R(3,5,a,3,5)-->[0, 1, 2, 3, 4, 5, 6, 7]
   *                  a.R(3,5,a,4,4)-->[0, 1, 2, 4, 6, 7]
   *                  a.R(5,3,a,0,4)-->[0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
   *                  5, 6, 7]
   *                  a.R(0,-1,b,0,4)-->[50, 60, 70, 80, 90, 0, 1, 2,
   *                  3, 4, 5, 6, 7]
   *                  a.R(0,-1,a,0,4)-->[0, 1, 2, 3, 4, 0, 1, 2, 3,
   *                  4, 5, 6, 7]
   *                  a.R(8,0,a,0,4)-->[0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
   *                  2, 3, 4]
   *                  </pre>
   */
  def replaceFromToWithFromTo(from: Int, to: Int, other: NumericArrayList[T], otherFrom: Int, otherTo: Int) {
    if (otherFrom > otherTo) {
      throw new IndexOutOfBoundsException("otherFrom: " + otherFrom + ", otherTo: " + otherTo)
    }

    if (this == other && to - from != otherTo - otherFrom) {
      // avoid stumbling over my own feet
      replaceFromToWithFromTo(from, to, partFromTo(otherFrom, otherTo), 0, otherTo - otherFrom)
      return
    }

    val length = otherTo - otherFrom + 1
    var diff = length
    var theLast = from - 1

    if (to >= from) {
      diff -= (to - from + 1)
      theLast = to
    }

    if (diff > 0) {
      beforeInsertDummies(theLast + 1, diff)
    } else {
      if (diff < 0) {
        removeFromTo(theLast + diff, theLast - 1)
      }
    }

    if (length > 0) {
      replaceFromToWithFrom(from, from + length - 1, other, otherFrom)
    }
  }

  /**
   * Retains (keeps) only the elements in the receiver that are contained in the specified other list. In other words,
   * removes from the receiver all of its elements that are not contained in the specified other list.
   *
   * @param other the other list to test against.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  def retainAll(other: NumericArrayList[T]): Boolean = {
    /* There are two possibilities to do the thing
       a) use other.indexOf(...)
       b) sort other, then use other.binarySearch(...)

       Let's try to figure out which one is faster. Let M=size, N=other.size, then
       a) takes O(M*N) steps
       b) takes O(N*logN + M*logN) steps (sorting is O(N*logN) and binarySearch is O(logN))

       Hence, if N*logN + M*logN < M*N, we use b) otherwise we use a).
    */
    val limit = other.size - 1
    var j = 0

    val N = other.size
    val M = _size
    if ((N + M) * spark.mllib.math.Arithmetic.log2(N) < M * N) {
      // it is faster to sort other before searching in it
      val sortedList = other.clone()
      sortedList.quickSort()

      for (i <- 0 until _size) {
        if (sortedList.binarySearchFromTo(elements(i), 0, limit) >= 0) {
          elements(j) = elements(i)
          j += 1
        }
      }
    } else {
      // it is faster to search in other without sorting
      for (i <- 0 until _size) {
        if (other.indexOfFromTo(elements(i), 0, limit) >= 0) {
          elements(j) = elements(i)
          j += 1
        }
      }
    }

    val modified = j != _size
    setSize(j)
    modified
  }

  /** Reverses the elements of the receiver. Last becomes first, second last becomes second first, and so on. */
  override def reverse() {
    val limit = _size / 2
    var j = _size - 1

    for (i <- 0 until limit) {
      //swap
      val tmp = elements(i)
      elements(i) = elements(j)
      elements(j) = tmp
      j -= 1
    }
  }

  /**
   * Replaces the element at the specified position in the receiver with the specified element.
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt;= size()</tt>.
   */
  def set(index: Int, element: T) {
    if (index >= size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size)
    }
    this(index) = element
  }

  /**
   * Replaces the element at the specified position in the receiver with the specified element; <b>WARNING:</b> Does not
   * check preconditions. Provided with invalid parameters this method may access invalid indexes without throwing any
   * exception! <b>You should only use this method when you are absolutely sure that the index is within bounds.</b>
   * Precondition (unchecked): <tt>index &gt;= 0 && index &lt; size()</tt>.
   *
   * This method is normally only used internally in large loops where bounds are explicitly checked before the loop and
   * need no be rechecked within the loop. However, when desperately, you can give this method <tt>public</tt>
   * visibility in subclasses.
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   */
  def update(index: Int, element: T) {
    elements(index) = element
  }

  /** Returns the number of elements contained in the receiver. */
  override def size: Int = _size

  /**
   * Returns a list which is a concatenation of <code>times</code> times the receiver.
   *
   * @param times the number of times the receiver shall be copied.
   */
  def times(times: Int): NumericArrayList[T] = {
    val newList = new NumericArrayList[T](times * _size)
    for (i <- 0 until times) {
      newList.addAllOfFromTo(this, 0, size - 1)
    }
    newList
  }

  /** Returns a <code>ArrayList</code> containing all the elements in the receiver. */
  def toList: List[T] = {
    val mySize = _size
    val buf = new ListBuffer[T]
    for (i <- 0 until mySize) {
      buf += elements(i)
    }
    buf.toList
  }

  def toArray: Array[T] = {
    val mySize = _size
    val newArray = new Array[T](mySize)

    Array.copy(elements, 0, newArray, 0, mySize)
    newArray
  }

  /** Returns a string representation of the receiver, containing the String representation of each element. */
  override def toString: String = {
    spark.mllib.math.Arrays.toString(elements)
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. Releases any superfluous internal memory. An
   * application can use this operation to minimize the storage of the receiver.
   */
  override def trimToSize() {
    elements = spark.mllib.math.Arrays.trimToCapacity(elements, _size)
  }
}
