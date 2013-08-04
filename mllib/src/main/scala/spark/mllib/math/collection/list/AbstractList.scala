package spark.mllib.math.collection.list

import spark.mllib.math.PersistentObject

/**
 * Abstract base class for resizable lists holding objects or primitive data types such as
 * `int`, `float`, etc.
 * <p>
 * <b>Note that this implementation is not synchronized.</b>
 *
 * @see     java.util.ArrayList
 * @see      java.util.Vector
 * @see      java.util.Arrays
 */
abstract class AbstractList extends PersistentObject {

  def size: Int

  def isEmpty: Boolean = size == 0

  /** Return a deep copy of the recipient. */
  override def clone(): AbstractList = super.clone().asInstanceOf[AbstractList]

  /**
   * Inserts <tt>length</tt> dummy elements before the specified position into the receiver. Shifts the element
   * currently at that position (if any) and any subsequent elements to the right. <b>This method must set the new size
   * to be <tt>size()+length</tt>.
   *
   * @param index  index before which to insert dummy elements (must be in [0,size])..
   * @param length number of dummy elements to be inserted.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt; size()</tt>.
   */
  protected def beforeInsertDummies(index: Int, length: Int): Unit

  /**
   * Removes all elements from the receiver.  The receiver will be empty after this call returns, but keep its current
   * capacity.
   */
  def clear() {
    removeFromTo(0, size - 1)
  }

  /**
   * Sorts the receiver into ascending order. This sort is guaranteed to be <i>stable</i>:  equal elements will not be
   * reordered as a result of the sort.<p>
   *
   * The sorting algorithm is a modified mergesort (in which the merge is omitted if the highest element in the low
   * sublist is less than the lowest element in the high sublist).  This algorithm offers guaranteed n*log(n)
   * performance, and can approach linear performance on nearly sorted lists.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting algorithm is the right one
   * for your data set.</b> It is generally better to call <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because
   * those methods automatically choose the best sorting algorithm.
   */
  final def mergeSort() {
    mergeSortFromTo(0, size - 1)
  }

  /**
   * Sorts the receiver into ascending order. This sort is guaranteed to be <i>stable</i>:  equal elements will not be
   * reordered as a result of the sort.<p>
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
   * @throws IndexOutOfBoundsException if <tt>(from&lt;0 || from&gt;to || to&gt;=size()) && to!=from-1</tt>.
   */
  def mergeSortFromTo(from: Int, to: Int): Unit

  /**
   * Sorts the receiver into ascending order.  The sorting algorithm is a tuned quicksort, adapted from Jon L. Bentley
   * and M. Douglas McIlroy's "Engineering a Sort Function", Software-Practice and Experience, Vol. 23(11) P. 1249-1265
   * (November 1993).  This algorithm offers n*log(n) performance on many data sets that cause other quicksorts to
   * degrade to quadratic performance.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting algorithm is the right one
   * for your data set.</b> It is generally better to call <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because
   * those methods automatically choose the best sorting algorithm.
   */
  final def quickSort() {
    quickSortFromTo(0, size - 1)
  }

  /**
   * Sorts the specified range of the receiver into ascending order.  The sorting algorithm is a tuned quicksort,
   * adapted from Jon L. Bentley and M. Douglas McIlroy's "Engineering a Sort Function", Software-Practice and
   * Experience, Vol. 23(11) P. 1249-1265 (November 1993).  This algorithm offers n*log(n) performance on many data sets
   * that cause other quicksorts to degrade to quadratic performance.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting algorithm is the right one
   * for your data set.</b> It is generally better to call <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because
   * those methods automatically choose the best sorting algorithm.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @throws IndexOutOfBoundsException if <tt>(from&lt;0 || from&gt;to || to&gt;=size()) && to!=from-1</tt>.
   */
  def quickSortFromTo(from: Int, to: Int): Unit

  /**
   * Removes the element at the specified position from the receiver. Shifts any subsequent elements to the left.
   *
   * @param index the index of the element to removed.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt;= size()</tt>.
   */
  def remove(index: Int) {
    removeFromTo(index, index)
  }

  /**
   * Removes from the receiver all elements whose index is between <code>from</code>, inclusive and <code>to</code>,
   * inclusive.  Shifts any succeeding elements to the left (reduces their index). This call shortens the list by
   * <tt>(to - from + 1)</tt> elements.
   *
   * @param fromIndex index of first element to be removed.
   * @param toIndex   index of last element to be removed.
   * @throws IndexOutOfBoundsException if <tt>(from&lt;0 || from&gt;to || to&gt;=size()) && to!=from-1</tt>.
   */
  def removeFromTo(fromIndex: Int, toIndex: Int): Unit

  /** Reverses the elements of the receiver. Last becomes first, second last becomes second first, and so on. */
  def reverse(): Unit

  /**
   * Sets the size of the receiver. If the new size is greater than the current size, new null or zero items are added
   * to the end of the receiver. If the new size is less than the current size, all components at index newSize and
   * greater are discarded. This method does not release any superfluos internal memory. Use method <tt>trimToSize</tt>
   * to release superfluos internal memory.
   *
   * @param newSize the new size of the receiver.
   * @throws IndexOutOfBoundsException if <tt>newSize &lt; 0</tt>.
   */
  def setSize(newSize: Int) {
    if (newSize < 0) {
      throw new IndexOutOfBoundsException("newSize:" + newSize)
    }

    val currentSize = size
    if (newSize != currentSize) {
      if (newSize > currentSize) {
        beforeInsertDummies(currentSize, newSize - currentSize)
      } else if (newSize < currentSize) {
        removeFromTo(newSize, currentSize - 1)
      }
    }
  }

  /**
   * Sorts the receiver into ascending order.
   *
   * The sorting algorithm is dynamically chosen according to the characteristics of the data set.
   *
   * This implementation simply calls <tt>sortFromTo(...)</tt>. Override <tt>sortFromTo(...)</tt> if you can determine
   * which sort is most appropriate for the given data set.
   */
  final def sort() {
    sortFromTo(0, size - 1)
  }

  /**
   * Sorts the specified range of the receiver into ascending order.
   *
   * The sorting algorithm is dynamically chosen according to the characteristics of the data set. This default
   * implementation simply calls quickSort. Override this method if you can determine which sort is most appropriate for
   * the given data set.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @throws IndexOutOfBoundsException if <tt>(from&lt;0 || from&gt;to || to&gt;=size()) && to!=from-1</tt>.
   */
  def sortFromTo(from: Int, to: Int) {
    quickSortFromTo(from, to)
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. Releases any superfluos internal memory. An
   * application can use this operation to minimize the storage of the receiver. <p> This default implementation does
   * nothing. Override this method in space efficient implementations.
   */
  def trimToSize() {
  }
}

object AbstractList {
  /** Checks if the given index is in range. */
  private[list] def checkRange(index: Int, theSize: Int) {
    if (index >= theSize || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + theSize)
    }
  }

  /**
   * Checks if the given range is within the contained array's bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>to!=from-1 || from&lt;0 || from&gt;to || to&gt;=size()</tt>.
   */
  private[list] def checkRangeFromTo(from: Int, to: Int, theSize: Int) {
    if (to == from - 1) return

    if (from < 0 || from > to || to >= theSize) {
      throw new IndexOutOfBoundsException("from: " + from + ", to: " + to + ", size=" + theSize)
    }
  }
}