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

package spark.mllib.math

import spark.mllib.math.numeric._
import spark.mllib.math.numeric.Numeric
import spark.mllib.math.numeric.FastImplicits._
import spark.mllib.math.function.NumericComparator

final object Sorting {
/* Specifies when to switch to insertion sort */
  private val SIMPLE_LENGTH = 7
  
  private def med3[T: Numeric: ClassManifest](array: Array[T], a: Int, b: Int, c: Int): Int = {
    val x = array(a)
    val y = array(b)
    val z = array(c)
    if (x < y) {
      if (y < z) b else { if (x < z) c else a }
    } else {
      if (y > z) b else { if (x > z) c else a }
    }
  }
  
  private def med3[T: Numeric: ClassManifest](array: Array[T], a: Int, b: Int, c: Int, 
      comp: NumericComparator[T]): Int = {
    val x = array(a)
    val y = array(b)
    val z = array(c)
    val comparisonxy = comp.compare(x, y)
    val comparisonxz = comp.compare(x, z)
    val comparisonyz = comp.compare(y, z)
    if (comparisonxy < 0) {
      if (comparisonyz < 0) b else { if (comparisonxz < 0) c else a}
    } else {
      if (comparisonyz > 0) b else { if (comparisonxz > 0) c else a }
    }
  }
  
  /**
   * This is used for 'external' sorting. The comparator takes <em>indices</em>,
   * not values, and compares the external values found at those indices.
   * @param a
   * @param b
   * @param c
   * @param comp
   * @return
   */
  private def med3(a: Int, b: Int, c: Int, comp: NumericComparator[Int]): Int = {
    val comparisonab = comp.compare(a, b)
    val comparisonac = comp.compare(a, c)
    val comparisonbc = comp.compare(b, c)
    if (comparisonab < 0) {
      if (comparisonbc < 0) b else { if(comparisonac < 0) c else a }
    } else {
      if (comparisonbc > 0) b else {if (comparisonac > 0) c else a }
    }
  }
  
  private def checkBounds(arrLength: Int, start: Int, end: Int) {
    if (start > end) {
      // K0033=Start index ({0}) is greater than end index ({1})
      throw new IllegalArgumentException("Start index " + start
          + " is greater than end index " + end)
    }
    if (start < 0) {
      throw new ArrayIndexOutOfBoundsException("Array index out of range "
          + start)
    }
    if (end > arrLength) {
      throw new ArrayIndexOutOfBoundsException("Array index out of range "
          + end)
    }
  }

  /**
   * In-place insertion sort that is fast for pre-sorted data.
   *
   * @param start Where to start sorting (inclusive)
   * @param end   Where to stop (exclusive)
   * @param comp  Sort order.
   * @param swap  How to swap items.
   */
  private def insertionSort[T: Numeric: ClassManifest](array: Array[T], start: Int, end: Int, 
      comp: NumericComparator[T]) {
    var temp: T = array(0)
    for (i <- start+1 until end) {
      for (j <- (start+1 to i).reverse if comp.compare(array(j - 1), array(j)) > 0) {
        temp = array(j - 1)
        array(j - 1) = array(j)
        array(j) = temp
      }
    }
  }
  
  /**
   * Sorts the specified range in the array in a specified order, asending
   * 
   * @param array
   *          the {@code double} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   * @see Double#compareTo(Double)
   */
  def quickSort[T: Numeric: ClassManifest](array: Array[T], start: Int, end: Int) {
    quickSort(array, start, end, new NumericComparator[T]() {
      override def compare(o1: T, o2: T): Int = {
        if (o1 > o2) 1 else if (o1 < o2) -1 else 0
      }
    })
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code double} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparison.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   * @see Double#compareTo(Double)
   */
  def quickSort[T: Numeric: ClassManifest](array: Array[T], start: Int, end: Int, 
      comp: NumericComparator[T]) {
    require (array != null)
    checkBounds(array.length, start, end)
    quickSort0(start, end, array, comp)
  }
  
  private def quickSort0[T: Numeric: ClassManifest](start: Int, end: Int, array: Array[T], 
      comp: NumericComparator[T]) {
    var temp: T = array(0)
    var length = end - start
    if (length < 7) {
      for (i <- (start+1) until end) {
        for (j <- (start+1) to i if comp.compare(array(j), array(j - 1)) < 0) {
          temp = array(j)
          array(j) = array(j - 1)
          array(j - 1) = temp
        }
      }
      return
    }
    var middle = (start + end) / 2
    if (length > 7) {
      var bottom = start
      var top = end - 1
      if (length > 40) {
        length /= 8
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length), comp)
        middle = med3(array, middle - length, middle, middle + length, comp)
        top = med3(array, top - (2 * length), top - length, top, comp)
      }
      middle = med3(array, bottom, middle, top, comp)
    }
    val partionValue = array(middle)
    var a = start
    var b = a
    var c = end - 1
    var d = c
    while (true) {
      var comparison = comp.compare(partionValue, array(b))
      while (b <= c && comparison >= 0) {
        if (comparison == 0) {
          temp = array(a)
          array(a) = array(b)
          a += 1
          array(b) = temp
        }
        b += 1
        comparison = comp.compare(partionValue, array(b))
      }
      comparison = comp.compare(array(c), partionValue)
      while (c >= b && comparison >= 0) {
        if (comparison == 0) {
          temp = array(c)
          array(c) = array(d)
          array(d) = temp
          d -= 1
        }
        c -= 1
      }
      if (b <= c) {
        temp = array(b)
        array(b) = array(c)
        b += 1
        array(c) = temp
        c -= 1
      }
    }
    length = if (a - start < b - a) a - start else b - a
    var l = start
    var h = b - length
    while (length > 0) {
      temp = array(l)
      array(l) = array(h)
      l += 1
      array(h) = temp
      h += 1
      length -= 1
    }
    length = if (d - c < end - 1 - d) d - c else end - 1 - d
    l = b
    h = end - length
    while (length > 0) {
      temp = array(l)
      array(l) = array(h)
      l += 1
      array(h) = temp
      h += 1
      length -= 1
    }
    length = b - a
    if (length > 0) {
      quickSort0(start, start + length, array, comp)
    }
    length = d - c
    if (length > 0) {
      quickSort0(end - length, end, array, comp)
    }
  }
  
  /**
   * Sorts some external data with QuickSort.
   * 
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparator.
   * @param swap an object that can exchange the positions of two items.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  def quickSort(start: Int, end: Int, comp: NumericComparator[Int], swap: Swapper) {
    checkBounds(end + 1, start, end)
    quickSort0(start, end, comp, swap)
  }
  
  private def quickSort0(start: Int, end: Int, comp: NumericComparator[Int], swap: Swapper) {
    var length = end - start
    if (length < 7) {
      insertionSort(start, end, comp, swap)
      return
    }
    var middle = (start + end) / 2
    if (length > 7) {
      var bottom = start
      var top = end - 1
      if (length > 40) {
        // for lots of data, bottom, middle and top are medians near the beginning, middle or end of the data
        val skosh = length / 8;
        bottom = med3(bottom, bottom + skosh, bottom + (2 * skosh), comp)
        middle = med3(middle - skosh, middle, middle + skosh, comp)
        top = med3(top - (2 * skosh), top - skosh, top, comp)
      }
      middle = med3(bottom, middle, top, comp)
    }

    var partitionIndex = middle // an index, not a value.
    
    // regions from a to b and from c to d are what we will recursively sort
    var a = start
    var b = a
    var c = end - 1
    var d = c
    while (b <= c) {
      // copy all values equal to the partition value to before a..b.  In the process, advance b
      // as long as values less than the partition or equal are found, also stop when a..b collides with c..d
      var comparison = comp.compare(b, partitionIndex)
      while (b <= c && comparison <= 0) {
        if (comparison == 0) {
          if (a == partitionIndex) {
            partitionIndex = b
          } else if (b == partitionIndex) {
            partitionIndex = a
          }
          swap.swap(a, b)
          a += 1
        }
        b += 1
        comparison = comp.compare(b, partitionIndex)
      }
      // at this point [start..a) has partition values, [a..b) has values < partition
      // also, either b>c or v[b] > partition value

      comparison = comp.compare(c, partitionIndex)
      while (c >= b && comparison >= 0) {
        if (comparison == 0) {
          if (c == partitionIndex) {
            partitionIndex = d
          } else if (d == partitionIndex) {
            partitionIndex = c
          }
          swap.swap(c, d)

          d -= 1
        }
        c -= 1
        comparison = comp.compare(c, partitionIndex)
      }
      // now we also know that [d..end] contains partition values,
      // [c..d) contains values > partition value
      // also, either b>c or (v[b] > partition OR v[c] < partition)

      if (b <= c) {
        // v[b] > partition OR v[c] < partition
        // swapping will let us continue to grow the two regions
        if (c == partitionIndex) {
          partitionIndex = b
        } else if (b == partitionIndex) {
          partitionIndex = d
        }
        swap.swap(b, c)
        b += 1
        c -= 1
      }
    }
    // now we know
    // b = c+1
    // [start..a) and [d..end) contain partition value
    // all of [a..b) are less than partition
    // all of [c..d) are greater than partition

    // shift [a..b) to beginning
    length = math.min(a - start, b - a);
    var l = start
    var h = b - length
    while (length > 0) {
      swap.swap(l, h)
      l += 1
      h += 1
      length -= 1
    }

    // shift [c..d) to end
    length = math.min(d - c, end - 1 - d)
    l = b
    h = end - length
    while (length > 0) {
      swap.swap(l, h)
      l += 1
      h += 1
      length -= 1
    }

    // recurse left and right
    length = b - a;
    if (length > 0) {
      quickSort0(start, start + length, comp, swap)
    }

    length = d - c;
    if (length > 0) {
      quickSort0(end - length, end, comp, swap)
    }
  }
  
  /**
   * In-place insertion sort that is fast for pre-sorted data.
   *
   * @param start Where to start sorting (inclusive)
   * @param end   Where to stop (exclusive)
   * @param comp  Sort order.
   * @param swap  How to swap items.
   */
  private def insertionSort(start: Int, end: Int, comp: NumericComparator[Int], swap: Swapper) {
    for (i <- start+1 until end) {
      for (j <- (start+1 to i).reverse if comp.compare(j - 1, j) > 0) {
        swap.swap(j - 1, j)
      }
    }
  }
  
//  private val NATURAL_NUMERIC_COMPARISON = new NumericComparator[Double]() {
//    override def compare(o1: , float o2): Int = {
//      return Float.compare(o1, o2);
//    }
//  };
    
    /**
     * Perform a merge sort on a range of a numeric array using Float.compare for an ordering.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  def mergeSort[T: Numeric: ClassManifest](array: Array[T], start: Int, end: Int) {
    mergeSort(array, start, end, new NumericComparator[T]() {
      override def compare(o1: T, o2: T) : Int = (o1 - o2).toInt
    })
  }

  /**
   * Perform a merge sort on a range of a numeric array using a specified ordering.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  def mergeSort[T: Numeric: ClassManifest](array: Array[T], start: Int, end: Int, 
      comp: NumericComparator[T]) {
    checkBounds(array.length, start, end)
    val out = Arrays.copyOf(array, array.length)
    mergeSort(out, array, start, end, comp);
  }

  private def mergeSort[T: Numeric: ClassManifest](in: Array[T], out: Array[T], start0: Int, 
      end0: Int, c: NumericComparator[T]) {
    var start = start0
    var end = end0
    val len = end - start
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (i <- start+1 until end) {
        val current = out(i)
        var prev = out(i - 1)
        if (c.compare(prev, current) > 0) {
          var j = i
          do {
            out(j) = prev
            j -= 1
            prev = out(j - 1)
          } while (j > start && (c.compare(prev, current) > 0))
          out(j) = current
        }
      }
      return;
    }
    val med = (end + start) >>> 1
    mergeSort(out, in, start, med, c)
    mergeSort(out, in, med, end, c)
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in(med - 1), in(med)) <= 0) {
      Array.copy(in, start, out, start, len)
      return
    }
    var r = med
    var i = start

    // use merging with exponential search
    do {
      val fromVal = in(start)
      val rVal = in(r)
      if (c.compare(fromVal, rVal) <= 0) {
        val l_1 = find(in, rVal, -1, start + 1, med - 1, c)
        val toCopy = l_1 - start + 1
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out(i) = rVal
        i += 1
        r += 1
        start = l_1 + 1
      } else {
        val r_1 = find(in, fromVal, 0, r + 1, end - 1, c)
        val toCopy = r_1 - r + 1
        Array.copy(in, r, out, i, toCopy)
        i += toCopy
        out(i) = fromVal
        i += 1
        start += 1
        r = r_1 + 1
      }
    } while ((end - r) > 0 && (med - start) > 0)
    
    // copy rest of array
    if ((end - r) <= 0) {
      Array.copy(in, start, out, i, med - start)
    } else {
      Array.copy(in, r, out, i, end - r)
    }
  }

  private def find[T: Numeric: ClassManifest](arr: Array[T], value: T, bnd: Int, l: Int, 
      r: Int, c: NumericComparator[T]): Int = {
    var left = l
    var right = r
    var m = left
    var d = 1
    while (m <= right) {
      if (c.compare(value, arr(m)) > bnd) {
        left = m + 1
        m += d
        d <<= 1
      } else {
        right = m - 1
      }
    }
    while (left <= right) {
      m = (left + right) >>> 1
      if (c.compare(value, arr(m)) > bnd) {
        left = m + 1
      } else {
        right = m - 1
      }
    }
    l - 1
  }
 
  //TODO:
//  private val NATURAL_NUMERIC_COMPARISON = new NumericComparator[T]() {
//    override def compare(double o1, double o2): Int = {
//      (o1 - o2).toInt
//    }
//  }
}