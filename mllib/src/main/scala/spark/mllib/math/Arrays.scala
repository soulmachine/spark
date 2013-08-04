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

/**
 * Array manipulations; complements <tt>java.util.Arrays</tt>.
 *
 * @see java.util.Arrays
 * @see org.apache.mahout.math.Sorting
 *
 */
final object Arrays {

  /**
   * Ensures that a given array can hold up to <tt>minCapacity</tt> elements.
   *
   * Returns the identical array if it can hold at least the number of elements specified. Otherwise, returns a new
   * array with increased capacity containing the same elements, ensuring that it can hold at least the number of
   * elements specified by the minimum capacity argument.
   *
   * @param minCapacity the desired minimum capacity.
   */
  def ensureCapacity[T: ClassManifest](array: Array[T], minCapacity: Int): Array[T] = {
    val oldCapacity = array.length

    if (minCapacity > oldCapacity) {
      val newCapacity = math.max((oldCapacity * 3) / 2 + 1, minCapacity)
      val newArray = new Array[T](newCapacity)
      Array.copy(array, 0, newArray, 0, oldCapacity)
      newArray
    } else {
      array
    }
  }

  /**
   * Returns a string representation of the specified array.  The string representation consists of a list of the
   * arrays's elements, enclosed in square brackets (<tt>"[]"</tt>).  Adjacent elements are separated by the characters
   * <tt>", "</tt> (comma and space).
   *
   * @return a string representation of the specified array.
   */
  def toString[T: ClassManifest](array: Array[T]): String = {
    val buf = new StringBuilder()
    buf.append('[')
    val maxIndex = array.length - 1
    for (i <- 0 to maxIndex) {
      buf.append(array(i))
      if (i < maxIndex) {
        buf.append(", ")
      }
    }
    buf.append(']')
    buf.toString()
  }

  /**
   * Ensures that the specified array cannot hold more than <tt>maxCapacity</tt> elements. An application can use this
   * operation to minimize array storage. <p> Returns the identical array if <tt>array.length &lt;= maxCapacity</tt>.
   * Otherwise, returns a new array with a length of <tt>maxCapacity</tt> containing the first <tt>maxCapacity</tt>
   * elements of <tt>array</tt>.
   *
   * @param maxCapacity the desired maximum capacity.
   */
  def trimToCapacity[T: ClassManifest](array: Array[T], maxCapacity: Int): Array[T] = {
    if (array.length > maxCapacity) {
      val newArray = new Array[T](maxCapacity)
      Array.copy(array, 0, newArray, 0, maxCapacity)
      newArray
    } else {
      array
    }
  }

  /**
   * {@link java.util.Arrays#copyOf} compatibility with Java 1.5.
   */
  def copyOf[T: ClassManifest](src: Array[T], length: Int): Array[T] = {
    val result = new Array[T](length)
    Array.copy(src, 0, result, 0, math.min(length, src.length))
    result;
  }
}
