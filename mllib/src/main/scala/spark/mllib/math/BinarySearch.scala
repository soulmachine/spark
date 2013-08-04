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

final object BinarySearch {
  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code double} array to search.
   * @param value
   *          the {@code double} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  def binarySearchFromTo[T: Numeric: ClassManifest](array: Array[T], value: T, from: Int, to: Int): Int = {
    var low = from
    var high = to
    var mid = -1
    while (low <= high) {
      mid = (low + high) >>> 1
      if (array(mid) < value) {
        low = mid + 1
      } else if (array(mid) == value) {
        return mid
      } else {
        high = mid - 1
      }
    }
    if (mid < 0) {
      return -1
    }
    return -mid - (if(value < array(mid)) 1 else 2)
  }
}