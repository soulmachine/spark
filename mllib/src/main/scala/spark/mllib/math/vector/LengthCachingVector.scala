/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package spark.mllib.math.vector

/**
 * Marker interface for vectors that may cache their squared length.
 */
trait LengthCachingVector {
  /**
   * Gets the currently cached squared length or if there is none, recalculates
   * the value and returns that.
   * @return The sum of the squares of all elements in the vector. 
   * Square root of this value is the length of the vector.
   */
  def lengthSquared: Double

  /**
   * Invalidates the length cache.  This should be called by all mutators of the vector.
   */
  def invalidateCachedLength()
}
