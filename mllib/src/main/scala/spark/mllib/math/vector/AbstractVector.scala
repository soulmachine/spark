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

package spark.mllib.math.vector

/** Implementations of generic capabilities like getLengthSquared and normalize. 
 * And Implementations of operations between two different kinds of vectors.
 */
abstract class AbstractVector(val dimension: Int) extends Vector with LengthCachingVector {
  protected var lengthSquared = -1.0

  def getLengthSquared(): Double = {
    if (lengthSquared >= 0.0) {
      lengthSquared
    } else {
      lengthSquared = dotSelf()
      lengthSquared
    }
  }

  def invalidateCachedLength() {
    lengthSquared = -1
  }
  
  def normalize(): Vector = this / scala.math.sqrt(getLengthSquared())
  
  def normalize(power: Double): Vector =  this / (norm(power))
  
  def logNormalize(): Vector = logNormalize(2.0, scala.math.sqrt(getLengthSquared()))

  def logNormalize(power: Double): Vector = logNormalize(power, norm(power))

  protected def logNormalize(power: Double, norm: Double): Vector
  
  protected def dotSelf(): Double
  
  override def canEqual(that: Any): Boolean = 
    that.isInstanceOf[Vector]

  protected def compareTo(that: Vector): Boolean = {
    var isEqual = true
    for (i <- 0 until dimension if isEqual) {
      if (this(i) != that(i)) isEqual = false
    }
    isEqual
  }
  
  override def hashCode(): Int = {
    var result = dimension
    for (i <- 0 until dimension) {
      val value = this(i)
      if (value != 0.0) result += i * value.hashCode
    }
    result
  }
}
