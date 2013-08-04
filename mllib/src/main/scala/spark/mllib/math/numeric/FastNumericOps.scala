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
 */package spark.mllib.math.numeric

/**
 * @author Erik Osheim
 */

import scala.{specialized => spec}

/**
 * NumericOps adds things like inline operators to A. It's intended to
 * be used as an implicit decorator like so:
 *
 *   def foo[A:Numeric](a:A, b:A) = a + b
 *      (this is translated into) = new NumericOps(a).+(b)
 */
final class FastNumericOps[@spec(Int,Long,Float,Double) A:Numeric](val lhs:A) {
  val n = implicitly[Numeric[A]]

  def abs = n.abs(lhs)
  def unary_- = n.negate(lhs)
  def signum = n.signum(lhs)

  def compare(rhs:A) = n.compare(lhs, rhs)
  def equiv(rhs:A) = n.equiv(lhs, rhs)
  def max(rhs:A) = n.max(lhs, rhs)
  def min(rhs:A) = n.min(lhs, rhs)

  def <=>(rhs:A) = n.compare(lhs, rhs)
  def ===(rhs:A) = n.equiv(lhs, rhs)
  def !==(rhs:A) = n.nequiv(lhs, rhs)
  def >(rhs:A) = n.gt(lhs, rhs)
  def >=(rhs:A) = n.gteq(lhs, rhs)
  def <(rhs:A) = n.lt(lhs, rhs)
  def <=(rhs:A) = n.lteq(lhs, rhs)
  def /(rhs:A) = n.div(lhs, rhs)
  def -(rhs:A) = n.minus(lhs, rhs)
  def %(rhs:A) = n.mod(lhs, rhs)
  def +(rhs:A) = n.plus(lhs, rhs)
  def *(rhs:A) = n.times(lhs, rhs)
  def **(rhs:A) = n.pow(lhs, rhs)
  def log = n.log(lhs)

  def toByte = n.toByte(lhs)
  def toShort = n.toShort(lhs)
  def toInt = n.toInt(lhs)
  def toLong = n.toLong(lhs)
  def toFloat = n.toFloat(lhs)
  def toDouble = n.toDouble(lhs)
  def toBigInt = n.toBigInt(lhs)
  def toBigDecimal = n.toBigDecimal(lhs)
}
