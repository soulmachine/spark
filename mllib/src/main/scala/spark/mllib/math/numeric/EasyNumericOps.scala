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
package spark.mllib.math.numeric

import scala.{specialized => spec}

/**
 * @author Erik Osheim
 *
 * NumericOps adds operators to A. It's intended to be used as an implicit
 * decorator like so:
 *
 *   def foo[A:Numeric](a:A, b:A) = a + b
 * 
 *   (compiled into) = new NumericOps(a).+(b)
 *   (w/plugin into) = numeric.add(a, b)
 */
final class EasyNumericOps[@spec(Int,Long,Float,Double) A:Numeric](val lhs:A) {
  val n = implicitly[Numeric[A]]

  def abs = n.abs(lhs)
  def unary_- = n.negate(lhs)
  def signum = n.signum(lhs)

  def compare[B:ConvertableFrom](rhs:B) = n.compare(lhs, n.fromType(rhs))
  def equiv[B:ConvertableFrom](rhs:B) = n.equiv(lhs, n.fromType(rhs))
  def max[B:ConvertableFrom](rhs:B) = n.max(lhs, n.fromType(rhs))
  def min[B:ConvertableFrom](rhs:B) = n.min(lhs, n.fromType(rhs))

  def <=>[B:ConvertableFrom](rhs:B) = n.compare(lhs, n.fromType(rhs))
  def ===[B:ConvertableFrom](rhs:B) = n.equiv(lhs, n.fromType(rhs))
  def !==[B:ConvertableFrom](rhs:B) = n.nequiv(lhs, n.fromType(rhs))

  def /[B:ConvertableFrom](rhs:B) = n.div(lhs, n.fromType(rhs))
  def >[B:ConvertableFrom](rhs:B) = n.gt(lhs, n.fromType(rhs))
  def >=[B:ConvertableFrom](rhs:B) = n.gteq(lhs, n.fromType(rhs))
  def <[B:ConvertableFrom](rhs:B) = n.lt(lhs, n.fromType(rhs))
  def <=[B:ConvertableFrom](rhs:B) = n.lteq(lhs, n.fromType(rhs))
  def -[B:ConvertableFrom](rhs:B) = n.minus(lhs, n.fromType(rhs))
  def %[B:ConvertableFrom](rhs:B) = n.mod(lhs, n.fromType(rhs))
  def +[B:ConvertableFrom](rhs:B) = n.plus(lhs, n.fromType(rhs))
  def *[B:ConvertableFrom](rhs:B) = n.times(lhs, n.fromType(rhs))
  def **[B:ConvertableFrom](rhs:B) = n.pow(lhs, n.fromType(rhs))

  def toByte = n.toByte(lhs)
  def toShort = n.toShort(lhs)
  def toInt = n.toInt(lhs)
  def toLong = n.toLong(lhs)
  def toFloat = n.toFloat(lhs)
  def toDouble = n.toDouble(lhs)
  def toBigInt = n.toBigInt(lhs)
  def toBigDecimal = n.toBigDecimal(lhs)
}
