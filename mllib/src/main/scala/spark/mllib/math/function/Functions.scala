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

package spark.mllib.math.function

import cern.jet.random.engine.MersenneTwister

import java.util.Date

/**
 * Function objects to be passed to generic methods, ported from mahout.
 *
 * Contains the functions of {@link java.lang.Math} as function
 * objects, as well as a few more basic functions. <p>Function objects conveniently allow to express arbitrary functions
 * in a generic manner. Essentially, a function object is an object that can perform a function on some arguments. It
 * has a minimal interface: a method <tt>apply</tt> that takes the arguments, computes something and returns some result
 * value. Function objects are comparable to function pointers in C used for call-backs. <p>Unary functions are of type
 * {@link org.apache.mahout.math.function.DoubleFunction}, binary functions of type {@link
 * org.apache.mahout.math.function.DoubleDoubleFunction}. All can be retrieved via <tt>public static final</tt>
 * variables named after the function. Unary predicates are of type
 * {@link DoubleProcedure},
 * binary predicates of type {@link org.apache.mahout.math.function.DoubleDoubleProcedure}. All can be retrieved via
 * <tt>public static final</tt> variables named <tt>isXXX</tt>.
 *
 * <p> Binary functions and predicates also exist as unary functions with the second argument being fixed to a constant.
 * These are generated and retrieved via factory methods (again with the same name as the function). Example: <ul>
 * <li><tt>Functions.pow</tt> gives the function <tt>a<sup>b</sup></tt>. <li><tt>Functions.pow.apply(2,3)==8</tt>.
 * <li><tt>Functions.pow(3)</tt> gives the function <tt>a<sup>3</sup></tt>. <li><tt>Functions.pow(3).apply(2)==8</tt>.
 * </ul> More general, any binary function can be made an unary functions by fixing either the first or the second
 * argument. See methods {@link #bindArg1(org.apache.mahout.math.function.DoubleDoubleFunction ,double)} and {@link
 * #bindArg2(org.apache.mahout.math.function.DoubleDoubleFunction ,double)}. The order of arguments can
 * be swapped so that the first argument becomes the
 * second and vice-versa. See method {@link #swapArgs(org.apache.mahout.math.function.DoubleDoubleFunction)}.
 * Example: <ul> <li><tt>Functions.pow</tt>
 * gives the function <tt>a<sup>b</sup></tt>. <li><tt>Functions.bindArg2(Functions.pow,3)</tt> gives the function
 * <tt>x<sup>3</sup></tt>. <li><tt>Functions.bindArg1(Functions.pow,3)</tt> gives the function <tt>3<sup>x</sup></tt>.
 * <li><tt>Functions.swapArgs(Functions.pow)</tt> gives the function <tt>b<sup>a</sup></tt>. </ul> <p> Even more
 * general, functions can be chained (composed, assembled). Assume we have two unary functions <tt>g</tt> and
 * <tt>h</tt>. The unary function <tt>g(h(a))</tt> applying both in sequence can be generated via {@link
 * #chain(org.apache.mahout.math.function.DoubleFunction , org.apache.mahout.math.function.DoubleFunction)}:
 * <ul> <li><tt>Functions.chain(g,h);</tt> </ul> Assume further we have a binary
 * function <tt>f</tt>. The binary function <tt>g(f(a,b))</tt> can be generated via {@link
 * #chain(org.apache.mahout.math.function.DoubleFunction , org.apache.mahout.math.function.DoubleDoubleFunction)}:
 * <ul> <li><tt>Functions.chain(g,f);</tt> </ul> The binary function
 * <tt>f(g(a),h(b))</tt> can be generated via
 * {@link #chain(org.apache.mahout.math.function.DoubleDoubleFunction , org.apache.mahout.math.function.DoubleFunction ,
 * org.apache.mahout.math.function.DoubleFunction)}: <ul>
 * <li><tt>Functions.chain(f,g,h);</tt> </ul> Arbitrarily complex functions can be composed from these building blocks.
 * For example <tt>sin(a) + cos<sup>2</sup>(b)</tt> can be specified as follows: <ul>
 * <li><tt>chain(plus,sin,chain(square,cos));</tt> </ul> or, of course, as
 * <pre>
 * new DoubleDoubleFunction() {
 * &nbsp;&nbsp;&nbsp;public final double apply(double a, double b) { return math.sin(a) + math.pow(math.cos(b),2); }
 * }
 * </pre>
 * <p> For aliasing see functions. Try this <table> <td class="PRE">
 * <pre>
 * // should yield 1.4399560356056456 in all cases
 * double a = 0.5;
 * double b = 0.2;
 * double v = math.sin(a) + math.pow(math.cos(b),2);
 * log.info(v);
 * Functions F = Functions.functions;
 * DoubleDoubleFunction f = F.chain(F.plus,F.sin,F.chain(F.square,F.cos));
 * log.info(f.apply(a,b));
 * DoubleDoubleFunction g = new DoubleDoubleFunction() {
 * &nbsp;&nbsp;&nbsp;public double apply(double a, double b) { return math.sin(a) + math.pow(math.cos(b),2); }
 * };
 * log.info(g.apply(a,b));
 * </pre>
 * </td> </table>
 *
 * <p> <H3>Performance</H3>
 *
 * Surprise. Using modern non-adaptive JITs such as SunJDK 1.2.2 (java -classic) there seems to be no or only moderate
 * performance penalty in using function objects in a loop over traditional code in a loop. For complex nested function
 * objects (e.g. <tt>F.chain(F.abs,F.chain(F.plus,F.sin,F.chain(F.square,F.cos)))</tt>) the penalty is zero, for trivial
 * functions (e.g. <tt>F.plus</tt>) the penalty is often acceptable. <center> <table border cellpadding="3"
 * cellspacing="0" align="center"> <tr valign="middle" bgcolor="#33CC66" nowrap align="center"> <td nowrap colspan="7">
 * <font size="+2">Iteration Performance [million function evaluations per second]</font><br> <font size="-1">Pentium
 * Pro 200 Mhz, SunJDK 1.2.2, NT, java -classic, </font></td> </tr> <tr valign="middle" bgcolor="#66CCFF" nowrap
 * align="center"> <td nowrap bgcolor="#FF9966" rowspan="2">&nbsp;</td> <td bgcolor="#FF9966" colspan="2"> <p> 30000000
 * iterations</p> </td> <td bgcolor="#FF9966" colspan="2"> 3000000 iterations (10 times less)</td> <td bgcolor="#FF9966"
 * colspan="2">&nbsp;</td> </tr> <tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> <td bgcolor="#FF9966">
 * <tt>F.plus</tt></td> <td bgcolor="#FF9966"><tt>a+b</tt></td> <td bgcolor="#FF9966">
 * <tt>F.chain(F.abs,F.chain(F.plus,F.sin,F.chain(F.square,F.cos)))</tt></td> <td bgcolor="#FF9966">
 * <tt>math.abs(math.sin(a) + math.pow(math.cos(b),2))</tt></td> <td bgcolor="#FF9966">&nbsp;</td> <td
 * bgcolor="#FF9966">&nbsp;</td> </tr> <tr valign="middle" bgcolor="#66CCFF" nowrap align="center"> <td nowrap
 * bgcolor="#FF9966">&nbsp;</td> <td nowrap>10.8</td> <td nowrap>29.6</td> <td nowrap>0.43</td> <td nowrap>0.35</td> <td
 * nowrap>&nbsp;</td> <td nowrap>&nbsp;</td> </tr> </table></center>
 */
object Functions {

  /*
   * <H3>Unary functions</H3>
   */
  /** Function that returns <tt>math.abs(a)</tt>. */
  val ABS = new DoubleFunction() {
    def apply(a: Double): Double = math.abs(a)
  }

  /** Function that returns <tt>math.acos(a)</tt>. */
  val ACOS = new DoubleFunction() {
    def apply(a: Double): Double = math.acos(a)
  }

  /** Function that returns <tt>math.asin(a)</tt>. */
  val ASIN = new DoubleFunction() {
    def apply(a: Double): Double = math.asin(a)
  }

  /** Function that returns <tt>math.atan(a)</tt>. */
  val ATAN = new DoubleFunction() {
    override def apply(a: Double): Double = math.atan(a)
  }

  /** Function that returns <tt>math.ceil(a)</tt>. */
  val CEIL = new DoubleFunction() {
    override def apply(a: Double): Double = math.ceil(a)
  }

  /** Function that returns <tt>math.cos(a)</tt>. */
  val COS = new DoubleFunction() {
    override def apply(a: Double): Double = math.cos(a)
  }

  /** Function that returns <tt>math.exp(a)</tt>. */
  val EXP = new DoubleFunction() {
    override def apply(a: Double): Double = math.exp(a)
  }

  /** Function that returns <tt>math.floor(a)</tt>. */
  val FLOOR = new DoubleFunction() {
    override def apply(a: Double): Double = math.floor(a)
  }

  /** Function that returns its argument. */
  val IDENTITY = new DoubleFunction() {
    override def apply(a: Double): Double = a
  }

  /** Function that returns <tt>1.0 / a</tt>. */
  val INV = new DoubleFunction() {
    override def apply(a: Double): Double = 1.0 / a
  }

  /** Function that returns <tt>math.log(a)</tt>. */
  val LOGARITHM = new DoubleFunction() {
    override def apply(a: Double): Double = math.log(a)
  }

  /** Function that returns <tt>math.log(a) / math.log(2)</tt>. */
  val LOG2 = new DoubleFunction() {
    override def apply(a: Double): Double = math.log(a) * 1.4426950408889634
  }

  /** Function that returns <tt>-a</tt>. */
  val NEGATE = new DoubleFunction() {
    override def apply(a: Double): Double = -a
  }

  /** Function that returns <tt>math.rint(a)</tt>. */
  val RINT = new DoubleFunction() {
    override def apply(a: Double): Double = math.rint(a)
  }

  /** Function that returns <tt>a < 0 ? -1 : a > 0 ? 1 : 0</tt>. */
  val SIGN = new DoubleFunction() {
    override def apply(a: Double): Double = if (a < 0) -1 else if (a > 0) 1 else 0
  }

  /** Function that returns <tt>math.sin(a)</tt>. */
  val SIN = new DoubleFunction() {
    override def apply(a: Double): Double = math.sin(a)
  }

  /** Function that returns <tt>math.sqrt(a)</tt>. */
  val SQRT = new DoubleFunction() {
    override def apply(a: Double): Double = math.sqrt(a)
  }

  /** Function that returns <tt>a * a</tt>. */
  val SQUARE = new DoubleFunction() {
    override def apply(a: Double): Double = a * a
  }

  /** Function that returns <tt> 1 / (1 + exp(-a) </tt> */
  val SIGMOID = new DoubleFunction() {
    override def apply(a: Double): Double = 1.0 / (1.0 + math.exp(-a))
  }

  /** Function that returns <tt> a * (1-a) </tt> */
  val SIGMOIDGRADIENT = new DoubleFunction() {
    override def apply(a: Double): Double = a * (1.0 - a)
  }

  /** Function that returns <tt>math.tan(a)</tt>. */
  val TAN = new DoubleFunction() {
    override def apply(a: Double): Double = math.tan(a)
  }

  /*
   * <H3>Binary functions</H3>
   */

  /** Function that returns <tt>math.atan2(a,b)</tt>. */
  val ATAN2 = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.atan2(a, b)
  }

  /** Function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>. */
  val COMPARE = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = if (a < b) -1 else if (a > b) 1 else 0
  }

  /** Function that returns <tt>a / b</tt>. */
  val DIV = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = a / b

    /**
     * x / 0 = infinity or undefined depending on x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * 0 / y = 0 unless y = 0
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * x / 0 = infinity or undefined depending on x
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * x / y != y / x
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = false

    /**
     * x / (y / z) = x * z / y
     * (x / y) / z = x / (y * z)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false

  }

  /** Function that returns <tt>a == b ? 1 : 0</tt>. */
  val EQUALS = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = if (a == b) 1 else 0

    /**
     * x = y iff y = x
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true
  }

  /** Function that returns <tt>a > b ? 1 : 0</tt>. */
  val GREATER = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = if (a > b) 1 else 0
  }

  /** Function that returns <tt>math.IEEEremainder(a,b)</tt>. */
  val IEEE_REMAINDER = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.IEEEremainder(a, b)
  }

  /** Function that returns <tt>a == b</tt>. */
  val IS_EQUAL = new DoubleDoubleProcedure() {
    override def apply(a: Double, b: Double): Boolean = a == b
  }

  /** Function that returns <tt>a < b</tt>. */
  val IS_LESS = new DoubleDoubleProcedure() {
    override def apply(a: Double, b: Double): Boolean = a < b
  }

  /** Function that returns <tt>a > b</tt>. */
  val IS_GREATER = new DoubleDoubleProcedure() {
    override def apply(a: Double, b: Double): Boolean = a > b
  }

  /** Function that returns <tt>a < b ? 1 : 0</tt>. */
  val LESS = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = if (a < b) 1 else 0
  }

  /** Function that returns <tt>math.log(a) / math.log(b)</tt>. */
  val LG = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.log(a) / math.log(b)
  }

  /** Function that returns <tt>math.max(a,b)</tt>. */
  val MAX = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.max(a, b)

    /**
     * max(x, 0) = x or 0 depending on the sign of x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * max(0, y) = y or 0 depending on the sign of y
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * max(x, 0) = x or 0 depending on the sign of x
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * max(x, max(y, z)) = max(max(x, y), z)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = true

    /**
     * max(x, y) = max(y, x)
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true
  }

  val MAX_ABS = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.max(math.abs(a), math.abs(b))

    /**
     * max(|x|, 0) = |x|
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = true

    /**
     * max(0, |y|) = |y|
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * max(|x|, 0) = |x|
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * max(|x|, max(|y|, |z|)) = max(max(|x|, |y|), |z|)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = true

    /**
     * max(|x|, |y|) = max(|y\, |x\)
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true
  }

  /** Function that returns <tt>math.min(a,b)</tt>. */
  val MIN = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.min(a, b)

    /**
     * min(x, 0) = x or 0 depending on the sign of x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * min(0, y) = y or 0 depending on the sign of y
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * min(x, 0) = x or 0 depending on the sign of x
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * min(x, min(y, z)) = min(min(x, y), z)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = true

    /**
     * min(x, y) = min(y, x)
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true
  }

  /** Function that returns <tt>a - b</tt>. */
  val MINUS = plusMult(-1)

  val MINUS_SQUARED = new DoubleDoubleFunction() {
    override def apply(x: Double, y: Double): Double = (x - y) * (x - y)

    /**
     * (x - 0)^2 = x^2 != x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * (0 - y)^2 != 0
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * (x - 0)^2 != x
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * (x - y)^2 = (y - x)^2
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true

    /**
     * (x - (y - z)^2)^2 != ((x - y)^2 - z)^2
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  /** Function that returns <tt>a % b</tt>. */
  val MOD = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = a % b
  }

  /** Function that returns <tt>a * b</tt>. */
  val MULT = new TimesFunction()

  /** Function that returns <tt>a + b</tt>. */
  val PLUS = plusMult(1)

  /** Function that returns <tt>math.abs(a) + math.abs(b)</tt>. */
  val PLUS_ABS = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.abs(a) + math.abs(b)

    /**
     * abs(x) + abs(0) = abs(x) != x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * abs(0) + abs(y) = abs(y) != 0 unless y = 0
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * abs(x) + abs(0) = abs(x) != 0 unless x = 0
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * abs(x) + abs(abs(y) + abs(z)) = abs(x) + abs(y) + abs(z)
     * abs(abs(x) + abs(y)) + abs(z) = abs(x) + abs(y) + abs(z)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = true

    /**
     * abs(x) + abs(y) = abs(y) + abs(x)
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true
  }

  val MINUS_ABS = new DoubleDoubleFunction() {
    override def apply(x: Double, y: Double): Double = math.abs(x - y)

    /**
     * |x - 0| = |x|
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * |0 - y| = |y|
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * |x - 0| = |x|
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * |x - y| = |y - x|
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true

    /**
     * |x - |y - z|| != ||x - y| - z| (|5 - |4 - 3|| = 1; ||5 - 4| - 3| = |1 - 3| = 2)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  /** Function that returns <tt>math.pow(a,b)</tt>. */
  val POW = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = math.pow(a, b)

    /**
     * x^0 = 1 for any x unless x = 0 (undefined)
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * 0^y = 0 for any y unless y = 0 (undefined, but math.pow(0, 0) = 1)
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * x^0 = 1 for any x (even x = 0)
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * x^y != y^x (2^3 != 3^2)
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = false

    /**
     * x^(y^z) != (x^y)^z ((2^3)^4 = 8^4 = 2^12 != 2^(3^4) = 2^81)
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  val SECOND = new DoubleDoubleFunction() {
    override def apply(x: Double, y: Double): Double = y

    /**
     * f(x, 0) = x for any x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * f(0, y) = y for any y
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * f(x, 0) = 0 for any x
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = true

    /**
     * f(x, y) = x != y = f(y, x) for any x, y unless x = y
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = false

    /**
     * f(x, f(y, z)) = f(x, z) = z
     * f(f(x, y), z) = z
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = true
  }

  /**
   * This function is specifically designed to be used when assigning a vector to one that is all zeros (created
   * by like()). It enables iteration only through the nonzeros of the right hand side by declaring isLikeRightPlus
   * to be true. This is NOT generally true for SECOND (hence the other function above).
   */
  val SECOND_LEFT_ZERO = new DoubleDoubleFunction() {
    override def apply(x: Double, y: Double): Double = {
      require(x == 0, "This special version of SECOND needs x == 0")
      return y
    }

    /**
     * f(x, 0) = 0 for any x; we're only assigning to left hand sides that are strictly 0
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = true

    /**
     * f(0, y) = y for any y
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * f(x, 0) = 0 for any x
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = true

    /**
     * f(x, y) = x != y = f(y, x) for any x, y unless x = y
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = false

    /**
     * f(x, f(y, z)) = f(x, z) = z
     * f(f(x, y), z) = z
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = true
  }

  val MULT_SQUARE_LEFT = new DoubleDoubleFunction() {
    override def apply(x: Double, y: Double): Double = x * x * y

    /**
     * x * x * 0 = 0
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * 0 * 0 * y = 0
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = true

    /**
     * x * x * 0 = 0
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = true

    /**
     * x * x * y != y * y * x
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = false

    /**
     * x * x * y * y * z != x * x * y * x * x * y * z
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  val MULT_RIGHT_PLUS1 = new DoubleDoubleFunction() {
    /**
     * Apply the function to the arguments and return the result
     *
     * @param x a double for the first argument
     * @param y a double for the second argument
     * @return the result of applying the function
     */
    override def apply(x: Double, y: Double): Double = x * (y + 1)

    /**
     * x * 1 = x
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = true

    /**
     * 0 * y = 0
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = true

    /**
     * x * 1 = x != 0
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * x * (y + 1) != y * (x + 1)
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = false

    /**
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  def reweigh(wx: Double, wy: Double): DoubleDoubleFunction = {
    val tw = wx + wy
    new DoubleDoubleFunction() {
      override def apply(x: Double, y: Double): Double = (wx * x + wy * y) / tw

      /**
       * f(x, 0) = wx * x / tw = x iff wx = tw (practically, impossible, as tw = wx + wy and wy > 0)
       * @return true iff f(x, 0) = x for any x
       */
      override def isLikeRightPlus: Boolean = wx == tw

      /**
       * f(0, y) = wy * y / tw = 0 iff y = 0
       * @return true iff f(0, y) = 0 for any y
       */
      override def isLikeLeftMult: Boolean = false

      /**
       * f(x, 0) = wx * x / tw = 0 iff x = 0
       * @return true iff f(x, 0) = 0 for any x
       */
      override def isLikeRightMult: Boolean = false

      /**
       * wx * x + wy * y = wx * y + wy * x iff wx = wy
       * @return true iff f(x, y) = f(y, x) for any x, y
       */
      override def isCommutative: Boolean = wx == wy

      /**
       * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
       */
      override def isAssociative: Boolean = false
    }
  }

  /**
   * Constructs a function that returns <tt>(from<=a && a<=to) ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>from</tt> and
   * <tt>to</tt> are fixed.
   */
  def between(from: Double, to: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = if (from <= a && a <= to) 1 else 0
  }

  /**
   * Constructs a unary function from a binary function with the first operand (argument) fixed to the given constant
   * <tt>c</tt>. The second operand is variable (free).
   *
   * @param function a binary function taking operands in the form <tt>function.apply(c,var)</tt>.
   * @return the unary function <tt>function(c,var)</tt>.
   */
  def bindArg1(function: DoubleDoubleFunction, c: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = function.apply(c, a)
  }

  /**
   * Constructs a unary function from a binary function with the second operand (argument) fixed to the given constant
   * <tt>c</tt>. The first operand is variable (free).
   *
   * @param function a binary function taking operands in the form <tt>function.apply(var,c)</tt>.
   * @return the unary function <tt>function(var,c)</tt>.
   */
  def bindArg2(function: DoubleDoubleFunction, c: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = function.apply(a, c)
  }

  /**
   * Constructs the function <tt>f( g(a), h(b) )</tt>.
   *
   * @param f a binary function.
   * @param g a unary function.
   * @param h a unary function.
   * @return the binary function <tt>f( g(a), h(b) )</tt>.
   */
  def chain(f: DoubleDoubleFunction, g: DoubleFunction,
    h: DoubleFunction): DoubleDoubleFunction = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = f.apply(g.apply(a), h.apply(b))

    /**
     * fx(c, 0) = f(g(x), h(0)) = f(g(x), 0) = g(x) = x if h(0) = 0 and f isLikeRightPlus and g(x) = x
     * Impossible to check whether g(x) = x for any x, so we return false.
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * fc(0, y) = f(g(0), h(y)) = f(0, h(y)) = 0 if g(0) = 0 and f isLikeLeftMult
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = g.apply(0) == 0 && f.isLikeLeftMult

    /**
     * fc(x, 0) = f(g(x), h(0)) = f(g(x), 0) = 0 if h(0) = 0 and f isLikeRightMult
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = h.apply(0) == 0 && f.isLikeRightMult

    /**
     * fc(x, y) = f(g(x), h(y)) = f(h(y), g(x))
     * fc(y, x) = f(g(y), h(x)) = f(h(x), g(y))
     * Either g(x) = g(y) for any x, y and h(x) = h(y) for any x, y or g = h and f isCommutative.
     * Can only check if g = h (reference equality, assuming they're both the same static function in
     * this file) and f isCommutative. There are however other scenarios when this might happen that are NOT
     * covered by this definition.
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = g.equals(h) && f.isCommutative

    /**
     * fc(x, fc(y, z)) = f(g(x), h(f(g(y), h(z))))
     * fc(fc(x, y), z) = f(g(f(g(x), h(y))), h(z))
     * Impossible to check.
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  /**
   * Constructs the function <tt>g( h(a,b) )</tt>.
   *
   * @param g a unary function.
   * @param h a binary function.
   * @return the binary function <tt>g( h(a,b) )</tt>.
   */
  def chain(g: DoubleFunction, h: DoubleDoubleFunction): DoubleDoubleFunction = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = g.apply(h.apply(a, b))

    /**
     * g(h(x, 0)) = g(x) = x for any x iff g(x) = x and h isLikeRightPlus
     * Impossible to check.
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * g(h(0, y)) = g(0) = 0 for any y iff g(0) = 0 and h isLikeLeftMult
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = !g.isDensifying && h.isLikeLeftMult

    /**
     * g(h(x, 0)) = g(0) = 0 for any x iff g(0) = 0 and h isLikeRightMult
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = !g.isDensifying && h.isLikeRightMult

    /**
     * fc(x, y) = g(h(x, y)) = g(h(y, x)) = fc(y, x) iff h isCommutative
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = h.isCommutative

    /**
     * fc(x, fc(y, z)) = g(h(x, g(h(y, z)))
     * fc(fc(x, y), z) = g(h(g(h(x, y)), z))
     * Impossible to check.
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }

  /**
   * Constructs the function <tt>g( h(a) )</tt>.
   *
   * @param g a unary function.
   * @param h a unary function.
   * @return the unary function <tt>g( h(a) )</tt>.
   */
  def chain(g: DoubleFunction, h: DoubleFunction): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = g.apply(h.apply(a))
  }

  /**
   * Constructs a function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>b</tt> is
   * fixed.
   */
  def compare(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = if (a < b) -1 else if (a > b) 1 else 0
  }

  /** Constructs a function that returns the constant <tt>c</tt>. */
  def constant(c: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = c
  }

  /** Constructs a function that returns <tt>a / b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def div(b: Double): DoubleFunction = mult(1 / b)

  /** Constructs a function that returns <tt>a == b ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def equals(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = if (a == b) 1 else 0
  }

  /** Constructs a function that returns <tt>a > b ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def greater(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = if (a > b) 1 else 0
  }

  /**
   * Constructs a function that returns <tt>math.IEEEremainder(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is
   * fixed.
   */
  def mathIEEEremainder(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = math.IEEEremainder(a, b)
  }

  /**
   * Constructs a function that returns <tt>from<=a && a<=to</tt>. <tt>a</tt> is a variable, <tt>from</tt> and
   * <tt>to</tt> are fixed.
   *
   * Note that DoubleProcedure is generated code and thus looks like an invalid reference unless you can see
   * the generated stuff.
   */
  def isBetween(from: Double, to: Double): DoubleProcedure = new DoubleProcedure() {
    override def apply(a: Double): Boolean = from <= a && a <= to
  }

  /** Constructs a function that returns <tt>a == b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def isEqual(b: Double): DoubleProcedure = new DoubleProcedure() {
    override def apply(a: Double): Boolean = a == b
  }

  /** Constructs a function that returns <tt>a > b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def isGreater(b: Double): DoubleProcedure = new DoubleProcedure() {
    override def apply(a: Double): Boolean = a > b
  }

  /** Constructs a function that returns <tt>a < b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def isLess(b: Double): DoubleProcedure = new DoubleProcedure() {
    override def apply(a: Double): Boolean = a < b
  }

  /** Constructs a function that returns <tt>a < b ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def less(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = if (a < b) 1 else 0
  }

  /**
   * Constructs a function that returns <tt><tt>math.log(a) / math.log(b)</tt></tt>. <tt>a</tt> is a variable,
   * <tt>b</tt> is fixed.
   */
  def lg(b: Double): DoubleFunction = new DoubleFunction() {
    private val logInv = 1 / math.log(b) // cached for speed
    override def apply(a: Double): Double = math.log(a) * logInv
  }

  /** Constructs a function that returns <tt>math.max(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def max(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = math.max(a, b)
  }

  /** Constructs a function that returns <tt>math.min(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def min(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = math.min(a, b)
  }

  /** Constructs a function that returns <tt>a - b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def minus(b: Double): DoubleFunction = plus(-b)

  /**
   * Constructs a function that returns <tt>a - b*constant</tt>. <tt>a</tt> and <tt>b</tt> are variables,
   * <tt>constant</tt> is fixed.
   */
  def minusMult(constant: Double): DoubleDoubleFunction = plusMult(-constant)

  /** Constructs a function that returns <tt>a % b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def mod(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = a % b
  }

  /** Constructs a function that returns <tt>a * b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def mult(b: Double): DoubleFunction = Mult.mult(b)

  /** Constructs a function that returns <tt>a + b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def plus(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = a + b
  }

  /**
   * Constructs a function that returns <tt>a + b*constant</tt>. <tt>a</tt> and <tt>b</tt> are variables,
   * <tt>constant</tt> is fixed.
   */
  def plusMult(constant: Double): DoubleDoubleFunction = new PlusMult(constant)

  /** Constructs a function that returns <tt>math.pow(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  def pow(b: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = if (b == 2) a * a else math.pow(a, b)
  }

  /**
   * Constructs a function that returns a new uniform random number in the open unit interval {@code (0.0,1.0)}
   * (excluding 0.0 and 1.0). Currently the engine is {@link MersenneTwister} and is
   * seeded with the current time. <p> Note that any random engine derived from {@link
   * org.apache.mahout.math.jet.random.engine.RandomEngine} and any random distribution derived from {@link
   * org.apache.mahout.math.jet.random.AbstractDistribution} are function objects, because they implement the proper
   * interfaces. Thus, if you are not happy with the default, just pass your favourite random generator to function
   * evaluating methods.
   */
  def random(): DoubleFunction = new DoubleFunction() {
    private val r = new MersenneTwister(new Date())
    override def apply(a: Double): Double = r.apply(a)
  }

  /**
   * Constructs a function that returns the number rounded to the given precision;
   * <tt>math.rint(a/precision)*precision</tt>. Examples:
   * <pre>
   * precision = 0.01 rounds 0.012 --> 0.01, 0.018 --> 0.02
   * precision = 10   rounds 123   --> 120 , 127   --> 130
   * </pre>
   */
  def round(precision: Double): DoubleFunction = new DoubleFunction() {
    override def apply(a: Double): Double = math.rint(a / precision) * precision
  }

  /**
   * Constructs a function that returns <tt>function.apply(b,a)</tt>, i.e. applies the function with the first operand
   * as second operand and the second operand as first operand.
   *
   * @param function a function taking operands in the form <tt>function.apply(a,b)</tt>.
   * @return the binary function <tt>function(b,a)</tt>.
   */
  def swapArgs(function: DoubleDoubleFunction): DoubleDoubleFunction = new DoubleDoubleFunction() {
    override def apply(a: Double, b: Double): Double = function.apply(b, a)
  }

  def minusAbsPow(exponent: Double): DoubleDoubleFunction = new DoubleDoubleFunction() {
    override def apply(x: Double, y: Double): Double = math.pow(math.abs(x - y), exponent)

    /**
     * |x - 0|^p = |x|^p != x unless x > 0 and p = 1
     * @return true iff f(x, 0) = x for any x
     */
    override def isLikeRightPlus: Boolean = false

    /**
     * |0 - y|^p = |y|^p
     * @return true iff f(0, y) = 0 for any y
     */
    override def isLikeLeftMult: Boolean = false

    /**
     * |x - 0|^p = |x|^p
     * @return true iff f(x, 0) = 0 for any x
     */
    override def isLikeRightMult: Boolean = false

    /**
     * |x - y|^p = |y - x|^p
     * @return true iff f(x, y) = f(y, x) for any x, y
     */
    override def isCommutative: Boolean = true

    /**
     * |x - |y - z|^p|^p != ||x - y|^p - z|^p
     * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
     */
    override def isAssociative: Boolean = false
  }
}