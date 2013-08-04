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

import spark.mllib.math.function.DoubleDoubleFunction
import spark.mllib.math.collection.set.OpenIntHashSet

import java.util.Iterator

/**
 * Abstract class encapsulating different algorithms that perform the Vector operations aggregate().
 * x.aggregte(y, fa, fc), for x and y Vectors and fa, fc DoubleDouble functions:
 * - applies the function fc to every element in x and y, fc(xi, yi)
 * - constructs a result iteratively, r0 = fc(x0, y0), ri = fc(r_{i-1}, fc(xi, yi)).
 * This works essentially like a map/reduce functional combo.
 *
 * The names of variables, methods and classes used here follow the following conventions:
 * The vector being assigned to (the left hand side) is called this or x.
 * The right hand side is called that or y.
 * The aggregating (reducing) function to be applied is called fa.
 * The combining (mapping) function to be applied is called fc.
 *
 * The different algorithms take into account the different characteristics of vector classes:
 * - whether the vectors support sequential iteration (isSequential())
 * - what the lookup cost is (getLookupCost())
 * - what the iterator advancement cost is (getIteratorAdvanceCost())
 *
 * The names of the actual classes (they're nested in VectorBinaryAssign) describe the used for assignment.
 * The most important optimization is iterating just through the nonzeros (only possible if f(0, 0) = 0).
 * There are 4 main possibilities:
 * - iterating through the nonzeros of just one vector and looking up the corresponding elements in the other
 * - iterating through the intersection of nonzeros (those indices where both vectors have nonzero values)
 * - iterating through the union of nonzeros (those indices where at least one of the vectors has a nonzero value)
 * - iterating through all the elements in some way (either through both at the same time, both one after the other,
 *   looking up both, looking up just one).
 *
 * The internal details are not important and a particular algorithm should generally not be called explicitly.
 * The best one will be selected through assignBest(), which is itself called through Vector.assign().
 *
 * See https://docs.google.com/document/d/1g1PjUuvjyh2LBdq2_rKLIcUiDbeOORA1sCJiSsz-JVU/edit# for a more detailed
 * explanation.
 */
abstract class VectorBinaryAggregate {

  /**
   * Returns true iff we can use this algorithm to apply fc to x and y component-wise and aggregate the result using fa.
   */
  def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction, fc: DoubleDoubleFunction): Boolean

  /**
   * Estimates the cost of using this algorithm to compute the aggregation. The algorithm is assumed to be valid.
   */
  def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction, fc: DoubleDoubleFunction): Double

  /**
   * Main method that applies fc to x and y component-wise aggregating the results with fa. It returns the result of
   * the aggregation.
   */
  def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction, fc: DoubleDoubleFunction): Double

}

object VectorBinaryAggregate {
  val OPERATIONS = Array(
    new AggregateNonzerosIterateThisLookupThat(),
    new AggregateNonzerosIterateThatLookupThis(),

    new AggregateIterateIntersection(),

    new AggregateIterateUnionSequential(),
    new AggregateIterateUnionRandom(),

    new AggregateAllIterateSequential(),
    new AggregateAllIterateThisLookupThat(),
    new AggregateAllIterateThatLookupThis(),
    new AggregateAllLoop()
  )

  /**
   * The best operation is the least expensive valid one.
   */
  def getBestOperation(x: Vector, y: Vector, fa: DoubleDoubleFunction,
      fc: DoubleDoubleFunction): VectorBinaryAggregate = {
    var bestOperationIndex = -1
    var bestCost = Double.PositiveInfinity
    for (i <- 0 until OPERATIONS.length) {
      if (OPERATIONS(i).isValid(x, y, fa, fc)) {
        val cost = OPERATIONS(i).estimateCost(x, y, fa, fc)
        if (cost < bestCost) {
          bestCost = cost
          bestOperationIndex = i
        }
      }
    }
    OPERATIONS(bestOperationIndex)
  }

  /**
   * This is the method that should be used when aggregating. It selects the best algorithm and applies it.
   */
  def aggregateBest(x: Vector, y: Vector, fa: DoubleDoubleFunction,
      fc: DoubleDoubleFunction): Double = {
    getBestOperation(x, y, fa, fc).aggregate(x, y, fa, fc)
  }

  class AggregateNonzerosIterateThisLookupThat extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      fa.isLikeRightPlus && (fa.isAssociativeAndCommutative || x.isSequentialAccess) &&
      fc.isLikeLeftMult
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      x.getNumNondefaultElements * x.getIteratorAdvanceCost * y.getLookupCost
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val xi = x.nonZeroes().iterator
      if (!xi.hasNext) {
        return 0
      }
      var xe = xi.next()
      var result = fc(xe.get(), y(xe.index))
      while (xi.hasNext) {
        xe = xi.next()
        result = fa(result, fc(xe.get(), y(xe.index)))
      }
      result
    }
  }

  class AggregateNonzerosIterateThatLookupThis extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      fa.isLikeRightPlus && (fa.isAssociativeAndCommutative || y.isSequentialAccess) &&
      fc.isLikeRightMult
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      y.getNumNondefaultElements * y.getIteratorAdvanceCost * x.getLookupCost * x.getLookupCost
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val yi = y.nonZeroes().iterator
      if (!yi.hasNext) {
        return 0
      }
      var ye = yi.next()
      var result = fc(x(ye.index), ye.get())
      while (yi.hasNext) {
        ye = yi.next()
        result = fa(result, fc(x(ye.index), ye.get()))
      }
      result
    }
  }

  class AggregateIterateIntersection extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      fa.isLikeRightPlus && fc.isLikeMult && x.isSequentialAccess && y.isSequentialAccess
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      math.min(x.getNumNondefaultElements * x.getIteratorAdvanceCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost)
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val xi = x.nonZeroes().iterator
      val yi = y.nonZeroes().iterator
      var xe: Vector.Element = null
      var ye: Vector.Element = null
      var advanceThis = true
      var advanceThat = true
      var validResult = false
      var result = 0.0
      while (true) {
        if (advanceThis) {
          if (xi.hasNext) {
            xe = xi.next()
          } else {
            return result
          }
        }
        if (advanceThat) {
          if (yi.hasNext) {
            ye = yi.next()
          } else {
            return result
          }
        }
        if (xe.index == ye.index) {
          val thisResult = fc(xe.get(), ye.get())
          if (validResult) {
            result = fa(result, thisResult)
          } else {
            result = thisResult
            validResult = true
          }
          advanceThis = true
          advanceThat = true
        } else {
          if (xe.index < ye.index) { // f(x, 0) = 0
            advanceThis = true
            advanceThat = false
          } else { // f(0, y) = 0
            advanceThis = false
            advanceThat = true
          }
        }
      }
      result
    }
  }

  class AggregateIterateUnionSequential extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      fa.isLikeRightPlus && !fc.isDensifying && x.isSequentialAccess && y.isSequentialAccess
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      math.max(x.getNumNondefaultElements * x.getIteratorAdvanceCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost)
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val xi = x.nonZeroes().iterator
      val yi = y.nonZeroes().iterator
      var xe: Vector.Element = null
      var ye: Vector.Element = null
      var advanceThis = true
      var advanceThat = true
      var validResult = false
      var result = 0.0
      while (true) {
        if (advanceThis) {
          if (xi.hasNext) {
            xe = xi.next()
          } else {
            xe = null
          }
        }
        if (advanceThat) {
          if (yi.hasNext) {
            ye = yi.next()
          } else {
            ye = null
          }
        }
        var thisResult = 0.0
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index == ye.index) {
            thisResult = fc(xe.get(), ye.get());
            advanceThis = true
            advanceThat = true
          } else {
            if (xe.index < ye.index) { // f(x, 0)
              thisResult = fc(xe.get(), 0)
              advanceThis = true
              advanceThat = false
            } else {
              thisResult = fc(0, ye.get())
              advanceThis = false
              advanceThat = true
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          thisResult = fc(xe.get(), 0)
          advanceThis = true
          advanceThat = false
        } else if (ye != null) { // just the second one has nonzeros
          thisResult = fc(0, ye.get())
          advanceThis = false
          advanceThat = true
        } else { // we're done, both are empty
          return result
        }
        if (validResult) {
          result = fa(result, thisResult);
        } else {
          result = thisResult
          validResult =  true
        }
      }
      result
    }
  }

  class AggregateIterateUnionRandom extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      fa.isLikeRightPlus && !fc.isDensifying && (fa.isAssociativeAndCommutative ||
          (x.isSequentialAccess && y.isSequentialAccess))
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      math.max(x.getNumNondefaultElements * x.getIteratorAdvanceCost * y.getLookupCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost * x.getLookupCost)
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val visited = new OpenIntHashSet()
      val xi = x.nonZeroes().iterator
      var validResult = false
      var result = 0.0
      var thisResult = 0.0
      while (xi.hasNext) {
        var xe = xi.next()
        thisResult = fc(xe.get(), y(xe.index))
        if (validResult) {
          result = fa(result, thisResult)
        } else {
          result = thisResult
          validResult = true
        }
        visited.add(xe.index)
      }
      val yi = y.nonZeroes().iterator
      while (yi.hasNext) {
        var ye = yi.next()
        if (!visited.contains(ye.index)) {
          thisResult = fc(x(ye.index), ye.get())
          if (validResult) {
            result = fa(result, thisResult)
          } else {
            result = thisResult
            validResult = true
          }
        }
      }
      result
    }
  }

  class AggregateAllIterateSequential extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      x.isSequentialAccess && y.isSequentialAccess && !x.isDense && !y.isDense
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      math.max(x.dimension * x.getIteratorAdvanceCost, y.dimension * y.getIteratorAdvanceCost)
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val xi = x.all().iterator
      val yi = y.all().iterator
      var validResult = false
      var result = 0.0
      while (xi.hasNext && yi.hasNext) {
        val xe = xi.next()
        val thisResult = fc(xe.get(), yi.next().get())
        if (validResult) {
          result = fa(result, thisResult)
        } else {
          result = thisResult
          validResult = true
        }
      }
      result
    }
  }

  class AggregateAllIterateThisLookupThat extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      (fa.isAssociativeAndCommutative || x.isSequentialAccess) && !x.isDense
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      x.dimension * x.getIteratorAdvanceCost * y.getLookupCost
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val xi = x.all().iterator
      var validResult = false
      var result = 0.0
      while (xi.hasNext) {
        val xe = xi.next();
        val thisResult = fc(xe.get(), y(xe.index))
        if (validResult) {
          result = fa(result, thisResult)
        } else {
          result = thisResult
          validResult = true
        }
      }
      result
    }
  }

  class AggregateAllIterateThatLookupThis extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      (fa.isAssociativeAndCommutative || y.isSequentialAccess) && !y.isDense
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      y.dimension * y.getIteratorAdvanceCost * x.getLookupCost
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      val yi = y.all().iterator
      var validResult = false
      var result = 0.0
      while (yi.hasNext) {
        val ye = yi.next()
        val thisResult = fc(x(ye.index), ye.get())
        if (validResult) {
          result = fa(result, thisResult)
        } else {
          result = thisResult
          validResult = true
        }
      }
      result
    }
  }

  class AggregateAllLoop extends VectorBinaryAggregate {

    override def isValid(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Boolean = {
      return true;
    }

    override def estimateCost(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      return x.dimension * x.getLookupCost * y.getLookupCost
    }

    override def aggregate(x: Vector, y: Vector, fa: DoubleDoubleFunction,
        fc: DoubleDoubleFunction): Double = {
      var result = fc(x(0), y(0))
      for (i <- 1 until x.dimension) {
        result = fa(result, fc(x(i), y(i)))
      }
      result
    }
  }
}