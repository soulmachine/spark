/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License") you may not use this file except in compliance with
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

import java.util.Iterator

import spark.mllib.math.vector.Vector.Element
import spark.mllib.math.function.DoubleDoubleFunction
import spark.mllib.math.collection.set.OpenIntHashSet
import spark.mllib.math.collection.map.OrderedIntDoubleMapping

/**
 * Abstract class encapsulating different algorithms that perform the Vector operations assign().
 * x.assign(y, f), for x and y Vectors and f a DoubleDouble function:
 * - applies the function f to every element in x and y, f(xi, yi)
 * - assigns xi = f(xi, yi) for all indices i
 *
 * The names of variables, methods and classes used here follow the following conventions:
 * The vector being assigned to (the left hand side) is called this or x.
 * The right hand side is called that or y.
 * The function to be applied is called f.
 *
 * The different algorithms take into account the different characteristics of vector classes:
 * - whether the vectors support sequential iteration (isSequential())
 * - whether the vectors support constant-time additions (isAddConstantTime())
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
 * Then, there are two additional sub-possibilities:
 * - if a new value can be added to x in constant time (isAddConstantTime()), the *Inplace updates are used
 * - otherwise (really just for SequentialAccessSparseVectors right now), the *Merge updates are used, where
 *   a sorted list of (index, value) pairs is merged into the vector at the end.
 *
 * The internal details are not important and a particular algorithm should generally not be called explicitly.
 * The best one will be selected through assignBest(), which is itself called through Vector.assign().
 *
 * See https://docs.google.com/document/d/1g1PjUuvjyh2LBdq2_rKLIcUiDbeOORA1sCJiSsz-JVU/edit# for a more detailed
 * explanation.
 */
abstract class VectorBinaryAssign {

  /**
   * Returns true iff we can use this algorithm to apply f to x and y component-wise and assign the result to x.
   */
  def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean

  /**
   * Estimates the cost of using this algorithm to compute the assignment. The algorithm is assumed to be valid.
   */
  def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double

  /**
   * Main method that applies f to x and y component-wise assigning the results to x. It returns the modified vector,
   * x.
   */
  def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector

}

object VectorBinaryAssign {
  val OPERATIONS = Array(
    new AssignNonzerosIterateThisLookupThat(),
    new AssignNonzerosIterateThatLookupThisMergeUpdates(),
    new AssignNonzerosIterateThatLookupThisInplaceUpdates(),

    new AssignIterateIntersection(),

    new AssignIterateUnionSequentialMergeUpdates(),
    new AssignIterateUnionSequentialInplaceUpdates(),
    new AssignIterateUnionRandomMergeUpdates(),
    new AssignIterateUnionRandomInplaceUpdates(),

    new AssignAllIterateSequentialMergeUpdates(),
    new AssignAllIterateSequentialInplaceUpdates(),
    new AssignAllIterateThisLookupThatMergeUpdates(),
    new AssignAllIterateThisLookupThatInplaceUpdates(),
    new AssignAllIterateThatLookupThisMergeUpdates(),
    new AssignAllIterateThatLookupThisInplaceUpdates(),
    new AssignAllLoopMergeUpdates(),
    new AssignAllLoopInplaceUpdates()
  )

  /**
   * The best operation is the least expensive valid one.
   */
  def  getBestOperation(x: Vector, y: Vector, f: DoubleDoubleFunction):VectorBinaryAssign = {
    var bestOperationIndex = -1
    var bestCost = Double.PositiveInfinity
    for (i <- 0 until OPERATIONS.length) {
      if (OPERATIONS(i).isValid(x, y, f)) {
        val cost = OPERATIONS(i).estimateCost(x, y, f)
        if (cost < bestCost) {
          bestCost = cost
          bestOperationIndex = i
        }
      }
    }
    OPERATIONS(bestOperationIndex)
  }

  /**
   * This is the method that should be used when assigning. It selects the best algorithm and applies it.
   * Note that it does NOT invalidate the cached length of the Vector and should only be used through the wrapprs
   * in AbstractVector.
   */
  def assignBest(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
    getBestOperation(x, y, f).assign(x, y, f)
  }

  /**
   * If f(0, y) = 0, the zeros in x don't matter and we can simply iterate through the nonzeros of x.
   * To get the corresponding element of y, we perform a lookup.
   * There are no *Merge or *Inplace versions because in this case x cannot become more dense because of f, meaning
   * all changes will occur at indices whose values are already nonzero.
   */
  class AssignNonzerosIterateThisLookupThat extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      f.isLikeLeftMult
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      x.getNumNondefaultElements * x.getIteratorAdvanceCost * y.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
      for (xe <- x.nonZeroes()) {
        xe.set(f(xe.get(), y(xe.index)))
      }
      x
    }
  }

  /**
   * If f(x, 0) = x, the zeros in y don't matter and we can simply iterate through the nonzeros of y.
   * We get the corresponding element of x through a lookup and update x inplace.
   */
  class AssignNonzerosIterateThatLookupThisInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      f.isLikeRightPlus
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      y.getNumNondefaultElements * y.getIteratorAdvanceCost * x.getLookupCost * x.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      for (ye <- y.nonZeroes()) {
        x(ye.index) = f(x(ye.index), ye.get())
      }
      x
    }
  }

  /**
   * If f(x, 0) = x, the zeros in y don't matter and we can simply iterate through the nonzeros of y.
   * We get the corresponding element of x through a lookup and update x by merging.
   */
  class AssignNonzerosIterateThatLookupThisMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      f.isLikeRightPlus && y.isSequentialAccess && !x.isAddConstantTime
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      y.getNumNondefaultElements * y.getIteratorAdvanceCost * y.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val updates = new OrderedIntDoubleMapping(false)
      for (ye <- y.nonZeroes()) {
        updates(ye.index) = f(x(ye.index), ye.get())
      }
      x.mergeUpdates(updates)
      x
    }
  }

  /**
   * If f(x, 0) = x and f(0, y) = 0 the zeros in x and y don't matter and we can iterate through the nonzeros
   * in both x and y.
   * This is only possible if both x and y support sequential access.
   */
  class AssignIterateIntersection extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      f.isLikeLeftMult && f.isLikeRightPlus && x.isSequentialAccess && y.isSequentialAccess
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.min(x.getNumNondefaultElements * x.getIteratorAdvanceCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost)
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val xi = x.nonZeroes().iterator
      val yi = y.nonZeroes().iterator
      var xe: Vector.Element = null
      var ye: Vector.Element = null
      var advanceThis = true
      var advanceThat = true
      while (true) {
        if (advanceThis) {
          if (xi.hasNext) {
            xe = xi.next()
          } else {
            return x
          }
        }
        if (advanceThat) {
          if (yi.hasNext) {
            ye = yi.next()
          } else {
            return x
          }
        }
        if (xe.index == ye.index) {
          xe.set(f(xe.get(), ye.get()))
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
      x
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case we iterate through them in parallel and update x by merging. Because we're iterating through
   * both vectors at the same time, x and y need to support sequential access.
   */
  class AssignIterateUnionSequentialMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !f.isDensifying && x.isSequentialAccess && y.isSequentialAccess && !x.isAddConstantTime
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.max(x.getNumNondefaultElements * x.getIteratorAdvanceCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost)
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
      val xi = x.nonZeroes().iterator
      val yi = y.nonZeroes().iterator
      var xe: Vector.Element = null
      var ye: Vector.Element = null
      var advanceThis = true
      var advanceThat = true
      val updates = new OrderedIntDoubleMapping(false)
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
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index == ye.index) {
            xe.set(f(xe.get(), ye.get()))
            advanceThis = true
            advanceThat = true
          } else {
            if (xe.index < ye.index) { // f(x, 0)
              xe.set(f(xe.get(), 0))
              advanceThis = true
              advanceThat = false
            } else {
              updates(ye.index) = f(0, ye.get())
              advanceThis = false
              advanceThat = true
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          xe.set(f(xe.get(), 0))
          advanceThis = true
          advanceThat = false
        } else if (ye != null) { // just the second one has nonzeros
          updates(ye.index) = f(0, ye.get())
          advanceThis = false
          advanceThat = true
        } else { // we're done, both are empty
          x.mergeUpdates(updates)
          return x
        }
      }
      x.mergeUpdates(updates)
      x
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case we iterate through them in parallel and update x inplace. Because we're iterating through
   * both vectors at the same time, x and y need to support sequential access.
   */
  class AssignIterateUnionSequentialInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !f.isDensifying && x.isSequentialAccess && y.isSequentialAccess && x.isAddConstantTime
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.max(x.getNumNondefaultElements * x.getIteratorAdvanceCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost)
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val xi = x.nonZeroes().iterator
      val yi = y.nonZeroes().iterator
      var xe: Vector.Element = null
      var ye: Vector.Element = null
      var advanceThis = true
      var advanceThat = true
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
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index == ye.index) {
            xe.set(f(xe.get(), ye.get()))
            advanceThis = true
            advanceThat = true
          } else {
            if (xe.index < ye.index) { // f(x, 0)
              xe.set(f(xe.get(), 0))
              advanceThis = true
              advanceThat = false
            } else {
              x(ye.index) = f(0, ye.get())
              advanceThis = false
              advanceThat = true
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          xe.set(f(xe.get(), 0))
          advanceThis = true
          advanceThat = false
        } else if (ye != null) { // just the second one has nonzeros
          x(ye.index) = f(0, ye.get())
          advanceThis = false
          advanceThat = true
        } else { // we're done, both are empty
          return x
        }
      }
      x
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case, we iterate through the nozeros of x and y alternatively (this works even when one of them
   * doesn't support sequential access). Since we're merging the results into x, when iterating through y, the
   * order of iteration matters and y must support sequential access.
   */
  class AssignIterateUnionRandomMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !f.isDensifying && !x.isAddConstantTime && y.isSequentialAccess
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.max(x.getNumNondefaultElements * x.getIteratorAdvanceCost * y.getLookupCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost * x.getLookupCost)
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val visited = new OpenIntHashSet()
      for (xe <- x.nonZeroes()) {
        xe.set(f(xe.get(), y(xe.index)))
        visited.add(xe.index)
      }
      val updates = new OrderedIntDoubleMapping(false)
      for (ye <- y.nonZeroes()) {
        if (!visited.contains(ye.index)) {
          updates(ye.index) = f(x(ye.index), ye.get())
        }
      }
      x.mergeUpdates(updates)
      x
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case, we iterate through the nozeros of x and y alternatively (this works even when one of them
   * doesn't support sequential access). Because updates to x are inplace, neither x, nor y need to support
   * sequential access.
   */
  class AssignIterateUnionRandomInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !f.isDensifying && x.isAddConstantTime
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.max(x.getNumNondefaultElements * x.getIteratorAdvanceCost * y.getLookupCost,
          y.getNumNondefaultElements * y.getIteratorAdvanceCost * x.getLookupCost)
    }
    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val visited = new OpenIntHashSet()
      for (xe <- x.nonZeroes()) {
        xe.set(f(xe.get(), y(xe.index)))
        visited.add(xe.index)
      }
      for (ye <- y.nonZeroes()) {
        if (!visited.contains(ye.index)) {
          x(ye.index) = f(x(ye.index), ye.get())
        }
      }
      x
    }
  }

  class AssignAllIterateSequentialMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      x.isSequentialAccess && y.isSequentialAccess && !x.isAddConstantTime && !x.isDense && !y.isDense
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.max(x.dimension * x.getIteratorAdvanceCost, y.dimension * y.getIteratorAdvanceCost)
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val xi = x.all().iterator
      val yi = y.all().iterator
      val updates = new OrderedIntDoubleMapping(false)
      while (xi.hasNext && yi.hasNext) {
        val xe = xi.next()
        updates(xe.index) = f(xe.get(), yi.next.get())
      }
      x.mergeUpdates(updates)
      x
    }
  }

  class AssignAllIterateSequentialInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      x.isSequentialAccess && y.isSequentialAccess && x.isAddConstantTime && !x.isDense && !y.isDense
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      math.max(x.dimension * x.getIteratorAdvanceCost, y.dimension * y.getIteratorAdvanceCost)
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val xi = x.all().iterator
      val yi = y.all().iterator
      while (xi.hasNext && yi.hasNext) {
        val xe = xi.next()
        x(xe.index) = f(xe.get(), yi.next().get())
      }
      x
    }
  }

  class AssignAllIterateThisLookupThatMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !x.isAddConstantTime && !x.isDense
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      x.dimension * x.getIteratorAdvanceCost * y.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
      val updates = new OrderedIntDoubleMapping(false)
      for (xe <- x.all()) {
        updates(xe.index) = f(xe.get(), y(xe.index))
      }
      x.mergeUpdates(updates)
      x
    }
  }

  class AssignAllIterateThisLookupThatInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      x.isAddConstantTime && !x.isDense
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      x.dimension * x.getIteratorAdvanceCost * y.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      for (xe <- x.all()) {
        x(xe.index) = f(xe.get(), y(xe.index))
      }
      x
    }
  }

  class AssignAllIterateThatLookupThisMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !x.isAddConstantTime && !y.isDense
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      y.dimension * y.getIteratorAdvanceCost * x.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction):Vector = {
      val updates = new OrderedIntDoubleMapping(false)
      for (ye <- y.all()) {
        updates(ye.index) = f(x(ye.index), ye.get())
      }
      x.mergeUpdates(updates)
      x
    }
  }

  class AssignAllIterateThatLookupThisInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      x.isAddConstantTime && !y.isDense
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      y.dimension * y.getIteratorAdvanceCost * x.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
      for (ye <- y.all()) {
        x(ye.index) = f(x(ye.index), ye.get())
      }
      x
    }
  }

  class AssignAllLoopMergeUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      !x.isAddConstantTime
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      x.dimension * x.getLookupCost * y.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
      val updates = new OrderedIntDoubleMapping(false)
      for (i <- 0 until x.dimension) {
        updates(i) = f(x(i), y(i))
      }
      x.mergeUpdates(updates)
      x
    }
  }

  class AssignAllLoopInplaceUpdates extends VectorBinaryAssign {

    override def isValid(x: Vector, y: Vector, f: DoubleDoubleFunction): Boolean = {
      x.isAddConstantTime
    }

    override def estimateCost(x: Vector, y: Vector, f: DoubleDoubleFunction): Double = {
      x.dimension * x.getLookupCost * y.getLookupCost
    }

    override def assign(x: Vector, y: Vector, f: DoubleDoubleFunction): Vector = {
      for (i <- 0 until x.dimension) {
        x(i) = f(x(i), y(i))
      }
      x
    }
  }
}