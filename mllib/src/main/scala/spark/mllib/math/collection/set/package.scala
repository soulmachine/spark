package spark.mllib.math.collection

package object set {
  type OpenIntHashSet = OpenNumericHashSet[Int]
  type OpenLongHashSet = OpenNumericHashSet[Long]
  type OpenFloatHashSet = OpenNumericHashSet[Float]
  type OpenDoubleHashSet = OpenNumericHashSet[Double]
}