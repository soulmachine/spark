package spark.mllib.math.collection

package object list {
  type IntArrayList = NumericArrayList[Int]
  type LongArrayList = NumericArrayList[Long]
  type FloatArrayList = NumericArrayList[Float]
  type DoubleArrayList = NumericArrayList[Double]
}