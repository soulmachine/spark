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

package spark.mllib.recommendation

import spark.RDD
import spark.SparkContext._
import spark.mllib.math.vector.{Vector, DenseVector}

import org.jblas._

class MatrixFactorizationModel(
    val rank: Int,
    val userFeatures: RDD[(Int, Vector)],
    val productFeatures: RDD[(Int, Vector)])
  extends Serializable
{
  /** Predict the rating of one user for one product. */
  def predict(user: Int, product: Int): Double = {
    val userVector = userFeatures.lookup(user).head
    val productVector = productFeatures.lookup(product).head
    userVector * productVector
  }

  // TODO: Figure out what good bulk prediction methods would look like.
  // Probably want a way to get the top users for a product or vice-versa.
}
