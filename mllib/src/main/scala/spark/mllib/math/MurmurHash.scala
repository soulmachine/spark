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

package spark.mllib.math

/** MurmurHash3 algorithm. */
object MurmurHash {
  
  /**
   * Hashes bytes in an array.
   * @param data The bytes to hash.
   * @param seed The seed for the hash.
   * @return The 32 bit hash of the bytes in question.
   */
  def hash(data: Array[Byte], seed: Int): Int = {
    util.MurmurHash.startHash(seed)
    util.MurmurHash.arrayHash(data)
  }

  /**
   * Hashes bytes in part of an array.
   * @param data    The data to hash.
   * @param offset  Where to start munging.
   * @param length  How many bytes to process.
   * @param seed    The seed to start with.
   * @return        The 32-bit hash of the data in question.
   */
  def hash(data: Array[Byte], offset: Int, length: Int, seed: Int): Int = {
    hash(data.slice(offset, offset + length), seed)
  }
}