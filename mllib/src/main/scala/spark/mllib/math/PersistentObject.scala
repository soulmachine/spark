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

/**
 * This empty class is the common root for all persistent capable classes.
 * If this class inherits from <tt>java.lang.Object</tt> then all subclasses are serializable with
 * the standard Java serialization mechanism.
 * If this class inherits from <tt>com.objy.db.app.ooObj</tt> then all subclasses are
 * <i>additionally</i> serializable with the Objectivity ODBMS persistance mechanism.
 * Thus, by modifying the inheritance of this class the entire tree of subclasses can
 * be switched to Objectivity compatibility (and back) with minimum effort.
 */
abstract class PersistentObject extends Serializable with Cloneable {

  /**
   * Returns a copy of the receiver. This default implementation does not nothing except making the otherwise
   * <tt>protected</tt> clone method <tt>public</tt>.
   *
   * @return a copy of the receiver.
   */
  override def clone(): AnyRef = {
    try {
      return super.clone()
    } catch {
      case exc: CloneNotSupportedException => throw new InternalError() //should never happen since we are cloneable
    }
  }
}
