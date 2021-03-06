@echo off

rem
rem Licensed to the Apache Software Foundation (ASF) under one or more
rem contributor license agreements.  See the NOTICE file distributed with
rem this work for additional information regarding copyright ownership.
rem The ASF licenses this file to You under the Apache License, Version 2.0
rem (the "License"); you may not use this file except in compliance with
rem the License.  You may obtain a copy of the License at
rem
rem    http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem

rem This script computes Spark's classpath and prints it to stdout; it's used by both the "run"
rem script and the ExecutorRunner in standalone cluster mode.

set SCALA_VERSION=2.9.3

rem Figure out where the Spark framework is installed
set FWDIR=%~dp0..\

rem Load environment variables from conf\spark-env.cmd, if it exists
if exist "%FWDIR%conf\spark-env.cmd" call "%FWDIR%conf\spark-env.cmd"

set CORE_DIR=%FWDIR%core
set REPL_DIR=%FWDIR%repl
set EXAMPLES_DIR=%FWDIR%examples
set BAGEL_DIR=%FWDIR%bagel
set MLLIB_DIR=%FWDIR%mllib
set TOOLS_DIR=%FWDIR%tools
set STREAMING_DIR=%FWDIR%streaming
set PYSPARK_DIR=%FWDIR%python

rem Build up classpath
set CLASSPATH=%SPARK_CLASSPATH%;%MESOS_CLASSPATH%;%FWDIR%conf;%CORE_DIR%\target\scala-%SCALA_VERSION%\classes
set CLASSPATH=%CLASSPATH%;%CORE_DIR%\target\scala-%SCALA_VERSION%\test-classes;%CORE_DIR%\src\main\resources
set CLASSPATH=%CLASSPATH%;%STREAMING_DIR%\target\scala-%SCALA_VERSION%\classes;%STREAMING_DIR%\target\scala-%SCALA_VERSION%\test-classes
set CLASSPATH=%CLASSPATH%;%STREAMING_DIR%\lib\org\apache\kafka\kafka\0.7.2-spark\*
set CLASSPATH=%CLASSPATH%;%REPL_DIR%\target\scala-%SCALA_VERSION%\classes;%EXAMPLES_DIR%\target\scala-%SCALA_VERSION%\classes
set CLASSPATH=%CLASSPATH%;%FWDIR%lib_managed\jars\*
set CLASSPATH=%CLASSPATH%;%FWDIR%lib_managed\bundles\*
set CLASSPATH=%CLASSPATH%;%FWDIR%repl\lib\*
set CLASSPATH=%CLASSPATH%;%FWDIR%python\lib\*
set CLASSPATH=%CLASSPATH%;%BAGEL_DIR%\target\scala-%SCALA_VERSION%\classes
set CLASSPATH=%CLASSPATH%;%MLLIB_DIR%\target\scala-%SCALA_VERSION%\classes
set CLASSPATH=%CLASSPATH%;%TOOLS_DIR%\target\scala-%SCALA_VERSION%\classes

rem Add hadoop conf dir - else FileSystem.*, etc fail
rem Note, this assumes that there is either a HADOOP_CONF_DIR or YARN_CONF_DIR which hosts
rem the configurtion files.
if "x%HADOOP_CONF_DIR%"=="x" goto no_hadoop_conf_dir
  set CLASSPATH=%CLASSPATH%;%HADOOP_CONF_DIR%
:no_hadoop_conf_dir

if "x%YARN_CONF_DIR%"=="x" goto no_yarn_conf_dir
  set CLASSPATH=%CLASSPATH%;%YARN_CONF_DIR%
:no_yarn_conf_dir

rem Add Scala standard library
set CLASSPATH=%CLASSPATH%;%SCALA_HOME%\lib\scala-library.jar;%SCALA_HOME%\lib\scala-compiler.jar;%SCALA_HOME%\lib\jline.jar

rem A bit of a hack to allow calling this script within run2.cmd without seeing output
if "%DONT_PRINT_CLASSPATH%"=="1" goto exit

echo %CLASSPATH%

:exit
