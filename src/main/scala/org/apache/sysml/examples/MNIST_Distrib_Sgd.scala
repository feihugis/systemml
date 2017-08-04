package org.apache.sysml.examples

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.mlcontext.ScriptFactory.dml
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.parser.Expression.ValueType
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext
import org.apache.sysml.runtime.io.{MatrixReaderFactory, MatrixWriter, MatrixWriterFactory}
import org.apache.sysml.runtime.matrix.data.InputInfo
import org.apache.sysml.runtime.util.MapReduceTool
import org.apache.sysml.scripts.nn.examples.Mnist_lenet_distrib_sgd

object MNIST_Distrib_Sgd {

  def writeMatrix(matrix: Matrix, file: String): Unit = {
    matrix.toMatrixObject.exportData(file, null)
  }

  def readMatrix(file: String, ml: MLContext): Matrix = {
    //val matrixObj = new MatrixObject(ValueType.DOUBLE, file)
    //new Matrix(matrixObj, ec)
    val s =
      """
        x = read("scratch_space/X_input")
      """

    val scr = dml(s).out("x")
    val res = ml.execute(scr)
    res.getMatrix("x")
  }

  def createMNISTDummyData(X_file: String, Y_file: String, X_val_file: String, Y_val_file: String): Unit = {
    val N = 3200
    val Nval = 32
    val Ntest = 32
    val C = 3
    val Hin = 112
    val Win = 112
    val K = 10

    val clf = new Mnist_lenet_distrib_sgd()
    val dummy = clf.generate_dummy_data(N, C, Hin, Win, K)
    writeMatrix(dummy.X, X_file)
    writeMatrix(dummy.Y, Y_file)

    val dummyVal = clf.generate_dummy_data(Nval, C, Hin, Win, K)
    writeMatrix(dummyVal.X, X_val_file)
    writeMatrix(dummyVal.Y, Y_val_file)
  }

  def setSparkConf(): SparkConf = {
    val conf = new SparkConf().setAppName("SystemML-MNIST-Distrib").setMaster("local[4]").set("spark.driver.memory", "2g")
    /*conf.set("spark.driver.memory", "2g")
    conf.set("spark.executor.memory", "2g")*/

    /*val memSize = 20 * 1024 * 1024 * 1024L
    conf.set("spark.testing.memory", memSize.toString)*/
    conf
  }

  def configMLContext(ml: MLContext): Unit = {
    ml.setStatistics(true)
    ml.setStatisticsMaxHeavyHitters(10000)
    ml.setConfigProperty("systemml.stats.finegrained", "true")
    ml.setExplain(true)
    ml.setExplainLevel(MLContext.ExplainLevel.RECOMPILE_RUNTIME)
  }

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {
    val conf = setSparkConf
    val sc = new SparkContext(conf)
    //sc.setLogLevel("TRACE")

    val ml = new MLContext(sc)
    configMLContext(ml)

    org.apache.sysml.api.DMLScript.rtplatform = org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM.HYBRID_SPARK

    val clf = new Mnist_lenet_distrib_sgd()

    val N = 3200
    val Nval = 32
    val Ntest = 32
    val C = 3
    val Hin = 112
    val Win = 112
    val K = 10
    val batchSize = 2
    val paralellBatches = 2
    val epochs = 1

    val X_file = "scratch_space/X_input"
    val Y_file = "scratch_space/Y_input"
    val X_val_file = "scratch_space/X_val_input"
    val Y_val_file = "scratch_space/Y_val_input"

    val X = readMatrix(X_file, ml)
    val Y = readMatrix(Y_file, ml)
    val X_val = readMatrix(X_val_file, ml)
    val Y_val = readMatrix(Y_val_file, ml)

    val params = clf.train(X, Y, X_val, Y_val, C, Hin, Win, batchSize, paralellBatches, epochs)
    println(params.toString)
  }

}
