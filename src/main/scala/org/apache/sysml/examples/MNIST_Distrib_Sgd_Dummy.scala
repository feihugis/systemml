package org.apache.sysml.examples

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.scripts.nn.examples.{Mnist_lenet_distrib_sgd, Mnist_lenet_distrib_sgd_optimize}

/**
  * Created by Fei Hu on 8/3/17.
  */
object MNIST_Distrib_Sgd_Dummy {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SystemML-MNIST-Distrib").setMaster("local[4]").set("spark.driver.memory", "2g")
    /*conf.set("spark.driver.memory", "2g")
    conf.set("spark.executor.memory", "2g")*/

    /*val memSize = 20 * 1024 * 1024 * 1024L
    conf.set("spark.testing.memory", memSize.toString)*/

    val sc = new SparkContext(conf)
    //sc.setLogLevel("TRACE")

    val ml = new MLContext(sc)

    ml.setStatistics(true)
    ml.setStatisticsMaxHeavyHitters(10000)
    ml.setConfigProperty("systemml.stats.finegrained", "true")
    ml.setExplain(true)
    ml.setExplainLevel(MLContext.ExplainLevel.RECOMPILE_RUNTIME)

    org.apache.sysml.api.DMLScript.rtplatform = org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM.HYBRID_SPARK

    val clf = new Mnist_lenet_distrib_sgd_optimize

    val N = 32
    val Nval = 16
    val Ntest = 16
    val C = 3
    val Hin = 112
    val Win = 112
    val K = 10
    val batchSize = 2
    val paralellBatches = 2
    val epochs = 1

    val dummy = clf.generate_dummy_data(N, C, Hin, Win, K)
    val dummyVal = clf.generate_dummy_data(Nval, C, Hin, Win, K)
    val dummyTest = clf.generate_dummy_data(Ntest, C, Hin, Win, K)

    //dummy.X.toMatrixObject.setUpdateType(UpdateType.INPLACE)
    //dummy.X.toMatrixObject.setDirty(false)
    //dummy.Y.toMatrixObject.setDirty(false)

    println(dummy.X.toBinaryBlocks.count())


    val params = clf.train(dummy.X, dummy.Y, dummyVal.X, dummyVal.Y, C, Hin, Win, batchSize, paralellBatches, epochs)
    println(params.toString)

    //val probs = clf.predict(dummyTest.X, C, Hin, Win, params.W1, params.b1, params.W2, params.b2, params.W3, params.b3, params.W4, params.b4)
    //val perf = clf.eval(probs, dummyTest.Y)
    //println(perf)

  }

}
