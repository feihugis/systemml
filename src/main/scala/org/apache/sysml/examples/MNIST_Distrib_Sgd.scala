package org.apache.sysml.examples

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.scripts.nn.examples.Mnist_lenet_distrib_sgd

object MNIST_Distrib_Sgd {

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SystemML-MNIST-Distrib").setMaster("local[8]")
    /*conf.set("spark.driver.memory", "2g")
    conf.set("spark.executor.memory", "2g")*/

    /*val memSize = 20 * 1024 * 1024 * 1024L
    conf.set("spark.testing.memory", memSize.toString)*/

    val sc = new SparkContext(conf)
    //sc.setLogLevel("OFF")
    val ml = new MLContext(sc)

    //ml.setStatistics(true)
    //ml.setStatisticsMaxHeavyHitters(10000)
    //ml.setConfigProperty("systemml.stats.finegrained", "true")
    //ml.setConfigProperty("systemml.stats.finegrained ", "true")
    //ml.setExplain(true)
    //ml.setExplainLevel(MLContext.ExplainLevel.RECOMPILE_RUNTIME)

    org.apache.sysml.api.DMLScript.rtplatform = org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM.HYBRID_SPARK

    val clf = new Mnist_lenet_distrib_sgd()

    val N = 32
    val Nval = 32
    val Ntest = 32
    val C = 3
    val Hin = 112
    val Win = 112
    val K = 10
    val batchSize = 2
    val paralellBatches = 2
    val epochs = 1

    val dummy = clf.generate_dummy_data(N, C, Hin, Win, K)
<<<<<<< HEAD
    val dummyVal = clf.generate_dummy_data(Nval, C, Hin, Win, K)
=======
    println(dummy.X.toBinaryBlocks.count())
   val dummyVal = clf.generate_dummy_data(Nval, C, Hin, Win, K)
>>>>>>> Updates
    val dummyTest = clf.generate_dummy_data(Ntest, C, Hin, Win, K)

    val params = clf.train(dummy.X, dummy.Y, dummyVal.X, dummyVal.Y, C, Hin, Win, batchSize, paralellBatches, epochs)
    println(params.toString)

<<<<<<< HEAD
    val probs = clf.predict(dummyTest.X, C, Hin, Win, params.W1, params.b1, params.W2, params.b2, params.W3, params.b3, params.W4, params.b4)
    val perf = clf.eval(probs, dummyTest.Y)
    println(perf)
=======
    //val probs = clf.predict(dummyTest.X, C, Hin, Win, params.W1, params.b1, params.W2, params.b2, params.W3, params.b3, params.W4, params.b4)
    //val perf = clf.eval(probs, dummyTest.Y)
    //println(perf)
>>>>>>> Updates
  }

}
