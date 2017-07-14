package org.apache.sysml.examples

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.scripts.nn.examples.Mnist_lenet_distrib_sgd

/**
  * Created by Fei Hu on 7/7/17.
  */
object MNIST_Distrib_Sgd {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SystemML-MNIST-Distrib").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val ml = new MLContext(sc)
    org.apache.sysml.api.DMLScript.rtplatform = org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM.SPARK

    val dMLScript = dmlFromFile("/Users/fei.hu1@ibm.com/Documents/GitHub/systemml/scripts/nn/examples/mnist_lenet_distrib_sgd-train-dummy-data.dml")
    //ml.execute(dMLScript)

    val clf = new Mnist_lenet_distrib_sgd()

    val N = 1
    val Nval = 1
    val Ntest = 1
    val C = 3
    val Hin = 224
    val Win = 224
    val K = 10
    val batchSize = 32
    val paralellBatches = 4
    val epochs = 1

    val dummy = clf.generate_dummy_data(N, C, Hin, Win, K)
    val dummyVal = clf.generate_dummy_data(Nval, C, Hin, Win, K)
    val dummyTest = clf.generate_dummy_data(Ntest, C, Hin, Win, K)

    val params = clf.train(dummy.X, dummy.Y, dummyVal.X, dummyVal.Y, C, Hin, Win, batchSize, paralellBatches, epochs)
    //println(params.toString)

    val probs = clf.predict(dummyTest.X, C, Hin, Win, params.W1, params.b1, params.W2, params.b2, params.W3, params.b3, params.W4, params.b4)
    val perf = clf.eval(probs, dummyTest.Y)
  }

}
