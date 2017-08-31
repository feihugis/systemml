package org.apache.sysml.examples

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.api.mlcontext._
/**
  * Created by Fei Hu on 8/3/17.
  */
object DML_Script_Test {
  def configMLContext(ml: MLContext): Unit = {
    ml.setStatistics(true)
    ml.setStatisticsMaxHeavyHitters(10000)
    ml.setConfigProperty("systemml.stats.finegrained", "true")
    ml.setExplain(true)
    ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SystemML-DML-Script-Test").setMaster("local[4]").set("spark.driver.memory", "2g")

    val sc = new SparkContext(conf)
    sc.setLogLevel("TRACE")
    val ml = new MLContext(sc)
    org.apache.sysml.api.DMLScript.rtplatform = org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM.HYBRID_SPARK
    configMLContext(ml)

    val s =
      """
         a = 190 %% 100
         b = 80
         x = min(a, b)
         y = a - b
         m = rand (rows = x, cols = y)
         gap = 10
         parfor (i in 1:y) {
          batch_beg = (min(x, b) - gap) %% nrow(m)
          batch_end = batch_beg + i
          m_batch = m[batch_beg:batch_end,]
          val_max = max(m_batch)
         }
      """

    val scr = dml(s).out("m")
    val res = ml.execute(scr)
    val m = res.getMatrix("m")
    //val x = res.getData("x")
    println(m.getMatrixMetadata)
  }

}
