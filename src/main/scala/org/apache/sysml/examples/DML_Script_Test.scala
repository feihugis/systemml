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
    ml.setExplainLevel(MLContext.ExplainLevel.HOPS)
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
         m = read("scratch_space/X_input")
         gap = 10

         print (nrow(m))

         parfor (i in 1:y, log=DEBUG) {
           batch_beg = (min(x, b) - gap) %% nrow(m)
           batch_end = batch_beg + i
           m_batch = m[batch_beg:batch_end, 1:100]
           val_max = max(m_batch)
           print (val_max)
         }
      """

    /*
    *
    * */

    val scr = dml(s).out("m")
    val res = ml.execute(scr)
    val m = res.getMatrix("m")
    //val x = res.getData("x")
    println(m.getMatrixMetadata)
  }

}
