package org.apache.sysml.examples

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.parser.Expression.ValueType
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType
import org.apache.sysml.runtime.io.{MatrixReaderFactory, MatrixWriter, MatrixWriterFactory}
import org.apache.sysml.runtime.matrix.data.InputInfo
import org.apache.sysml.runtime.util.MapReduceTool
/**
  * Created by Fei Hu on 8/3/17.
  */
object DML_Script_Test {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SystemML-DML-Script-Test").setMaster("local[4]").set("spark.driver.memory", "2g")

    val sc = new SparkContext(conf)
    sc.setLogLevel("TRACE")
    val ml = new MLContext(sc)

    val s =
      """
        x = read("scratch_space/X_input")
      """

    val scr = dml(s).out("x")
    val res = ml.execute(scr)
    val x = res.getMatrix("x")
    println(x.getMatrixMetadata)
  }

}
