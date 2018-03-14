package NYCtaxi

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

object Xgboost2{
  
  
  

  def main(args: Array[String]) {
     Logger.getLogger("org").setLevel(Level.ERROR)
     //import ml.dmlc.xgboost4j.scala.Booster
     //import ml.dmlc.xgboost4j.scala.spark.XGBoost
      
     import org.apache.spark.sql.SQLContext
     val sc = new SparkContext("local[*]", "Xgboost2")
     val sqlContext = new SQLContext(sc)
     //val train_data = sqlContext.read.parquet("../parquet_file/*.parquet")
     val train_data = sqlContext.read.parquet("../parquet_file/part-00000-11fa81c8-4062-4104-914a-0778b470228d-c000.snappy.parquet")
     println(train_data.count())

     
     
      import scala.io.Source
    
      val topcoord="../coordinate_more_than_1000_scala.csv"
      
      import scala.collection.mutable.ListBuffer
      
      var coordinates_name_list = new ListBuffer[String]()
      var coordinates_name_list2 = new ListBuffer[String]()
      
      
      var num = 0
      
      for (line <- Source.fromFile(topcoord).getLines){
        if (num >0){
          var coordinates: String =line.split('|')(1)
          var varname: String =line.split('|')(2)
  
          coordinates_name_list += coordinates
          coordinates_name_list2 += varname
          
        }
        num +=1
      }
      val coordinates_name = coordinates_name_list.toArray
      val coordinates_name2 =coordinates_name_list2.toArray
      
      val feature0=Array("log_trip_duration")
      val feature1=Array("log_duration",  "pickup_longitude", "pickup_latitude",
         "dropoff_longitude", "dropoff_latitude", "great_circle_distance","distance")
      val feature2=Array(   
         "snow", "holiday", "vendor_id", "pickup_hour_0",
         "pickup_hour_1", "pickup_hour_2", "pickup_hour_3", "pickup_hour_4",
         "pickup_hour_5", "pickup_hour_6", "pickup_hour_7", "pickup_hour_8",
         "pickup_hour_9", "pickup_hour_10", "pickup_hour_11",
         "pickup_hour_12", "pickup_hour_13", "pickup_hour_14",
         "pickup_hour_15", "pickup_hour_16", "pickup_hour_17",
         "pickup_hour_18", "pickup_hour_19", "pickup_hour_20",
         "pickup_hour_21", "pickup_hour_22", "pickup_hour_23",
         "pickup_weekday_0", "pickup_weekday_1", "pickup_weekday_2",
         "pickup_weekday_3", "pickup_weekday_4", "pickup_weekday_5",
         "pickup_weekday_6", "pickup_month_1", "pickup_month_2",
         "pickup_month_3", "pickup_month_4", "pickup_month_5",
         "pickup_month_6", "passenger_count_0", "passenger_count_1",
         "passenger_count_2", "passenger_count_3", "passenger_count_4",
         "passenger_count_5", "passenger_count_6", "passenger_count_7")
      
      //val colNames=feature1 ++ feature2 ++coordinates_name2
      //println(colNames)
      //println(colNames.length)
      
      
      import org.apache.spark.ml.feature.OneHotEncoder
      import org.apache.spark.ml.feature.VectorAssembler
      import org.apache.spark.ml.feature.StringIndexer
  
      
      
      val categoricalFeatColNames = feature2 ++ coordinates_name2
      val Indexers = categoricalFeatColNames.map ( x =>
        new StringIndexer()
        .setInputCol(x)
        .setOutputCol(x+"_idx") 
      )
      
   
      val encoder = categoricalFeatColNames.map ( x =>
       new OneHotEncoder()
      .setInputCol(x+"_idx")
      .setOutputCol(x+"_enc")
      )
  
      
      val idxdCategoricalFeatColName = categoricalFeatColNames.map(_ + "_enc")
      val allencFeatColNames = feature1 ++ idxdCategoricalFeatColName
      val assembler = new VectorAssembler()
        .setInputCols(Array(allencFeatColNames: _*))
        .setOutputCol("features")
        
      
      import org.apache.spark.ml.linalg.Vectors
      import org.apache.spark.ml.Pipeline
      import org.apache.spark.ml.evaluation.RegressionEvaluator
      import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
      //import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoostClassificationModel}
      
      val trainTest = train_data.randomSplit(Array(0.8, 0.2))
      val trainingData = trainTest(0)
      val testData = trainTest(1)
      
      val gbt = new GBTRegressor()
        .setLabelCol("log_trip_duration")
        .setFeaturesCol("features")
        .setMaxIter(10)
        .setMaxDepth(10)
        .setSeed(42)
        
      
      val pipeline1 = new Pipeline().setStages(Indexers ++ encoder)  
        
      val transformed = pipeline1.fit(trainingData).transform(trainingData)  
      
      val pipeline2 = new Pipeline()
        .setStages(Array(assembler, gbt))
        
      val model=pipeline2.fit(transformed)
      val predictions = model.transform(testData)
      predictions.select("prediction", "log_trip_duration", "features").show(5)
      
      val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")
      val rmse = evaluator.evaluate(predictions)
      println("Root Mean Squared Error (RMSE) on test data = " + rmse)
  
      val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
      println("Learned regression GBT model:\n" + gbtModel.toDebugString)
      //import ml.dmlc.xgboost4j.scala.DMatrix
      //import ml.dmlc.xgboost4j.scala.XGBoost
    
     

      

  }
}