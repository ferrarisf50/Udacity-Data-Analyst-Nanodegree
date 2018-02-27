package NYCtaxi

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

object Xgboost{
  
  

  
  
  def main(args: Array[String]) {
   
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
      .getOrCreate()
    
    // Convert our csv file to a DataSet, using our Person case
    // class to infer the schema.
      
    import spark.implicits._
    val inputLines = spark.sparkContext.textFile("../train_full_parsed_clean5.csv")
    
    val header=inputLines.first
    
    val linesnoheader = inputLines.filter(l => l != header)
    
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
    val coordinates_name = coordinates_name_list.toList
    val coordinates_name2 =coordinates_name_list2.toList
    
    //println(coordinates_name)
    
    def topcoordinates(line:String): List[Any]={
      val re = """(\[\-\d+\.\d+\,\s\d+\.\d+\])""".r
      val coordinates=re.findAllIn(line).toList
      val row_coord= coordinates_name.map(x => if (coordinates contains x) 1 else 0).toList
      
      val feature=line.split(',')
      val feature1=feature.slice(0,56).map(x => x.toFloat).toList
      val feature2=feature1.slice(8,56).map(x => x.toInt).toList
      val feature3=feature1.slice(0,8)
      return feature3++feature2++row_coord
      //return feature2
    }
    val coord = linesnoheader.map(topcoordinates)
    
    
    
    println(coord.first)
    println(coord.first.length)
    
    val feature=List("log_duration", "log_trip_duration", "pickup_longitude", "pickup_latitude",
       "dropoff_longitude", "dropoff_latitude", "great_circle_distance","distance",
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
    
    val colNames=feature++coordinates_name2
    val colNames2=colNames.toSeq
    println(colNames2)
    println(colNames2.length)
     
    val train = coord.toDF(colNames2: _*)
    println("YES")
    /*
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.feature.VectorIndexer
    import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
   
    val feature_col=colNames.slice(0,1)++colNames.slice(2,2854)
    val assembler = new VectorAssembler()
      .setInputCols(feature_col.toArray)
      .setOutputCol("features")
      
    val trainTest = train.randomSplit(Array(0.8, 0.2))
    val trainingData = trainTest(0)
    val testData = trainTest(1)
    
    val gbt = new GBTRegressor()
      .setLabelCol("log_trip_duration")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setMaxDepth(10)
      .setSeed(42)
    
    val pipeline = new Pipeline()
      .setStages(Array(assembler, gbt))
    
    val model = pipeline.fit(trainingData)
    
    */
  }
}