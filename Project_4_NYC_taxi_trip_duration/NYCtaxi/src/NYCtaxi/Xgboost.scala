package NYCtaxi

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

object Xgboost{
  
  def topcoordinates(line:String): String={
    return line    
  }

  
  
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
    
    println(coordinates_name)
    
    
    
  }
}