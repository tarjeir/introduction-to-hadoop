import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.elasticsearch.spark.rdd.EsSpark

val data = sc.textFile("hdfs:///data/movies.csv")
val date = """(\d\d\d\d)""".r

def toMovie(movie:Any): Any = movie match {
  case Array(movieId:String,title:String,genres:String) => {
    val yearFound = for (m <- date findFirstMatchIn title) yield m group 1
    val year = yearFound getOrElse "0"
    Map("movieId" -> movieId, "year" -> year.toInt,"title" -> title, "genres" -> genres.toString.split('|'))
  }
}

val movieData = data.filter(line => !line.contains("movieId")).
                map(_.split(',')).
                filter(array => array.length == 3).
                map(toMovie)

val uncleanMovieData = data.filter(line => !line.contains("movieId")).
                       map(_.split(",\"|\",")).
                       filter(array => array.length == 3).
                       map(toMovie)

val settingsMap = Map("es.nodes"->"127.0.0.1", "es.port" -> "9201","es.index.auto.create" -> "false", "es.mapping.id" -> "movieId")
val indexAndType = "recommendation/movies"

EsSpark.saveToEs(movieData, indexAndType,settingsMap)
EsSpark.saveToEs(uncleanMovieData, indexAndType, settingsMap)
