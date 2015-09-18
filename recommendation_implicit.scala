import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.elasticsearch.spark.rdd.EsSpark
import org.apache.spark.sql.Row

val events = sqlContext.jsonFile("hdfs:///data/movieevents.json")
events.registerTempTable("event")

val userIdAndMovieId = sqlContext.sql("SELECT userId, movieId FROM event").rdd

val viewRating = userIdAndMovieId.map {
  case Row(userId:Long, movieId:Long) => Rating(userId.toInt,movieId.toInt,1.0)
}

val rank = 10
val numIterations = 10
val implicitModel = ALS.trainImplicit(viewRating,rank,numIterations)
implicitModel.save(sc, "hdfs:///data/trainedimplicitmodel/")

val users = viewRating.map { case Rating(user, product, rate) =>
  user
}.distinct()

val modelBroadcast = sc.broadcast(implicitModel)

val topRecommendedProducts = users.map(user => modelBroadcast.value.recommendProducts(user,10)).
                                  flatMap(r => r)

val recommendedMovies = topRecommendedProducts.map {
                    case Rating(userId, movieId, rate) =>{
                      val concated = userId.toString+movieId.toString
                      Map("id" -> concated,"userId" -> userId, "movieId" -> movieId, "rate" -> rate)
                    }
                  }

EsSpark.saveToEs(recommendedMovies, "recommendation/implicitrecommendations",Map("es.port" -> "9201","es.index.auto.create" -> "false", "es.mapping.id" -> "id","es.mapping.parent" -> "movieId"))
