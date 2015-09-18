import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.elasticsearch.spark.rdd.EsSpark

val data = sc.textFile("hdfs:///data/ratings.csv")

val ratings = data.
                filter(line => !line.contains("userId,movieId,rating,timestamp")).
                map(_.split(',') match {
                  case Array(userid,movieid,rating,timestamp)  => {
                    Rating(userid.toInt,movieid.toInt,rating.toDouble)
                  }
                })

val rank = 10
val numIterations = 10
val model = ALS.train(ratings, rank, numIterations, 0.01)
model.save(sc, "hdfs:///data/trainedmodel/")


val users = ratings.map { case Rating(user, product, rate) =>
  user
}.distinct()
val modelBroadcast = sc.broadcast(model)
val topRecommendedProducts = users.
                                  map(user => modelBroadcast.value.recommendProducts(user,10)).
                                  flatMap(r => r)

val recommendedMovies = topRecommendedProducts.map {
                    case Rating(userId, movieId, rate) =>{
                      val concated = userId.toString+movieId.toString
                      Map("id" -> concated,"userId" -> userId, "movieId" -> movieId, "rate" -> rate)
                    }
                  }

EsSpark.saveToEs(recommendedMovies, "recommendation/recommendations",Map("es.nodes"->"127.0.0.1","es.port" -> "9201","es.index.auto.create" -> "false", "es.mapping.id" -> "id","es.mapping.parent" -> "movieId"))
