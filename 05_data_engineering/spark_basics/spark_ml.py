"""
05 - Data Engineering: Apache Spark Basics
Practice: DataFrames, transformations, ML pipelines, feature engineering
"""

# ── Installation ──────────────────────────────────────────────
# pip install pyspark==3.5.1

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer,
    OneHotEncoder, PCA as SparkPCA
)
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# ══════════════════════════════════════════════════════════════
# 1. SparkSession Setup
# ══════════════════════════════════════════════════════════════

def create_session(app_name: str = "MLOps-Practice") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")  # reduce for local dev
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark version: {spark.version}")
    return spark


# ══════════════════════════════════════════════════════════════
# 2. DataFrame Basics
# ══════════════════════════════════════════════════════════════

def dataframe_basics(spark: SparkSession):
    print("\n── Spark DataFrame Basics ──────────────────────────────")

    # Create DataFrame from Python list
    schema = StructType([
        StructField("sepal_length", FloatType(), True),
        StructField("sepal_width",  FloatType(), True),
        StructField("petal_length", FloatType(), True),
        StructField("petal_width",  FloatType(), True),
        StructField("label",        IntegerType(), True),
    ])

    data = [
        (5.1, 3.5, 1.4, 0.2, 0),
        (7.0, 3.2, 4.7, 1.4, 1),
        (6.3, 3.3, 6.0, 2.5, 2),
        (4.9, 3.0, 1.4, 0.2, 0),
        (5.7, 2.8, 4.1, 1.3, 1),
    ]
    df = spark.createDataFrame(data, schema=schema)

    # Basic operations
    df.printSchema()
    df.show()
    print(f"Row count: {df.count()}")
    df.describe().show()

    # Transformations (lazy — no execution yet)
    df_transformed = (
        df
        .withColumn("sepal_area", F.col("sepal_length") * F.col("sepal_width"))
        .withColumn("petal_ratio", F.col("petal_length") / F.col("petal_width"))
        .filter(F.col("label").isin([0, 1]))
    )
    df_transformed.show()
    return df


# ══════════════════════════════════════════════════════════════
# 3. Feature Engineering with Spark ML
# ══════════════════════════════════════════════════════════════

def feature_engineering(spark: SparkSession):
    print("\n── Feature Engineering Pipeline ────────────────────────")

    # Load CSV (replace with your actual path)
    # df = spark.read.csv("data/raw/iris.csv", header=True, inferSchema=True)

    # Mock data for demo
    schema = StructType([
        StructField("sepal_length", FloatType()),
        StructField("sepal_width",  FloatType()),
        StructField("petal_length", FloatType()),
        StructField("petal_width",  FloatType()),
        StructField("label",        IntegerType()),
    ])
    data = [(5.1, 3.5, 1.4, 0.2, 0), (7.0, 3.2, 4.7, 1.4, 1),
            (6.3, 3.3, 6.0, 2.5, 2), (5.0, 3.6, 1.4, 0.2, 0),
            (5.9, 3.0, 5.1, 1.8, 2), (6.7, 3.1, 4.7, 1.5, 1)] * 20
    df = spark.createDataFrame(data, schema=schema)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                                withMean=True, withStd=True)
    pca       = SparkPCA(k=2, inputCol="features", outputCol="pca_features")

    pipeline = Pipeline(stages=[assembler, scaler, pca])
    model    = pipeline.fit(df)
    df_transformed = model.transform(df)

    df_transformed.select("label", "features", "pca_features").show(5, truncate=False)
    return df_transformed


# ══════════════════════════════════════════════════════════════
# 4. Spark ML: Training a Classifier
# ══════════════════════════════════════════════════════════════

def train_spark_classifier(spark: SparkSession):
    print("\n── Spark ML: Random Forest Classifier ─────────────────")

    schema = StructType([
        StructField("sepal_length", FloatType()),
        StructField("sepal_width",  FloatType()),
        StructField("petal_length", FloatType()),
        StructField("petal_width",  FloatType()),
        StructField("label",        IntegerType()),
    ])
    data = [(5.1, 3.5, 1.4, 0.2, 0), (7.0, 3.2, 4.7, 1.4, 1),
            (6.3, 3.3, 6.0, 2.5, 2), (5.0, 3.6, 1.4, 0.2, 0),
            (5.9, 3.0, 5.1, 1.8, 2), (6.7, 3.1, 4.7, 1.5, 1)] * 25
    df = spark.createDataFrame(data, schema=schema)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features")
    rf        = RandomForestClassifier(featuresCol="features", labelCol="label",
                                        numTrees=50, seed=42)

    pipeline = Pipeline(stages=[assembler, scaler, rf])
    model    = pipeline.fit(train_df)
    preds    = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    preds.select("label", "prediction", "probability").show(10)


# ══════════════════════════════════════════════════════════════
# 5. Aggregate Data Pipelines (ETL style)
# ══════════════════════════════════════════════════════════════

def data_pipeline(spark: SparkSession):
    print("\n── ETL Pipeline ────────────────────────────────────────")
    # Simulating log-style data
    logs = [
        ("2025-01-01", "model_a", 0.91, "success"),
        ("2025-01-01", "model_b", 0.84, "success"),
        ("2025-01-02", "model_a", 0.78, "degraded"),
        ("2025-01-02", "model_b", 0.95, "success"),
        ("2025-01-03", "model_a", 0.62, "failure"),
    ]
    schema = "date STRING, model STRING, accuracy DOUBLE, status STRING"
    df = spark.createDataFrame(logs, schema=schema)

    # Aggregation: daily model performance
    result = (
        df
        .groupBy("model")
        .agg(
            F.avg("accuracy").alias("avg_accuracy"),
            F.min("accuracy").alias("min_accuracy"),
            F.count("*").alias("n_runs"),
            F.sum(F.when(F.col("status") == "failure", 1).otherwise(0)).alias("failures"),
        )
        .withColumn("failure_rate", F.col("failures") / F.col("n_runs"))
        .orderBy("avg_accuracy", ascending=False)
    )
    result.show()


if __name__ == "__main__":
    spark = create_session()
    dataframe_basics(spark)
    feature_engineering(spark)
    train_spark_classifier(spark)
    data_pipeline(spark)
    spark.stop()
    print("\n✅ All Spark exercises complete!")
