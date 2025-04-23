from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, StringIndexerModel

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

# Load data
df = spark.read.parquet("/opt/ml/processing/input")

# Fill missing values
fill_values = {
    'category_col_1': 'unknown',
    'category_col_2': 'unknown',
    'numeric_col_1': 0,
    'numeric_col_2': 0
}
df = df.fillna(fill_values)

# Data consistency check: drop rows with null keys
df = df.filter(col("id").isNotNull())

# Split the dataset by pre-defined 'split' column
train_df = df.filter(col("split") == "train")
val_df = df.filter(col("split") == "val")

# List of categorical columns to encode
categorical_cols = ['category_col_1', 'category_col_2']
indexers = []
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
    indexers.append(indexer.fit(train_df))

# Apply fitted indexers to train and validation sets
for model in indexers:
    train_df = model.transform(train_df)
    val_df = model.transform(val_df)

# Write outputs
train_df.write.mode("overwrite").parquet("/opt/ml/processing/output/train")
val_df.write.mode("overwrite").parquet("/opt/ml/processing/output/val")

