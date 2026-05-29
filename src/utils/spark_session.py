import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark_session(app_name: str = "PharmaLakehouse") -> SparkSession:
    """
    Create and return a local SparkSession.

    PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are set to sys.executable
    so Spark uses the exact Python binary that is currently running —
    the one inside the virtual environment. Without this, on Windows,
    Spark picks up the system 'python' alias which points to the
    Microsoft Store stub and fails to start workers.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark