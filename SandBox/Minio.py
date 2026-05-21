import io
import pickle
import pandas as pd
from minio import Minio
from Redis import DataCache

class DataBaseMinio:
    def __init__(self, host='minio-storage', port=9000):
        self.minio_client = Minio(
            f"{host}:{port}",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

        self.cache = DataCache()

        self.BUCKET_NAME = "analysis-artifacts"
        self.prepare_minio()

    def prepare_minio(self):
        try:
            if not self.minio_client.bucket_exists(self.BUCKET_NAME):
                self.minio_client.make_bucket(self.BUCKET_NAME)
        except Exception as e:
            print(f"Error during MinIO bucket check/creation: {e}")
            raise e

    def write_df(self, df: pd.DataFrame, minio_path: str, thread_id: str, write_through: bool = True):
        if write_through:
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False, engine='pyarrow', compression='snappy')

            parquet_size = parquet_buffer.tell()
            parquet_buffer.seek(0)

            self.minio_client.put_object(
                self.BUCKET_NAME,
                minio_path,
                data=parquet_buffer,
                length=parquet_size,
                content_type='application/octet-stream'
            )

        self.cache.set_dataframe(thread_id, minio_path, df)

    def write_image(self, local_file_path: str, minio_path: str):
        self.minio_client.fput_object(
            bucket_name=self.BUCKET_NAME,
            object_name=minio_path,
            file_path=local_file_path
        )

    def write_pickle(self, obj, minio_path: str, thread_id: str, write_through: bool = True):
        if write_through:
            pickle_buffer = io.BytesIO()
            pickle.dump(obj, pickle_buffer, protocol=pickle.HIGHEST_PROTOCOL)

            pickle_size = pickle_buffer.tell()
            pickle_buffer.seek(0)

            self.minio_client.put_object(
                self.BUCKET_NAME,
                minio_path,
                data=pickle_buffer,
                length=pickle_size,
                content_type='application/x-python-pickle'
            )

        self.cache.set_pickle(thread_id, minio_path, obj)

    def get_df(self, minio_path: str, thread_id: str):
        df = self.cache.get_dataframe(thread_id, minio_path)

        if df is not None:
            print("cache hit")
            return df

        try:
            response = self.minio_client.get_object(self.BUCKET_NAME, minio_path)
            parquet_buffer = io.BytesIO(response.read())

            df = pd.read_parquet(parquet_buffer, engine='pyarrow')
            self.cache.set_dataframe(thread_id, minio_path, df)

            return df
        except Exception as e:
            print(f"Error retrieving dataframe: {e}")
            raise e
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()

    def get_pickle(self, minio_path: str, thread_id: str):
        obj = self.cache.get_pickle(thread_id, minio_path)

        if obj is not None:
            print("cache hit")
            return obj

        try:
            response = self.minio_client.get_object(self.BUCKET_NAME, minio_path)
            pickle_buffer = io.BytesIO(response.read())

            obj = pickle.load(pickle_buffer)
            self.cache.set_pickle(thread_id, minio_path, obj)

            return obj
        except Exception as e:
            print(f"Error retrieving pickled object: {e}")
            raise e
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()


    def delete_df(self, minio_path: str):
        try:
            self.minio_client.remove_object(self.BUCKET_NAME, minio_path)
        except Exception as e:
            print(f"MinIO delete error: {e}")