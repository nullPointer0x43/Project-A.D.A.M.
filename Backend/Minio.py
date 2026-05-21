import io
import os
import pickle
import pandas as pd
from minio import Minio
from minio.deleteobjects import DeleteObject
from Redis import DataCache
from URLs import DEFAULT_MINIO_HOST, DEFAULT_MINIO_PORT, DEFAULT_MINIO_ACCESS_KEY, DEFAULT_MINIO_SECRET_KEY


DEFAULT_MINIO_SECURE = False

DEFAULT_BUCKET_NAME = "analysis-artifacts"


class DataBaseMinio:
    def __init__(self, host=DEFAULT_MINIO_HOST, port=DEFAULT_MINIO_PORT):
        self.minio_client = Minio(
            f"{host}:{port}",
            access_key=DEFAULT_MINIO_ACCESS_KEY,
            secret_key=DEFAULT_MINIO_SECRET_KEY,
            secure=DEFAULT_MINIO_SECURE
        )

        self.cache = DataCache()
        self.BUCKET_NAME = DEFAULT_BUCKET_NAME
        self.prepare_minio()

    def prepare_minio(self):
        if not self.minio_client.bucket_exists(self.BUCKET_NAME):
            self.minio_client.make_bucket(self.BUCKET_NAME)

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

    def return_plot_path(self, minio_path: str):
        try:
            self.minio_client.stat_object(self.BUCKET_NAME, minio_path)
            path_parts = minio_path.split('/')
            thread_id = path_parts[0]
            insight_id = path_parts[2]
            filename = path_parts[3]

            base_backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            proxy_url = f"{base_backend_url}/api/artifacts/{thread_id}/insights/{insight_id}/{filename}"

            return proxy_url

        except Exception as e:
            print(f"Error mapping proxy path for {minio_path}: {e}")
            return None

    def delete_folder(self, prefix: str):
        try:
            objects_to_delete = self.minio_client.list_objects(
                self.BUCKET_NAME,
                prefix=prefix,
                recursive=True
            )

            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

            if delete_list:
                errors = self.minio_client.remove_objects(self.BUCKET_NAME, delete_list)
                for error in errors:
                    print(f"Failed to delete object {error.object_name}: {error}")
                print(f"Successfully purged MinIO directory path: {prefix}")
            else:
                print(f"No assets found under MinIO directory path: {prefix}")

        except Exception as e:
            print(f"Error while running bulk folder deletion in MinIO: {str(e)}")
            raise e