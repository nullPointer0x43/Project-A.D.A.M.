import pickle
import redis
import pandas as pd
import io


class DataCache:
    def __init__(self, host='redis-cache', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)

    def get_dataframe(self, user_id, path):
        key = f"df:{user_id}:{path}"
        data = self.client.get(key)

        if data:
            return pd.read_feather(io.BytesIO(data))
        return None

    def set_dataframe(self, thread_id, path, df, ttl=1800):
        key = f"df:{thread_id}:{path}"

        buffer = io.BytesIO()
        df.to_feather(buffer)

        self.client.setex(key, ttl, buffer.getvalue())

    def get_pickle(self, user_id: str, path: str):
        key = f"df:{user_id}:{path}"
        data = self.client.get(key)

        if data:
            print("Cache hit")
            return pickle.loads(data)

        return None

    def set_pickle(self, thread_id, path, obj, ttl=1800):
        key = f"obj:{thread_id}:{path}"

        pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        self.client.setex(key, ttl, pickled_data)

    def delete_dataframe(self, thread_id, path):
        key = f"df:{thread_id}:{path}"
        result = self.client.delete(key)

        return result > 0

    def clear_user_cache(self, thread_id):
        pattern = f"df:{thread_id}:*"
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0