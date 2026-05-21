import io
import pickle
import redis
import pandas as pd
from URLs import DEFAULT_REDIS_HOST

DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0

KEY_PREFIX_DF = "df:{user_id}:{path}"
KEY_PREFIX_OBJ = "obj:{thread_id}:{path}"
PATTERN_USER_CACHE = "df:{thread_id}:*"


class DataCache:
    def __init__(self, host=DEFAULT_REDIS_HOST, port=DEFAULT_REDIS_PORT, db=DEFAULT_REDIS_DB):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)

    def get_dataframe(self, user_id, path):
        key = KEY_PREFIX_DF.format(user_id=user_id, path=path)
        data = self.client.get(key)

        if data:
            return pd.read_feather(io.BytesIO(data))
        return None

    def set_dataframe(self, thread_id, path, df, ttl=1800):
        key = KEY_PREFIX_DF.format(user_id=thread_id, path=path)

        buffer = io.BytesIO()
        df.to_feather(buffer)

        self.client.setex(key, ttl, buffer.getvalue())

    def get_pickle(self, user_id: str, path: str):
        key = KEY_PREFIX_DF.format(user_id=user_id, path=path)
        data = self.client.get(key)

        if data:
            print("Cache hit")
            return pickle.loads(data)

        return None

    def set_pickle(self, thread_id, path, obj, ttl=1800):
        key = KEY_PREFIX_OBJ.format(thread_id=thread_id, path=path)

        pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        self.client.setex(key, ttl, pickled_data)

    def delete_dataframe(self, thread_id, path):
        key = KEY_PREFIX_DF.format(user_id=thread_id, path=path)
        result = self.client.delete(key)

        return result > 0

    def clear_user_cache(self, thread_id):
        pattern = PATTERN_USER_CACHE.format(thread_id=thread_id)
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0