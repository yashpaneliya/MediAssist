import json
from typing import Union
from fastapi import Request
from redis import RedisError
from redis.asyncio import Redis
from utils.logger import logger
from core.config import get_settings

settings = get_settings()

class RedisCache:
    """
    A singleton class for interacting with Redis using aioredis.

    Provides methods to get/set values, manage TTLs, and handle JSON serialization,
    with built-in logging and error handling.
    """
    _instance = None
    _client: Union[Redis,None] = None

    def __new__(cls):
        """Ensure only one instance of RedisCache exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
        return cls._instance
    
    

    async def init(self):
        """
        Initializes the Redis client asynchronously.

        Should be called once at app startup before using Redis operations.
        """
        if self._client is None:
            redis_url = (
                f"redis://:{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else "redis://"
            ) + f"{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

            try:
                self._client = await Redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                pong = await self._client.ping()
                logger.info(f"Redis connection established: {pong}")
                logger.info("Redis client initialized in RedisCache.")
            except RedisError as e:
                logger.error(f"Failed to initialize Redis: {e}")
                raise RuntimeError("Redis initialization failed") from e
            

    def get_client(self) -> Redis:
        """
        Returns the initialized Redis client.

        Raises:
            RuntimeError: if Redis client hasn't been initialized.
        """
        if self._client is None:
            raise RuntimeError("Redis client not initialized. Call `await RedisCache().init()` first.")
        return self._client
    
    def get_stateKey(self , session_id: str) -> str:
        return f"state:{session_id}"

    async def get(self, key: str) :
        """
        Get the raw string value for a given Redis key.

        Args:
            key (str): The Redis key.

        Returns:
            str | None: The string value or None if not found or error occurred.
        """
        try:
            value = await self._client.get(key)
            logger.debug(f"GET Redis key={key} found={value is not None}")
            return value
        except RedisError as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(self, key: str, value: str) -> bool:
        """
        Set a key with a string value (no expiry).

        Args:
            key (str): The Redis key.
            value (str): The value to store.

        Returns:
            bool: Success status.
        """
        try:
            await self._client.set(key, value)
            logger.debug(f"SET Redis key={key}")
            return True
        except RedisError as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        """
        Set a key with a string value and TTL (expire time).

        Args:
            key (str): The Redis key.
            ttl (int): Time to live in seconds.
            value (str): The value to store.

        Returns:
            bool: Success status.
        """
        try:
            await self._client.setex(key, ttl, value)
            logger.debug(f"SETEX Redis key={key} ttl={ttl}")
            return True
        except RedisError as e:
            logger.error(f"Redis SETEX error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key (str): The Redis key.

        Returns:
            bool: Success status.
        """
        try:
            await self._client.delete(key)
            logger.debug(f"DEL Redis key={key}")
            return True
        except RedisError as e:
            logger.error(f"Redis DEL error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key (str): The Redis key.

        Returns:
            bool: True if key exists, False otherwise.
        """
        try:
            result = await self._client.exists(key)
            logger.debug(f"EXISTS Redis key={key} exists={result}")
            return result == 1
        except RedisError as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    async def get_json(self, key: str) :
        """
        Get a JSON-decoded object from Redis by key.

        Args:
            key (str): The Redis key.

        Returns:
            dict | None: Decoded dictionary or None.
        """
        raw = await self.get(key)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"Redis JSON decode error for key {key}: {e}")
        return None

    async def set_json(self, key: str, value: dict, ttl: int = None) -> bool:
        """
        Store a JSON-encoded dictionary in Redis.

        Args:
            key (str): The Redis key.
            value (dict): Dictionary to store.
            ttl (int, optional): Time to live in seconds.

        Returns:
            bool: Success status.
        """
        try:
            json_value = json.dumps(value)
            if ttl:
                return await self.setex(key, ttl, json_value)
            else:
                return await self.set(key, json_value)
        except Exception as e:
            logger.error(f"Redis JSON set error for key {key}: {e}")
            return False

    async def close(self):
        try:
            if self._client:
                logger.info("Closing Redis client connection.")
                await self._client.close()
        except RedisError as e:
            logger.error(f"Redis close error: {e}")