"""Redis atomic operations wrapper."""

from __future__ import annotations

try:  # Optional dependency; fallback to in-memory cache methods if unavailable.
    import redis
except Exception:  # pragma: no cover - optional runtime dependency
    redis = None


class RedisAtomicClient:
    """Thin wrapper around common Redis atomic operations."""

    # 方法注释（业务）:
    # - 业务：初始化 Redis 原子客户端，建立可用连接并保留降级兜底能力。
    # - 入参：`redis_url`(str)=Redis 连接地址（可选）。
    # - 出参：`None`，主要完成对象内部状态初始化。
    # - 逻辑：当 URL 与依赖可用时尝试连接并 `ping`；失败则保持 `_client=None` 以便上层走内存兜底。
    def __init__(self, *, redis_url: str = "") -> None:
        self.redis_url = (redis_url or "").strip()
        self._client = None
        if self.redis_url and redis is not None:
            try:
                client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                client.ping()
                self._client = client
            except Exception:
                self._client = None

    # 方法注释（业务）:
    # - 业务：判断当前 Redis 是否已启用。
    # - 入参：无。
    # - 出参：`bool`，返回连接状态。
    # - 逻辑：仅通过 `_client` 是否非空判断可用性。
    @property
    def redis_enabled(self) -> bool:
        return self._client is not None

    # 方法注释（业务）:
    # - 业务：执行 Redis 字符串读取。
    # - 入参：`key`(str)=缓存键。
    # - 出参：`str | None`，返回键对应值或空。
    # - 逻辑：无客户端直接返回空；有客户端时调用 `get`，异常统一兜底为空。
    def get(self, key: str) -> str | None:
        if self._client is None:
            return None
        try:
            value = self._client.get(key)
        except Exception:
            return None
        return None if value is None else str(value)

    # 方法注释（业务）:
    # - 业务：执行 Redis 字符串写入（覆盖写）。
    # - 入参：`key`(str)=缓存键；`value`(str)=缓存值；`ex`(int | None)=过期秒数。
    # - 出参：`bool`，返回写入是否成功。
    # - 逻辑：无客户端返回失败；调用 `set`，异常时返回失败。
    def set(self, key: str, value: str, *, ex: int | None = None) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.set(key, value, ex=ex))
        except Exception:
            return False

    # 方法注释（业务）:
    # - 业务：执行 Redis `SET NX`（仅当 key 不存在时写入）。
    # - 入参：`key`(str)=缓存键；`value`(str)=缓存值；`ex`(int | None)=过期秒数。
    # - 出参：`bool`，返回是否抢占写入成功。
    # - 逻辑：无客户端返回失败；调用 `set(..., nx=True)`，异常统一返回失败。
    def set_nx(self, key: str, value: str, *, ex: int | None = None) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.set(key, value, ex=ex, nx=True))
        except Exception:
            return False

    # 方法注释（业务）:
    # - 业务：执行 Redis 键删除。
    # - 入参：`key`(str)=缓存键。
    # - 出参：`int`，返回删除条数。
    # - 逻辑：无客户端返回 0；调用 `delete`，异常统一返回 0。
    def delete(self, key: str) -> int:
        if self._client is None:
            return 0
        try:
            return int(self._client.delete(key) or 0)
        except Exception:
            return 0

    # 方法注释（业务）:
    # - 业务：执行 Redis 列表区间读取。
    # - 入参：`key`(str)=列表键；`start`(int)=起始下标；`end`(int)=结束下标。
    # - 出参：`list[str]`，返回列表切片结果。
    # - 逻辑：无客户端返回空数组；调用 `lrange` 并统一转字符串列表，异常时返回空数组。
    def lrange(self, key: str, start: int, end: int) -> list[str]:
        if self._client is None:
            return []
        try:
            rows = self._client.lrange(key, start, end) or []
        except Exception:
            return []
        return [str(item) for item in rows]

    # 方法注释（业务）:
    # - 业务：原子替换 Redis 列表内容（先删后写）。
    # - 入参：`key`(str)=列表键；`values`(list[str])=新列表值；`ttl_seconds`(int | None)=过期秒数。
    # - 出参：`bool`，返回替换是否成功。
    # - 逻辑：使用 pipeline 执行 delete + rpush (+ expire)；异常统一返回失败。
    def replace_list(self, key: str, values: list[str], *, ttl_seconds: int | None = None) -> bool:
        if self._client is None:
            return False
        try:
            pipe = self._client.pipeline()
            pipe.delete(key)
            if values:
                pipe.rpush(key, *values)
                if ttl_seconds is not None:
                    pipe.expire(key, max(1, int(ttl_seconds)))
            pipe.execute()
            return True
        except Exception:
            return False

    # 方法注释（业务）:
    # - 业务：向 Redis 列表尾部追加单条记录。
    # - 入参：`key`(str)=列表键；`value`(str)=追加值；`ttl_seconds`(int | None)=过期秒数。
    # - 出参：`bool`，返回追加是否成功。
    # - 逻辑：使用 pipeline 执行 rpush (+ expire)；异常统一返回失败。
    def append_list_item(self, key: str, value: str, *, ttl_seconds: int | None = None) -> bool:
        if self._client is None:
            return False
        try:
            pipe = self._client.pipeline()
            pipe.rpush(key, value)
            if ttl_seconds is not None:
                pipe.expire(key, max(1, int(ttl_seconds)))
            pipe.execute()
            return True
        except Exception:
            return False
