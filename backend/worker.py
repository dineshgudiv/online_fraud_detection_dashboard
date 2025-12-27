"""RQ worker entrypoint."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

from redis import Redis
from rq import Queue, Worker

from core import config


def _safe_redis_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    netloc = hostname
    if parsed.username:
        if parsed.password:
            netloc = f"{parsed.username}:****@{hostname}"
        else:
            netloc = f"{parsed.username}@{hostname}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = Redis.from_url(config.REDIS_URL)
    queue_list = [Queue(name, connection=conn) for name in config.RQ_QUEUES]
    queue_names = [queue.name for queue in queue_list]
    logging.info(
        "Starting RQ worker (queues=%s, redis=%s)",
        ",".join(queue_names),
        _safe_redis_url(config.REDIS_URL),
    )
    worker = Worker(queue_list, connection=conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
