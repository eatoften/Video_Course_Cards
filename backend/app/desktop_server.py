import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
HEALTH_PATH = "/health"
LOGGER_NAME = "video_course_cards.desktop"


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    reload: bool
    reuse_existing: bool
    desktop: bool
    log_file: Path | None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


def parse_args(argv: list[str] | None = None) -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="Run the Video Course Cards local FastAPI backend.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("VCC_BACKEND_HOST", DEFAULT_HOST),
        help="Host address for the local backend.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VCC_BACKEND_PORT", DEFAULT_PORT)),
        help="Port for the local backend.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=_env_bool("VCC_BACKEND_RELOAD", False),
        help="Enable uvicorn reload mode. Use only during development.",
    )
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Start a new backend even if the configured /health is ready.",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        default=_env_bool("VCC_DESKTOP", False),
        help="Run with desktop-app defaults.",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("VCC_BACKEND_LOG_FILE"),
        help="Optional backend log file path.",
    )

    args = parser.parse_args(argv)

    return ServerConfig(
        host=args.host,
        port=args.port,
        reload=args.reload,
        reuse_existing=not args.no_reuse_existing,
        desktop=args.desktop,
        log_file=Path(args.log_file) if args.log_file else None,
    )


def configure_logging(config: ServerConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if config.log_file is not None:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(
                config.log_file,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def is_backend_ready(base_url: str, timeout_seconds: float = 1.0) -> bool:
    request = Request(
        f"{base_url}{HEALTH_PATH}",
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def wait_for_backend(
    base_url: str,
    *,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 0.25,
) -> bool:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if is_backend_ready(base_url):
            return True

        time.sleep(interval_seconds)

    return False


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    configure_logging(config)
    logger = logging.getLogger(LOGGER_NAME)

    logger.info(
        "desktop backend entry started host=%s port=%s desktop=%s log_file=%s",
        config.host,
        config.port,
        config.desktop,
        config.log_file,
    )

    if config.reuse_existing and is_backend_ready(config.base_url):
        logger.info("reusing existing backend at %s", config.base_url)
        print(
            f"Video Course Cards backend already running at {config.base_url}.",
            flush=True,
        )
        return 0

    logger.info("starting uvicorn backend at %s", config.base_url)
    print(
        f"Starting Video Course Cards backend at {config.base_url}.",
        flush=True,
    )

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="info",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
