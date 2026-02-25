"""
RDT Trading System - Custom Log Handlers

Provides handlers for:
- Grafana Loki (push-based log aggregation)
- Elasticsearch (search and analytics)

Features:
- Batch sending for efficiency
- Retry logic for failed sends
- Buffer for offline operation
- Async support
"""

import atexit
import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class LogRecord:
    """Structured log record for handlers."""
    timestamp: datetime
    level: str
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else datetime.now(timezone.utc).isoformat(),
            'level': self.level,
            'message': self.message,
            'labels': self.labels,
            **self.extra,
        }


class BaseHandler(ABC):
    """Base class for log handlers with common functionality."""

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_buffer_size: int = 10000,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 10.0,
    ):
        """
        Initialize the base handler.

        Args:
            batch_size: Number of records to batch before sending
            flush_interval: Seconds between automatic flushes
            max_buffer_size: Maximum records to buffer (drops oldest when full)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Thread-safe buffer
        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._buffer_lock = threading.Lock()

        # Offline buffer for failed sends
        self._offline_buffer: deque = deque(maxlen=max_buffer_size)

        # Stats
        self._sent_count = 0
        self._failed_count = 0
        self._dropped_count = 0

        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        # HTTP session with retry
        self._session = self._create_session()

        # Register cleanup
        atexit.register(self.close)

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "PUT"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def start(self):
        """Start the background flush thread."""
        if self._started:
            return

        self._stop_event.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        self._started = True

    def stop(self):
        """Stop the background flush thread."""
        if not self._started:
            return

        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=self.timeout)
        self._started = False

    def _flush_loop(self):
        """Background loop for periodic flushing."""
        while not self._stop_event.is_set():
            try:
                self.flush()
            except Exception as e:
                import sys
                print(f"Error in flush loop: {e}", file=sys.stderr)
            self._stop_event.wait(self.flush_interval)

    def write(self, record: LogRecord):
        """
        Write a log record to the buffer.

        Args:
            record: LogRecord to write
        """
        with self._buffer_lock:
            self._buffer.append(record)

            # Auto-flush if buffer is full
            if len(self._buffer) >= self.batch_size:
                self._flush_buffer()

    def emit(self, record: dict):
        """
        Emit a log record (Loguru compatibility).

        Args:
            record: Loguru record dictionary
        """
        try:
            log_record = self._parse_loguru_record(record)
            self.write(log_record)
        except Exception:
            pass  # Don't fail logging

    def _parse_loguru_record(self, record: dict) -> LogRecord:
        """Parse a Loguru record into a LogRecord."""
        # Extract timestamp
        timestamp = record.get('time')
        if timestamp and hasattr(timestamp, 'timestamp'):
            timestamp = datetime.fromtimestamp(timestamp.timestamp(), tz=timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Extract level
        level = record.get('level', {})
        level_name = level.name if hasattr(level, 'name') else str(level)

        # Extract message
        message = str(record.get('message', ''))

        # Extract extra context
        extra = record.get('extra', {}).copy()

        # Build labels from known fields
        labels = {
            'level': level_name.lower(),
        }
        if 'service' in extra:
            labels['service'] = str(extra.pop('service'))
        if 'correlation_id' in extra:
            labels['correlation_id'] = str(extra.pop('correlation_id'))

        return LogRecord(
            timestamp=timestamp,
            level=level_name,
            message=message,
            labels=labels,
            extra=extra,
        )

    def flush(self):
        """Flush buffered records to the destination."""
        records = []
        with self._buffer_lock:
            while self._buffer:
                records.append(self._buffer.popleft())

        # Also try to send any offline buffered records
        offline_records = []
        while self._offline_buffer:
            try:
                offline_records.append(self._offline_buffer.popleft())
            except IndexError:
                break

        all_records = offline_records + records
        if not all_records:
            return

        self._flush_buffer(all_records)

    def _flush_buffer(self, records: List[LogRecord] = None):
        """Flush records to the destination."""
        if records is None:
            records = []
            with self._buffer_lock:
                while self._buffer:
                    records.append(self._buffer.popleft())

        if not records:
            return

        try:
            self._send_batch(records)
            self._sent_count += len(records)
        except Exception as e:
            self._failed_count += len(records)
            # Move to offline buffer for retry
            for record in records:
                if len(self._offline_buffer) < self.max_buffer_size:
                    self._offline_buffer.append(record)
                else:
                    self._dropped_count += 1

    @abstractmethod
    def _send_batch(self, records: List[LogRecord]):
        """Send a batch of records. Implement in subclass."""
        pass

    def close(self):
        """Close the handler and flush remaining records."""
        self.stop()
        try:
            self.flush()
        except Exception:
            pass
        self._session.close()

    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            'sent': self._sent_count,
            'failed': self._failed_count,
            'dropped': self._dropped_count,
            'buffered': len(self._buffer),
            'offline_buffered': len(self._offline_buffer),
        }


class LokiHandler(BaseHandler):
    """
    Handler for sending logs to Grafana Loki.

    Loki uses a push model where logs are sent to the /loki/api/v1/push endpoint.
    Logs are organized by labels (streams) and contain timestamped entries.
    """

    def __init__(
        self,
        url: str,
        labels: Dict[str, str] = None,
        auth: tuple = None,
        tenant_id: str = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_buffer_size: int = 10000,
        retry_attempts: int = 3,
        timeout: float = 10.0,
    ):
        """
        Initialize the Loki handler.

        Args:
            url: Loki push URL (e.g., 'http://localhost:3100')
            labels: Default labels for all log streams
            auth: Optional (username, password) tuple for basic auth
            tenant_id: Optional tenant ID for multi-tenant Loki
            batch_size: Number of records to batch before sending
            flush_interval: Seconds between automatic flushes
            max_buffer_size: Maximum records to buffer
            retry_attempts: Number of retry attempts
            timeout: Request timeout in seconds
        """
        super().__init__(
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_buffer_size=max_buffer_size,
            retry_attempts=retry_attempts,
            timeout=timeout,
        )

        self.url = url.rstrip('/')
        self.push_url = urljoin(self.url + '/', 'loki/api/v1/push')
        self.default_labels = labels or {'app': 'rdt-trading'}
        self.auth = auth
        self.tenant_id = tenant_id

    def _send_batch(self, records: List[LogRecord]):
        """Send a batch of records to Loki."""
        if not records:
            return

        # Group records by labels (streams)
        streams: Dict[str, List[tuple]] = {}
        for record in records:
            # Combine default labels with record labels
            labels = {**self.default_labels, **record.labels}
            label_key = json.dumps(labels, sort_keys=True)

            if label_key not in streams:
                streams[label_key] = {'labels': labels, 'entries': []}

            # Loki expects timestamp in nanoseconds
            ts_ns = int(record.timestamp.timestamp() * 1e9)

            # Build log line with extra fields
            log_line = record.message
            if record.extra:
                extra_str = ' ' + json.dumps(record.extra, default=str)
                log_line += extra_str

            streams[label_key]['entries'].append([str(ts_ns), log_line])

        # Build Loki push payload
        payload = {
            'streams': [
                {
                    'stream': stream_data['labels'],
                    'values': stream_data['entries'],
                }
                for stream_data in streams.values()
            ]
        }

        # Send to Loki
        headers = {'Content-Type': 'application/json'}
        if self.tenant_id:
            headers['X-Scope-OrgID'] = self.tenant_id

        response = self._session.post(
            self.push_url,
            json=payload,
            headers=headers,
            auth=self.auth,
            timeout=self.timeout,
        )
        response.raise_for_status()


class ElasticsearchHandler(BaseHandler):
    """
    Handler for sending logs to Elasticsearch.

    Uses the Elasticsearch bulk API for efficient indexing.
    """

    def __init__(
        self,
        url: str,
        index_prefix: str = 'rdt-logs',
        auth: tuple = None,
        api_key: str = None,
        use_data_stream: bool = False,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_buffer_size: int = 10000,
        retry_attempts: int = 3,
        timeout: float = 10.0,
        verify_ssl: bool = True,
    ):
        """
        Initialize the Elasticsearch handler.

        Args:
            url: Elasticsearch URL (e.g., 'http://localhost:9200')
            index_prefix: Prefix for index names (daily indices: prefix-YYYY.MM.DD)
            auth: Optional (username, password) tuple for basic auth
            api_key: Optional API key for authentication
            use_data_stream: Use data streams instead of indices
            batch_size: Number of records to batch before sending
            flush_interval: Seconds between automatic flushes
            max_buffer_size: Maximum records to buffer
            retry_attempts: Number of retry attempts
            timeout: Request timeout in seconds
            verify_ssl: Verify SSL certificates
        """
        super().__init__(
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_buffer_size=max_buffer_size,
            retry_attempts=retry_attempts,
            timeout=timeout,
        )

        self.url = url.rstrip('/')
        self.index_prefix = index_prefix
        self.auth = auth
        self.api_key = api_key
        self.use_data_stream = use_data_stream
        self.verify_ssl = verify_ssl

        # Configure session for SSL
        self._session.verify = verify_ssl

    def _get_index_name(self, timestamp: datetime) -> str:
        """Get the index name for a given timestamp."""
        if self.use_data_stream:
            return self.index_prefix
        date_str = timestamp.strftime('%Y.%m.%d')
        return f"{self.index_prefix}-{date_str}"

    def _send_batch(self, records: List[LogRecord]):
        """Send a batch of records to Elasticsearch using bulk API."""
        if not records:
            return

        # Build bulk request body
        bulk_body = []
        for record in records:
            # Action line
            index_name = self._get_index_name(record.timestamp)
            if self.use_data_stream:
                action = {'create': {'_index': index_name}}
            else:
                action = {'index': {'_index': index_name}}
            bulk_body.append(json.dumps(action))

            # Document
            doc = {
                '@timestamp': record.timestamp.isoformat(),
                'level': record.level,
                'message': record.message,
                **record.labels,
                **record.extra,
            }
            bulk_body.append(json.dumps(doc, default=str))

        # Join with newlines and add trailing newline
        body = '\n'.join(bulk_body) + '\n'

        # Build headers
        headers = {'Content-Type': 'application/x-ndjson'}
        if self.api_key:
            headers['Authorization'] = f'ApiKey {self.api_key}'

        # Send bulk request
        response = self._session.post(
            f'{self.url}/_bulk',
            data=body,
            headers=headers,
            auth=self.auth,
            timeout=self.timeout,
        )
        response.raise_for_status()

        # Check for errors in response
        result = response.json()
        if result.get('errors'):
            # Log individual errors but don't fail the whole batch
            error_count = sum(1 for item in result.get('items', [])
                            if 'error' in item.get('index', item.get('create', {})))
            if error_count > 0:
                raise Exception(f"Elasticsearch bulk indexing had {error_count} errors")


class LoguruLokiSink:
    """
    Loguru sink for Loki integration.

    Usage:
        from loguru import logger
        from logging.handlers import LoguruLokiSink

        sink = LoguruLokiSink('http://localhost:3100')
        sink.start()
        logger.add(sink)
    """

    def __init__(
        self,
        url: str,
        labels: Dict[str, str] = None,
        auth: tuple = None,
        tenant_id: str = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize the Loguru Loki sink.

        Args:
            url: Loki push URL
            labels: Default labels for all log streams
            auth: Optional basic auth tuple
            tenant_id: Optional tenant ID
            batch_size: Records to batch before sending
            flush_interval: Seconds between flushes
        """
        self._handler = LokiHandler(
            url=url,
            labels=labels,
            auth=auth,
            tenant_id=tenant_id,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )

    def start(self):
        """Start the background flush thread."""
        self._handler.start()

    def stop(self):
        """Stop and flush remaining logs."""
        self._handler.close()

    def write(self, message):
        """Write a log message (Loguru sink interface)."""
        record = message.record if hasattr(message, 'record') else {}
        self._handler.emit(record)

    def __call__(self, message):
        """Make the sink callable for Loguru."""
        self.write(message)


class LoguruElasticsearchSink:
    """
    Loguru sink for Elasticsearch integration.

    Usage:
        from loguru import logger
        from logging.handlers import LoguruElasticsearchSink

        sink = LoguruElasticsearchSink('http://localhost:9200')
        sink.start()
        logger.add(sink)
    """

    def __init__(
        self,
        url: str,
        index_prefix: str = 'rdt-logs',
        auth: tuple = None,
        api_key: str = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize the Loguru Elasticsearch sink.

        Args:
            url: Elasticsearch URL
            index_prefix: Prefix for index names
            auth: Optional basic auth tuple
            api_key: Optional API key
            batch_size: Records to batch before sending
            flush_interval: Seconds between flushes
        """
        self._handler = ElasticsearchHandler(
            url=url,
            index_prefix=index_prefix,
            auth=auth,
            api_key=api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )

    def start(self):
        """Start the background flush thread."""
        self._handler.start()

    def stop(self):
        """Stop and flush remaining logs."""
        self._handler.close()

    def write(self, message):
        """Write a log message (Loguru sink interface)."""
        record = message.record if hasattr(message, 'record') else {}
        self._handler.emit(record)

    def __call__(self, message):
        """Make the sink callable for Loguru."""
        self.write(message)


class BufferedFileHandler:
    """
    Buffered file handler for high-throughput logging.

    Writes logs to a file with buffering and rotation support.
    """

    def __init__(
        self,
        filepath: str,
        max_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        flush_interval: float = 1.0,
    ):
        """
        Initialize the buffered file handler.

        Args:
            filepath: Path to the log file
            max_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            flush_interval: Seconds between flushes
        """
        self.filepath = filepath
        self.max_size = max_size
        self.backup_count = backup_count
        self.flush_interval = flush_interval

        self._buffer: queue.Queue = queue.Queue(maxsize=10000)
        self._file = None
        self._current_size = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

        self._open_file()

    def _open_file(self):
        """Open the log file."""
        import os
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        self._file = open(self.filepath, 'a', encoding='utf-8')
        self._current_size = self._file.tell()

    def _rotate_file(self):
        """Rotate the log file."""
        import os
        import shutil

        with self._lock:
            if self._file:
                self._file.close()

            # Rotate existing backups
            for i in range(self.backup_count - 1, 0, -1):
                src = f"{self.filepath}.{i}"
                dst = f"{self.filepath}.{i + 1}"
                if os.path.exists(src):
                    shutil.move(src, dst)

            # Move current file to .1
            if os.path.exists(self.filepath):
                shutil.move(self.filepath, f"{self.filepath}.1")

            # Open new file
            self._open_file()

    def write(self, message):
        """Write a message to the buffer."""
        try:
            self._buffer.put_nowait(str(message))
        except queue.Full:
            pass  # Drop message if buffer is full

    def _flush_loop(self):
        """Background flush loop."""
        while not self._stop_event.is_set():
            self._flush()
            self._stop_event.wait(self.flush_interval)
        self._flush()  # Final flush

    def _flush(self):
        """Flush buffered messages to file."""
        messages = []
        while True:
            try:
                messages.append(self._buffer.get_nowait())
            except queue.Empty:
                break

        if not messages:
            return

        with self._lock:
            for msg in messages:
                line = msg if msg.endswith('\n') else msg + '\n'
                self._file.write(line)
                self._current_size += len(line.encode('utf-8'))

            self._file.flush()

            # Check for rotation
            if self._current_size >= self.max_size:
                self._rotate_file()

    def start(self):
        """Start the background flush thread."""
        self._stop_event.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self):
        """Stop the flush thread."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)

    def close(self):
        """Close the handler."""
        self.stop()
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

    def __call__(self, message):
        """Make the handler callable for Loguru."""
        self.write(message)
