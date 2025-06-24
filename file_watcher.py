import asyncio
import logging
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable
from config import config

logger = logging.getLogger(__name__)

class DocumentChangeHandler(FileSystemEventHandler):
    def __init__(self, rebuild_callback: Callable):
        self.rebuild_callback = rebuild_callback
        self.debounce_timer = None
        self.debounce_delay = 2.0  # Wait 2 seconds after last change before rebuilding
        self.lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logger.info(f"New document detected: {event.src_path}")
            self._schedule_rebuild()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logger.info(f"Document modified: {event.src_path}")
            self._schedule_rebuild()

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logger.info(f"Document deleted: {event.src_path}")
            self._schedule_rebuild()

    def _schedule_rebuild(self):
        """Schedule a rebuild with debouncing to avoid multiple rebuilds for rapid changes."""
        with self.lock:
            if self.debounce_timer and self.debounce_timer.is_alive():
                self.debounce_timer.cancel()
            
            self.debounce_timer = threading.Timer(self.debounce_delay, self._trigger_rebuild)
            self.debounce_timer.start()

    def _trigger_rebuild(self):
        """Trigger the rebuild callback from the timer thread."""
        try:
            logger.info("Triggering vector store rebuild due to document changes...")
            # Create a new event loop for the callback if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async callback
            if asyncio.iscoroutinefunction(self.rebuild_callback):
                loop.run_until_complete(self.rebuild_callback())
            else:
                self.rebuild_callback()
        except Exception as e:
            logger.error(f"Error during delayed rebuild: {str(e)}")

class DocumentWatcher:
    def __init__(self, docs_dir: str, rebuild_callback: Callable):
        self.docs_dir = docs_dir
        self.rebuild_callback = rebuild_callback
        self.observer = None
        self.handler = None

    def start(self):
        """Start watching the documents directory."""
        if not os.path.exists(self.docs_dir):
            logger.warning(f"Documents directory {self.docs_dir} does not exist. File watching disabled.")
            return

        self.handler = DocumentChangeHandler(self.rebuild_callback)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.docs_dir, recursive=True)
        self.observer.start()
        logger.info(f"Started watching {self.docs_dir} for document changes")

    def stop(self):
        """Stop watching the documents directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped document file watcher")

    def is_alive(self) -> bool:
        """Check if the observer is still running."""
        return self.observer and self.observer.is_alive()