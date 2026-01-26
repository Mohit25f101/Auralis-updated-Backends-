"""
============================================
AURALIS v5.0 - Cache Service
============================================
In-memory caching for analysis results.
"""

from collections import OrderedDict
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading

from utils.logger import logger


class CacheService:
    """
    Simple in-memory cache with TTL and size limits.
    
    Features:
    - LRU eviction
    - TTL expiration
    - Thread-safe
    """
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        """
        Initialize cache service.
        
        Args:
            max_size: Maximum number of items
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            
            # Check expiration
            if datetime.now() > item['expires_at']:
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            return item['data']
    
    def set(self, key: str, value: Dict[str, Any]):
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Evict if full
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            # Add item
            self._cache[key] = {
                'data': value,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + self.ttl
            }
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired items.
        
        Returns:
            Number of items removed
        """
        now = datetime.now()
        removed = 0
        
        with self._lock:
            expired = [
                key for key, item in self._cache.items()
                if now > item['expires_at']
            ]
            
            for key in expired:
                del self._cache[key]
                removed += 1
        
        if removed > 0:
            logger.debug(f"Removed {removed} expired cache items")
        
        return removed
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl.total_seconds(),
        }