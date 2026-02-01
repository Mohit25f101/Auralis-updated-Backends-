# ==============================
# 📄 api/middleware/error_handler.py
# ==============================
"""
Global Error Handling Middleware
Provides comprehensive error handling, logging, and rate limiting
"""

import traceback
import logging
import time
from datetime import datetime
from typing import Callable, Dict, Optional, List
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("auralis")


# ==============================
# CUSTOM EXCEPTIONS
# ==============================

class AuralisException(Exception):
    """Base exception for Auralis"""
    
    def __init__(
        self,
        message: str,
        code: str = "AURALIS_ERROR",
        status_code: int = 500,
        details: Optional[Dict] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class AudioLoadError(AuralisException):
    """Audio loading failed"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="AUDIO_LOAD_ERROR",
            status_code=400,
            details=details
        )


class TranscriptionError(AuralisException):
    """Transcription failed"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="TRANSCRIPTION_ERROR",
            status_code=500,
            details=details
        )


class AnalysisError(AuralisException):
    """Analysis failed"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="ANALYSIS_ERROR",
            status_code=500,
            details=details
        )


class ModelNotLoadedError(AuralisException):
    """Model not loaded"""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' is not loaded or unavailable",
            code="MODEL_NOT_LOADED",
            status_code=503,
            details={"model": model_name}
        )


class ValidationError(AuralisException):
    """Validation failed"""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field} if field else {}
        )


class SemanticAnalysisError(AuralisException):
    """Semantic analysis failed"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="SEMANTIC_ANALYSIS_ERROR",
            status_code=500,
            details=details
        )


class EntityExtractionError(AuralisException):
    """Entity extraction failed"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="ENTITY_EXTRACTION_ERROR",
            status_code=500,
            details=details
        )


class RateLimitExceeded(AuralisException):
    """Rate limit exceeded"""
    
    def __init__(self, limit: int, window: int):
        super().__init__(
            message=f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"limit": limit, "window_seconds": window}
        )


class FileTooLargeError(AuralisException):
    """File too large"""
    
    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(
            message=f"File too large ({size_mb:.1f}MB). Maximum allowed: {max_mb}MB",
            code="FILE_TOO_LARGE",
            status_code=413,
            details={"size_mb": size_mb, "max_mb": max_mb}
        )


class UnsupportedFormatError(AuralisException):
    """Unsupported file format"""
    
    def __init__(self, format: str, supported: List[str]):
        super().__init__(
            message=f"Unsupported format: {format}. Supported: {', '.join(supported)}",
            code="UNSUPPORTED_FORMAT",
            status_code=400,
            details={"format": format, "supported": supported}
        )


# ==============================
# ERROR HANDLER MIDDLEWARE
# ==============================

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors globally
    
    Features:
    - Catches all exceptions
    - Formats error responses consistently
    - Logs errors with context
    - Generates unique error IDs
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            # Re-raise HTTP exceptions (handled by FastAPI)
            raise exc
            
        except AuralisException as exc:
            # Handle Auralis-specific exceptions
            error_id = self._generate_error_id()
            
            logger.error(
                f"Auralis Error [{error_id}]: {exc.code} - {exc.message}",
                extra={"error_id": error_id, "details": exc.details}
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "error_id": error_id,
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as exc:
            # Handle unexpected exceptions
            error_id = self._generate_error_id()
            
            logger.error(
                f"Unexpected Error [{error_id}]: {str(exc)}",
                extra={"error_id": error_id}
            )
            logger.error(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_id": error_id,
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please try again.",
                    "details": {"type": type(exc).__name__},
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking"""
        import uuid
        return f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"


# ==============================
# REQUEST LOGGING MIDDLEWARE
# ==============================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all requests
    
    Logs:
    - Request method and path
    - Response status code
    - Processing time
    - Client IP
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={
                "client_ip": client_ip,
                "method": request.method,
                "path": request.url.path
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            f"← {request.method} {request.url.path} [{response.status_code}] {duration_ms:.1f}ms",
            extra={
                "client_ip": client_ip,
                "status_code": response.status_code,
                "duration_ms": duration_ms
            }
        )
        
        # Add processing time header
        response.headers["X-Processing-Time-Ms"] = f"{duration_ms:.1f}"
        
        return response


# ==============================
# RATE LIMIT MIDDLEWARE
# ==============================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware
    
    Features:
    - Per-IP rate limiting
    - Configurable limits and windows
    - Automatic cleanup of old entries
    """
    
    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        exclude_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > 60:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Get request times for this IP
        request_times = self.requests[client_ip]
        
        # Remove old requests outside window
        request_times = [
            t for t in request_times
            if t > current_time - self.window_seconds
        ]
        self.requests[client_ip] = request_times
        
        # Check rate limit
        if len(request_times) >= self.max_requests:
            logger.warning(
                f"Rate limit exceeded for {client_ip}",
                extra={"client_ip": client_ip, "requests": len(request_times)}
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Too many requests. Limit: {self.max_requests} per {self.window_seconds}s",
                    "retry_after_seconds": self.window_seconds,
                    "timestamp": datetime.now().isoformat()
                },
                headers={"Retry-After": str(self.window_seconds)}
            )
        
        # Add current request
        request_times.append(current_time)
        self.requests[client_ip] = request_times
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(self.max_requests - len(request_times))
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)
        
        return response
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old entries to prevent memory growth"""
        cutoff = current_time - self.window_seconds
        
        ips_to_remove = []
        for ip, times in self.requests.items():
            # Remove old times
            self.requests[ip] = [t for t in times if t > cutoff]
            
            # Mark empty entries for removal
            if not self.requests[ip]:
                ips_to_remove.append(ip)
        
        # Remove empty entries
        for ip in ips_to_remove:
            del self.requests[ip]


# ==============================
# EXCEPTION HANDLERS SETUP
# ==============================

def setup_exception_handlers(app: FastAPI):
    """
    Setup exception handlers for the FastAPI app
    
    Handles:
    - AuralisException and subclasses
    - HTTPException
    - ValueError
    - Generic exceptions
    """
    
    @app.exception_handler(AuralisException)
    async def auralis_exception_handler(request: Request, exc: AuralisException):
        error_id = f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.error(f"Auralis Error [{error_id}]: {exc.code} - {exc.message}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "error_id": error_id,
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "path": str(request.url),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "path": str(request.url),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"Validation Error: {str(exc)}")
        
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "code": "VALIDATION_ERROR",
                "message": str(exc),
                "path": str(request.url),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        import uuid
        error_id = f"ERR-{uuid.uuid4().hex[:8].upper()}"
        
        logger.error(f"Unhandled Exception [{error_id}]: {str(exc)}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_id": error_id,
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred. Please try again later.",
                "path": str(request.url),
                "timestamp": datetime.now().isoformat()
            }
        )


# ==============================
# CORS PREFLIGHT HANDLER
# ==============================

class CORSPreflightMiddleware(BaseHTTPMiddleware):
    """Handle CORS preflight requests"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        if request.method == "OPTIONS":
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        return await call_next(request)