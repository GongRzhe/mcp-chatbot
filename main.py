import asyncio
import json
import logging
import os
import random
import shutil
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Literal, Union, Tuple

import aiohttp  # Replace requests with aiohttp for true async HTTP
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Set up more detailed logging for debugging API responses
# Uncomment this line to enable debug logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ServerState(Enum):
    """Enum representing server connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


class CircuitBreaker:
    """Implements the Circuit Breaker pattern to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 30) -> None:
        """Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_time: Time in seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def execute(self, coro):
        """Execute the coroutine if the circuit is not open.
        
        Args:
            coro: Coroutine to execute
            
        Returns:
            Result of the coroutine or raises exception
            
        Raises:
            CircuitBreakerOpenError: If the circuit is open
        """
        async with self._lock:
            if self.state == "OPEN":
                # Check if recovery time has elapsed
                if time.time() - self.last_failure_time >= self.recovery_time:
                    self.state = "HALF_OPEN"
                    logging.info("Circuit half-open, attempting recovery...")
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            try:
                result = await coro
                
                # Reset failure count on success
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logging.info("Circuit closed, service recovered successfully")
                
                return result
            
            except Exception as e:
                # Update failure stats
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Open circuit if threshold reached
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logging.warning(f"Circuit opened after {self.failure_count} failures")
                
                raise e


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        # API keys for different LLM providers
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.openroute_api_key = os.getenv("OPENROUTE_API_KEY")
        self.ollama_host = "http://127.0.0.1:11434"
        self.github_api_key = os.getenv("GITHUB_API_KEY")
        
        # Default LLM provider and model
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "llama-3.2-90b-vision-preview")
        
        # Connection settings
        self.connection_timeout = int(os.getenv("CONNECTION_TIMEOUT", "10"))
        self.read_timeout = int(os.getenv("READ_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay_base = float(os.getenv("RETRY_DELAY_BASE", "1.0"))
        self.retry_max_delay = float(os.getenv("RETRY_MAX_DELAY", "30.0"))
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing server configuration.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_api_key(self, provider: str) -> str:
        """Get the API key for the specified provider.
        
        Args:
            provider: The LLM provider name.
            
        Returns:
            The API key as a string.
            
        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        api_key_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
            "openroute": self.openroute_api_key,
            "github": self.github_api_key,
        }
        
        api_key = api_key_map.get(provider.lower())
        if not api_key and provider.lower() != "ollama":
            raise ValueError(f"{provider.upper()}_API_KEY not found in environment variables")
        return api_key


class AsyncRetry:
    """Handles asynchronous retry logic with exponential backoff and jitter."""
    
    def __init__(
        self, 
        max_retries: int, 
        base_delay: float, 
        max_delay: float,
        jitter: bool = True
    ) -> None:
        """Initialize retry parameters.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter: Whether to add randomness to delay to prevent thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    async def execute(self, coro, *args, **kwargs):
        """Execute a coroutine with retries.
        
        Args:
            coro: Coroutine function to retry
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine
            
        Returns:
            Result from the coroutine
            
        Raises:
            Exception: Last exception raised after all retries fail
        """
        attempt = 0
        last_exception = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    logging.info(f"Retry attempt {attempt}/{self.max_retries}")
                    
                return await coro(*args, **kwargs)
                
            except Exception as e:
                attempt += 1
                last_exception = e
                
                if attempt > self.max_retries:
                    logging.error(f"Max retries ({self.max_retries}) exceeded")
                    raise
                
                # Calculate exponential backoff delay
                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                
                # Add jitter if enabled
                if self.jitter:
                    delay = delay * (0.5 + random.random())
                
                logging.warning(f"Operation failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # This should never happen but just in case
        raise last_exception if last_exception else RuntimeError("Retry failed for unknown reason")


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any], global_config: Configuration) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.global_config: Configuration = global_config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None
        self.state: ServerState = ServerState.DISCONNECTED
        self.last_connection_attempt: float = 0
        self.connection_failures: int = 0
        self.max_consecutive_failures: int = 3
        self.reconnect_delay: float = 5.0
        self.tools: List[Any] = []
        self.circuit_breaker = CircuitBreaker()
        self.retry_handler = AsyncRetry(
            max_retries=global_config.max_retries,
            base_delay=global_config.retry_delay_base,
            max_delay=global_config.retry_max_delay
        )
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """Initialize the server connection with reconnection logic.
        
        Returns:
            bool: True if connection succeeded, False otherwise
        """
        if self.state == ServerState.CONNECTING:
            logging.info(f"Server {self.name} already connecting, waiting...")
            return False
            
        self.state = ServerState.CONNECTING
        self.last_connection_attempt = time.time()
        
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        
        try:
            # Use circuit breaker pattern for initialization
            await self.circuit_breaker.execute(self._initialize_internal(server_params))
            
            # Connection successful, reset failure counter
            self.connection_failures = 0
            self.state = ServerState.CONNECTED
            
            # Start health check task
            self._start_health_check()
            
            return True
            
        except Exception as e:
            self.connection_failures += 1
            logging.error(f"Error initializing server {self.name}: {e}")
            self.state = ServerState.FAILED
            await self.cleanup()
            return False

    async def _initialize_internal(self, server_params: StdioServerParameters) -> None:
        """Internal initialization function wrapped by circuit breaker.
        
        Args:
            server_params: Server parameters for initialization
        """
        logging.info(f"Connecting to server {self.name}...")
        self.stdio_context = stdio_client(server_params)
        read, write = await self.stdio_context.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        self.capabilities = await self.session.initialize()
        logging.info(f"Server {self.name} connected successfully")
        
        # Fetch tools immediately after successful connection
        await self._update_tools()

    async def _update_tools(self) -> None:
        """Update the list of available tools from the server."""
        if not self.session:
            return
            
        try:
            tools_response = await self.session.list_tools()
            self.tools = []
            
            supports_progress = (
                self.capabilities 
                and 'progress' in self.capabilities
            )
            
            if supports_progress:
                logging.info(f"Server {self.name} supports progress tracking")
            
            for item in tools_response:
                if isinstance(item, tuple) and item[0] == 'tools':
                    for tool in item[1]:
                        self.tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                        if supports_progress:
                            logging.info(f"Tool '{tool.name}' will support progress tracking")
        except Exception as e:
            logging.error(f"Error updating tools for server {self.name}: {e}")

    def _start_health_check(self) -> None:
        """Start periodic health check task."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def _health_check_loop(self) -> None:
        """Periodic health check to ensure server connection is still alive."""
        while True:
            try:
                await asyncio.sleep(self.global_config.health_check_interval)
                if self.state == ServerState.CONNECTED:
                    # Ping the server by listing tools
                    await self._update_tools()
                    logging.debug(f"Health check passed for server {self.name}")
                elif self.state in [ServerState.DISCONNECTED, ServerState.FAILED]:
                    # Try to reconnect if disconnected
                    if time.time() - self.last_connection_attempt > self.reconnect_delay:
                        logging.info(f"Attempting to reconnect to server {self.name}...")
                        await self.initialize()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"Health check failed for server {self.name}: {e}")
                if self.state == ServerState.CONNECTED:
                    self.state = ServerState.FAILED
                    await self.cleanup()
                    # Schedule reconnection
                    self.last_connection_attempt = time.time()

    async def list_tools(self) -> List[Any]:
        """List available tools from the server with reconnection logic.
        
        Returns:
            A list of available tools.
        """
        # If disconnected, try to reconnect
        if self.state != ServerState.CONNECTED:
            if time.time() - self.last_connection_attempt > self.reconnect_delay:
                reconnected = await self.initialize()
                if not reconnected:
                    return []
            else:
                return []
        
        # Return cached tools list to avoid frequent server calls
        return self.tools

    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool with advanced retry and reconnection logic.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            
        Returns:
            Tool execution result.
            
        Raises:
            RuntimeError: If server is not initialized or tool execution fails.
        """
        # If not connected, try to reconnect
        if self.state != ServerState.CONNECTED:
            reconnected = await self.initialize()
            if not reconnected:
                raise RuntimeError(f"Cannot execute tool: server {self.name} is not connected")
        
        # Use circuit breaker pattern to prevent cascading failures
        try:
            return await self.circuit_breaker.execute(
                self.retry_handler.execute(self._execute_tool_internal, tool_name, arguments)
            )
        except Exception as e:
            logging.error(f"Failed to execute tool {tool_name} after retries: {e}")
            # Mark server as failed if execution consistently fails
            self.state = ServerState.FAILED
            await self.cleanup()
            self.last_connection_attempt = time.time()
            raise RuntimeError(f"Tool execution failed: {str(e)}")

    async def _execute_tool_internal(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal tool execution function wrapped by retry handler.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            
        Returns:
            Tool execution result.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )

        if supports_progress:
            logging.info(f"Executing {tool_name} with progress tracking...")
            result = await self.session.call_tool(
                tool_name, 
                arguments,
                progress_token=f"{tool_name}_execution"
            )
        else:
            logging.info(f"Executing {tool_name}...")
            result = await self.session.call_tool(tool_name, arguments)

        return result

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            # Cancel health check task
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
                self._health_check_task = None
                
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
                        
                if self.state != ServerState.DISCONNECTED:
                    self.state = ServerState.DISCONNECTED
                    
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.
        
        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with multiple LLM providers using async HTTP."""
    
    # Default provider configurations for API endpoints and models
    PROVIDER_CONFIGS = {
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "models_url": "https://api.groq.com/openai/v1/models",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}"
            },
            "models_extractor": lambda resp: [model["id"] for model in resp.get("data", [])],
            "default_models": [
                "llama-3.2-90b-vision-preview",
                "llama-3.2-70b-instruct-preview"
            ]
        },
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "models_url": "https://api.openai.com/v1/models",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}"
            },
            "models_extractor": lambda resp: [m["id"] for m in resp.get("data", []) if "gpt" in m["id"].lower()],
            "default_models": [
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ]
        },
        "anthropic": {
            "url": "https://api.anthropic.com/v1/messages",
            "models_url": "https://api.anthropic.com/v1/models",
            "headers": lambda api_key: {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            "models_extractor": lambda resp: [model["id"] for model in resp.get("data", [])],
            "default_models": [
                "claude-3-5-sonnet-20240620",
                "claude-3-opus-20240229"
            ]
        },
        "gemini": {
            "url": "https://gemini.gongrzhe.us.kg/v1/chat/completions",
            "models_url": "https://gemini.gongrzhe.us.kg/v1/models",
            "default_models": [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro"
            ],
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}"
            },
            "models_extractor": lambda resp: [model["id"] for model in resp],
        },
        "openroute": {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "models_url": "https://openrouter.ai/api/v1/models",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "MCP-Chatbot"
            },
            "models_extractor": lambda resp: [model["id"] for model in resp],
            "default_models": [
                "openai/gpt-4o",
                "anthropic/claude-3-5-sonnet",
                "meta-llama/llama-3.2-70b-instruct"
            ]
        },
        "ollama": {
            "url": None,  # Set dynamically
            "models_url": None,  # Set dynamically
            "models_extractor": lambda resp: [model["name"] for model in resp],
            "default_models": [
                "llama3",
                "mistral",
                "mixtral"
            ]
        }
    }

    def __init__(
        self, 
        config: Configuration,
        provider: str = None,
        model: str = None
    ) -> None:
        self.config = config
        self.provider = provider or config.default_provider
        self.model = model or config.default_model
        self.api_key = config.get_api_key(self.provider) if self.provider != "ollama" else None
        
        # Store available models for each provider
        self.available_models = {}
        
        # Create session for HTTP requests
        self.session = None
        self.session_lock = asyncio.Lock()
        
        # Create retry handler
        self.retry_handler = AsyncRetry(
            max_retries=config.max_retries,
            base_delay=config.retry_delay_base,
            max_delay=config.retry_max_delay
        )
        
        # Create circuit breakers for each provider
        self.circuit_breakers = {
            provider: CircuitBreaker() 
            for provider in self.PROVIDER_CONFIGS.keys()
        }
        
        # Health check state
        self.provider_health = {provider: True for provider in self.PROVIDER_CONFIGS.keys()}
        self._health_check_task = None
                
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists and create if needed.
        
        Returns:
            aiohttp.ClientSession: The HTTP session
        """
        async with self.session_lock:
            if self.session is None or self.session.closed:
                # Configure timeout and connection limits
                timeout = aiohttp.ClientTimeout(
                    connect=self.config.connection_timeout,
                    total=self.config.read_timeout
                )
                connector = aiohttp.TCPConnector(
                    limit=20,  # Connection pool size
                    ttl_dns_cache=300,  # DNS cache TTL
                    enable_cleanup_closed=True
                )
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    raise_for_status=True
                )
                
        return self.session

    async def initialize(self) -> None:
        """Initialize the client and fetch initial model information."""
        await self._ensure_session()
        
        # Start health checks
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Fetch models for all providers in parallel
        fetch_tasks = []
        for provider_name in self.PROVIDER_CONFIGS.keys():
            # Only try to fetch if we have an API key (except for Ollama)
            if provider_name == "ollama" or self.config.get_api_key(provider_name) is not None:
                task = asyncio.create_task(self._fetch_provider_models(provider_name))
                fetch_tasks.append(task)
        
        # Wait for all fetches to complete (don't raise exceptions)
        if fetch_tasks:
            await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
        # Ensure we have models for the current provider
        if self.provider not in self.available_models:
            self.available_models[self.provider] = self.PROVIDER_CONFIGS[self.provider]["default_models"]
            
    async def _fetch_provider_models(self, provider: str) -> None:
        """Fetch models for a specific provider with error handling.
        
        Args:
            provider: The provider to fetch models for
        """
        try:
            await self._fetch_available_models(provider)
            logging.info(f"Successfully loaded {len(self.available_models[provider])} models for {provider}")
        except Exception as e:
            logging.warning(f"Could not fetch models for {provider}: {e}")
            # Use default models as fallback
            self.available_models[provider] = self.PROVIDER_CONFIGS[provider]["default_models"]
    
    async def _health_check_loop(self) -> None:
        """Periodically check the health of LLM providers."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Create tasks to check all providers in parallel
                health_check_tasks = []
                
                # Check current provider first with higher priority
                check_task = asyncio.create_task(
                    self._check_provider_health(self.provider)
                )
                health_check_tasks.append(check_task)
                
                # Check other providers in the background
                for provider in self.PROVIDER_CONFIGS.keys():
                    if provider != self.provider:
                        # Only check if we have an API key
                        if provider == "ollama" or self.config.get_api_key(provider):
                            task = asyncio.create_task(
                                self._check_provider_health(provider)
                            )
                            health_check_tasks.append(task)
                
                # Wait for all health checks to complete
                await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                # Log overall health status
                healthy_providers = [p for p, status in self.provider_health.items() if status]
                logging.debug(f"Provider health status: {len(healthy_providers)}/{len(self.provider_health)} healthy")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying on error
    
    async def _check_provider_health(self, provider: str) -> None:
        """Check health of a specific provider.
        
        Args:
            provider: The provider to check
        """
        try:
            # For health checks, we don't need to store the models
            # Just check if the API is responsive
            await self._fetch_available_models(provider)
            
            # If we made it here, the provider is healthy
            was_unhealthy = not self.provider_health.get(provider, True)
            self.provider_health[provider] = True
            
            # Log recovery if the provider was previously unhealthy
            if was_unhealthy:
                logging.info(f"Provider {provider} has recovered and is now healthy")
                
        except Exception as e:
            # Provider is unhealthy
            was_healthy = self.provider_health.get(provider, True)
            self.provider_health[provider] = False
            
            # Only log warning if this is a new failure
            if was_healthy:
                logging.warning(f"Health check failed for provider {provider}: {e}")
            else:
                logging.debug(f"Provider {provider} still unhealthy: {e}")
                
    async def _fetch_available_models(self, provider: str) -> List[str]:
        """Fetch available models from the provider's API asynchronously.
        
        Args:
            provider: The provider to fetch models for
            
        Returns:
            A list of available model IDs
            
        Raises:
            ValueError: If the provider is not supported
            aiohttp.ClientError: If the API request fails
        """
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Get provider configuration
        config = self.PROVIDER_CONFIGS[provider]
        
        # For Google, we don't have a models endpoint, use defaults
        # if provider == "google":
        #     logging.info("Google doesn't have a models API, using default models")
        #     self.available_models[provider] = config["default_models"]
        #     return self.available_models[provider]
        
        # Specialized handlers for each provider
        if provider == "groq":
            return await self._fetch_groq_models(config)
        elif provider == "openai":
            return await self._fetch_openai_models(config)
        elif provider == "gemini":
            return await self._fetch_gemini_models(config)
        elif provider == "anthropic":
            return await self._fetch_anthropic_models(config)
        elif provider == "openroute":
            return await self._fetch_openroute_models(config)
        elif provider == "ollama":
            return await self._fetch_ollama_models(config)
        else:
            # Generic fallback for any other providers
            return await self._fetch_generic_models(provider, config)
    
    async def _fetch_groq_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Groq models with specialized handling."""
        try:
            api_key = self.config.get_api_key("groq")
            if not api_key:
                self.available_models["groq"] = config["default_models"]
                return config["default_models"]
                
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers) as response:
                data = await response.json()
                logging.debug(f"Groq API response: {data}")
                
                if "data" in data and isinstance(data["data"], list):
                    models = [model["id"] for model in data["data"] if "id" in model]
                    
                    if models:
                        self.available_models["groq"] = models
                        return models
                
                logging.warning("Could not extract models from Groq API response")
                self.available_models["groq"] = config["default_models"]
                return config["default_models"]
                
        except Exception as e:
            logging.warning(f"Error fetching Groq models: {e}")
            self.available_models["groq"] = config["default_models"]
            return config["default_models"]
    
    async def _fetch_openai_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch OpenAI models with specialized handling."""
        try:
            api_key = self.config.get_api_key("openai")
            if not api_key:
                self.available_models["openai"] = config["default_models"]
                return config["default_models"]
                
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers) as response:
                data = await response.json()
                logging.debug(f"OpenAI API response: {data}")
                
                if "data" in data and isinstance(data["data"], list):
                    # Filter for only GPT models
                    models = [m["id"] for m in data["data"] if "id" in m and "gpt" in m["id"].lower()]
                    
                    if models:
                        self.available_models["openai"] = models
                        return models
                
                logging.warning("Could not extract models from OpenAI API response")
                self.available_models["openai"] = config["default_models"]
                return config["default_models"]
                
        except Exception as e:
            logging.warning(f"Error fetching OpenAI models: {e}")
            self.available_models["openai"] = config["default_models"]
            return config["default_models"]
        
    async def _fetch_gemini_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch gemini models with specialized handling."""
        try:
            api_key = self.config.get_api_key("gemini")
            if not api_key:
                self.available_models["gemini"] = config["default_models"]
                return config["default_models"]
                
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers) as response:
                data = await response.json()
                logging.debug(f"gemini API response: {data}")
                
                if "data" in data and isinstance(data["data"], list):
                    # Filter for only GPT models
                    models = [m["id"] for m in data["data"] if "id" in m and "gemini" in m["id"].lower()]
                    
                    if models:
                        self.available_models["gemini"] = models
                        return models
                
                logging.warning("Could not extract models from gemini API response")
                self.available_models["gemini"] = config["default_models"]
                return config["default_models"]
                
        except Exception as e:
            logging.warning(f"Error fetching gemini models: {e}")
            self.available_models["gemini"] = config["default_models"]
            return config["default_models"]
    
    async def _fetch_anthropic_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Anthropic models with specialized handling."""
        try:
            api_key = self.config.get_api_key("anthropic")
            if not api_key:
                self.available_models["anthropic"] = config["default_models"]
                return config["default_models"]
                
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                data = await response.json()
                logging.debug(f"Anthropic API response: {data}")
                
                if "data" in data and isinstance(data["data"], list):
                    models = [model["id"] for model in data["data"] if "id" in model]
                    
                    if models:
                        self.available_models["anthropic"] = models
                        return models
                
                logging.warning("Could not extract models from Anthropic API response")
                self.available_models["anthropic"] = config["default_models"]
                return config["default_models"]
                
        except Exception as e:
            logging.warning(f"Error fetching Anthropic models: {e}")
            self.available_models["anthropic"] = config["default_models"]
            return config["default_models"]

    async def _fetch_openroute_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch OpenRoute models with specialized handling."""
        try:
            api_key = self.config.get_api_key("openroute")
            if not api_key:
                self.available_models["openroute"] = config["default_models"]
                return config["default_models"]
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "MCP-Chatbot"
            }
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                data = await response.json()
                logging.debug(f"OpenRoute API response type: {type(data)}")
                logging.debug(f"OpenRoute API response: {data}")
                
                models = []
                # Handle different possible response formats
                if isinstance(data, list):
                    # Direct array of models
                    for model in data:
                        if isinstance(model, dict) and "id" in model:
                            models.append(model["id"])
                elif isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        # Models in a "data" key
                        for model in data["data"]:
                            if isinstance(model, dict) and "id" in model:
                                models.append(model["id"])
                    elif "models" in data and isinstance(data["models"], list):
                        # Models in a "models" key
                        for model in data["models"]:
                            if isinstance(model, dict) and "id" in model:
                                models.append(model["id"])
                
                if models:
                    self.available_models["openroute"] = models
                    return models
                
                logging.warning("Could not extract models from OpenRoute API response")
                self.available_models["openroute"] = config["default_models"]
                return config["default_models"]
        except Exception as e:
            logging.warning(f"Error fetching OpenRoute models: {e}")
            self.available_models["openroute"] = config["default_models"]
            return config["default_models"]

    async def _fetch_google_models(self, config: Dict[str, Any]) -> List[str]:
        """Fetch Google models with specialized handling."""
        try:
            api_key = self.config.get_api_key("google")
            if not api_key:
                self.available_models["google"] = config["default_models"]
                return config["default_models"]
                
            # For Google, we need to use a specific request format
            models_url = f"{config['models_url']}?key={api_key}"
            
            session = await self._ensure_session()
            try:
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    data = await response.json()
                    logging.debug(f"Google API response: {data}")
                    
                    models = []
                    
                    # Extract model IDs based on Google's API format
                    if "models" in data and isinstance(data["models"], list):
                        for model in data["models"]:
                            if isinstance(model, dict) and "name" in model:
                                # Extract the last part of the name path
                                name_parts = model["name"].split("/")
                                model_id = name_parts[-1]
                                models.append(model_id)
                    
                    # Check for non-standard response formats
                    elif "modelCard" in data and isinstance(data["modelCard"], list):
                        for card in data["modelCard"]:
                            if isinstance(card, dict) and "name" in card:
                                models.append(card["name"])
                    
                    if models:
                        self.available_models["google"] = models
                        return models
                        
            except Exception as e:
                logging.warning(f"Error with Google models API: {e}")
            
            # If we couldn't get the models, use defaults
            logging.warning("Using default models for Google (models API may not be available)")
            self.available_models["google"] = config["default_models"]
            return config["default_models"]
                
        except Exception as e:
            logging.warning(f"Error fetching Google models: {e}")
            self.available_models["google"] = config["default_models"]
            return config["default_models"]

    async def _fetch_ollama_models(self, config: Dict[str, Any]) -> List[str]:
        """
        Fetch Ollama models from API with improved error handling and response processing.
        
        According to official Ollama API documentation, /api/tags is the correct endpoint
        to list locally available models. This implementation also handles alternate URL
        formats in case of custom configurations.
        
        Args:
            config: Configuration dictionary containing default models and settings
            
        Returns:
            List of available Ollama model names
        """
        # Validate configuration
        default_models = config.get("default_models", [])
        if not isinstance(default_models, list):
            logging.error("Invalid default_models in config (not a list)")
            return []
        
        # Get and normalize Ollama host URL
        ollama_host = self.config.ollama_host or "http://localhost:11434"
        ollama_host = ollama_host.rstrip('/')
        logging.debug(f"Base Ollama host URL: {ollama_host}")
        
        # Define potential API endpoints to try
        standard_endpoint = f"{ollama_host}/api/tags"
        alternate_endpoint = f"{ollama_host}/v1/api/tags"
        
        # Get session with proper error handling
        try:
            session = await self._ensure_session()
        except Exception as e:
            logging.error(f"Failed to create HTTP session: {e}")
            self.available_models["ollama"] = default_models
            return default_models
        
        # Define timeout
        timeout = aiohttp.ClientTimeout(total=15)
        
        # Try standard endpoint first
        logging.debug(f"Attempting to fetch models from standard endpoint: {standard_endpoint}")
        try:
            return await self._try_fetch_from_endpoint(session, standard_endpoint, timeout, default_models)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logging.info(f"Standard endpoint returned 404, trying alternate endpoint with /v1/ prefix")
                try:
                    return await self._try_fetch_from_endpoint(session, alternate_endpoint, timeout, default_models)
                except Exception as e2:
                    logging.error(f"Failed to fetch models from alternate endpoint: {e2}")
            else:
                logging.error(f"HTTP error with Ollama API: {e.status}, message='{e.message}'")
        except Exception as e:
            logging.error(f"Error fetching models from standard endpoint: {e}")
        
        # If all endpoints failed, use defaults
        logging.warning("All Ollama API endpoints failed, using default models list")
        self.available_models["ollama"] = default_models
        return default_models

    async def _try_fetch_from_endpoint(self, session, endpoint, timeout, default_models):
        """
        Helper method to try fetching models from a specific endpoint.
        
        Args:
            session: HTTP client session
            endpoint: API endpoint URL
            timeout: Request timeout
            default_models: Default models to use if request fails
            
        Returns:
            List of model names
            
        Raises:
            Exception: If the request fails in any way
        """
        async with session.get(endpoint, timeout=timeout) as response:
            response.raise_for_status()
            
            # Parse JSON response
            data = await response.json()
            logging.debug(f"Ollama API response from {endpoint}: {data}")
            
            # Extract model names from the response
            models = []
            
            if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                # Process API format
                for model in data["models"]:
                    if isinstance(model, dict) and "name" in model:
                        models.append(model["name"])
            
            if models:
                logging.info(f"Successfully fetched {len(models)} models from {endpoint}")
                self.available_models["ollama"] = models
                return models
            else:
                logging.warning(f"No models found in response from {endpoint}")
                self.available_models["ollama"] = default_models
                return default_models
        
    async def _fetch_generic_models(self, provider: str, config: Dict[str, Any]) -> List[str]:
        """Generic model fetching for other providers."""
        try:
            api_key = self.config.get_api_key(provider)
            if not api_key:
                self.available_models[provider] = config["default_models"]
                return config["default_models"]
                
            # Prepare headers using the config's header function
            headers = config["headers"](api_key)
            
            session = await self._ensure_session()
            async with session.get(config["models_url"], headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                data = await response.json()
                logging.debug(f"{provider.capitalize()} API response: {data}")
                
                # Use the provider's extractor function to get models
                try:
                    models = config["models_extractor"](data)
                    
                    if models:
                        self.available_models[provider] = models
                        return models
                except Exception as e:
                    logging.warning(f"Error extracting models for {provider}: {e}")
                
                # Fallback to defaults
                self.available_models[provider] = config["default_models"]
                return config["default_models"]
        except Exception as e:
            logging.warning(f"Error fetching models for {provider}: {e}")
            self.available_models[provider] = config["default_models"]
            return config["default_models"]

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM asynchronously.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
        """
        # Check if we need to reconnect to a different provider
        if not self.provider_health[self.provider]:
            logging.warning(f"Current provider {self.provider} is unhealthy")
            
            # Find a healthy alternative provider
            for alt_provider, is_healthy in self.provider_health.items():
                if is_healthy and (alt_provider == "ollama" or self.config.get_api_key(alt_provider)):
                    logging.info(f"Switching to healthy provider: {alt_provider}")
                    await self.change_provider(alt_provider)
                    break
        
        # Use retry mechanism with circuit breaker for resiliency
        try:
            circuit_breaker = self.circuit_breakers[self.provider]
            
            return await circuit_breaker.execute(
                self.retry_handler.execute(self._get_provider_response, messages)
            )
        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            
            # Mark provider as unhealthy
            self.provider_health[self.provider] = False
            
            # Try to find a fallback provider
            for alt_provider, is_healthy in self.provider_health.items():
                if is_healthy and alt_provider != self.provider and (
                    alt_provider == "ollama" or self.config.get_api_key(alt_provider)
                ):
                    logging.info(f"Falling back to alternative provider: {alt_provider}")
                    try:
                        await self.change_provider(alt_provider)
                        return await self.retry_handler.execute(self._get_provider_response, messages)
                    except Exception as fallback_error:
                        logging.error(f"Fallback to {alt_provider} also failed: {fallback_error}")
                        break
            
            return f"I encountered an error connecting to the language model service. Please try again in a moment. (Error: {error_message})"

    async def _get_provider_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the current provider.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
        """
        try:
            if self.provider.lower() == "groq":
                return await self._call_groq(messages)
            elif self.provider.lower() == "openai":
                return await self._call_openai(messages)
            elif self.provider.lower() == "anthropic":
                return await self._call_anthropic(messages)
            elif self.provider.lower() == "google":
                return await self._call_google(messages)
            elif self.provider.lower() == "openroute":
                return await self._call_openroute(messages)
            elif self.provider.lower() == "ollama":
                return await self._call_ollama(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except aiohttp.ClientError as e:
            error_message = f"HTTP error: {str(e)}"
            logging.error(error_message)
            
            # Add more detailed error information if available
            if hasattr(e, 'status'):
                status_code = e.status
                logging.error(f"Status code: {status_code}")
            
            if hasattr(e, 'message'):
                logging.error(f"Response details: {e.message}")
            
            raise
    
    async def change_provider(self, provider: str, model: str = None) -> None:
        """Change the LLM provider and model asynchronously.
        
        Args:
            provider: The new provider name.
            model: The new model name (optional).
            
        Raises:
            ValueError: If the provider or model is not supported.
        """
        # Check if provider is supported
        if provider.lower() not in self.PROVIDER_CONFIGS:
            supported = ", ".join(self.PROVIDER_CONFIGS.keys())
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {supported}")
        
        # Set the provider
        provider = provider.lower()
        self.provider = provider
        
        # Fetch available models for this provider if we haven't already
        if provider not in self.available_models:
            try:
                await self._fetch_available_models(provider)
            except Exception as e:
                logging.warning(f"Could not fetch models for {provider}: {e}")
                # Use default models as fallback
                self.available_models[provider] = self.PROVIDER_CONFIGS[provider]["default_models"]
        
        # Check if model is supported for this provider and set it
        if model:
            if model not in self.available_models[provider]:
                raise ValueError(f"Model '{model}' is not available for {provider}. Use /llm to see available models.")
            self.model = model
        else:
            # If no model specified, use the first available model for this provider
            self.model = self.available_models[provider][0]
        
        # Set the API key
        if provider != "ollama":
            self.api_key = self.config.get_api_key(provider)
        else:
            self.api_key = None
            
        logging.info(f"Switched to {provider} provider with model {self.model}")

    async def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        """Call the Groq API asynchronously."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            return data['choices'][0]['message']['content']

    async def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI API asynchronously."""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            return data['choices'][0]['message']['content']

    async def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Call the Anthropic API asynchronously."""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Convert OpenAI message format to Anthropic format
        anthropic_messages = []
        system_content = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "messages": anthropic_messages,
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        
        if system_content:
            payload["system"] = system_content
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            
            # Anthropic API returns content as an array of content blocks
            if isinstance(data['content'], list) and len(data['content']) > 0:
                # Get the text from the first content block
                return data['content'][0]['text']
            else:
                return "No text content returned from Anthropic API"

    async def _call_google(self, messages: List[Dict[str, str]]) -> str:
        """Call the Google Gemini API using the official method."""
        # First ensure we have the required package
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("The 'google-generativeai' package is required. Install it with 'pip install google-generativeai'")
        
        # Configure the API with the provided key
        genai.configure(api_key=self.api_key)
        
        # Convert OpenAI message format to Gemini format
        system_content = None
        user_content = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content.append(msg["content"])
            # Note: assistant messages would be handled differently in a chat context
        
        # Combine user messages if there are multiple
        prompt = "\n".join(user_content) if user_content else ""
        
        # Initialize the model with system instruction if available
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 4096,
        }
        
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_content
        )
        
        # Generate content using the model
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        # Extract and return the response text
        if hasattr(response, 'text'):
            return response.text
        else:
            # Handle potential alternative response structures
            return str(response)

    async def _call_openroute(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenRoute API asynchronously."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://your-site.com",  # Required by OpenRouter
            "X-Title": "Your Application Name"  # Optional but recommended
        }
        
        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False
        }
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            return data['choices'][0]['message']['content']

    async def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call the Ollama API asynchronously."""
        # Ollama typically runs locally
        url = "http://localhost:11434/api/chat"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": 0.7,
                "num_predict": 4096
            }
        }
        
        session = await self._ensure_session()
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json()
            if "message" in data:
                return data["message"]["content"]
            return "No text content returned from Ollama API"

    async def cleanup(self) -> None:
        """Clean up client resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session if open
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._model_refresh_interval: int = 300  # Refresh models every 5 minutes
        self._model_refresh_task: Optional[asyncio.Task] = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")
        
        # Clean up LLM client
        await self.llm_client.cleanup()

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.
        
        Args:
            llm_response: The response from the LLM.
            
        Returns:
            The result of tool execution or the original response.
        """
        import json
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")
                
                # Try to find the tool across all servers
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                            
                            if isinstance(result, dict) and 'progress' in result:
                                progress = result['progress']
                                total = result['total']
                                logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                                
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            
                            # Try reconnecting the server
                            await server.initialize()
                            
                            # Retry the tool execution once more after reconnection
                            try:
                                result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                                return f"Tool execution result: {result}"
                            except Exception as retry_error:
                                return f"Error executing tool after reconnection: {str(retry_error)}"
                
                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    def display_llm_list(self) -> None:
        """Display a formatted list of available LLM providers and models."""
        print("\n===== Available LLM Providers and Models =====")
        
        for provider in self.llm_client.PROVIDER_CONFIGS.keys():
            # Check if API key is available (except for Ollama which doesn't need one)
            api_key_status = "✓" if provider == "ollama" or self.llm_client.config.get_api_key(provider) else "✗"
            
            if provider == self.llm_client.provider:
                print(f"\n🔹 {provider.upper()} {api_key_status}")
            else:
                print(f"\n• {provider.upper()} {api_key_status}")
            
            # Get models for this provider
            models = self.llm_client.available_models.get(provider, [])
            
            # If we couldn't fetch any models, show the defaults
            if not models:
                models = self.llm_client.PROVIDER_CONFIGS[provider]["default_models"]
                print("  (Using default models - API may be unavailable)")
                
            # Print models with the current one highlighted (limit to 10)
            for model in models[:10]:  # Limit to first 10 models to avoid flooding the console
                if provider == self.llm_client.provider and model == self.llm_client.model:
                    print(f"   ➜ {model}")
                else:
                    print(f"     {model}")
                    
            # If there are more models, indicate this
            remaining = len(models) - 10
            if remaining > 0:
                print(f"     ... and {remaining} more models")
            
            # If no API key is available, add a note
            if api_key_status == "✗" and provider != "ollama":
                print("     (Add API key in .env file to use this provider)")
        
        print("\nUse '/switch <provider> <model>' to change the LLM")
        print("Example: /switch openai gpt-4o")
        print("\nAPI Key Status: ✓ = configured, ✗ = missing")
    
    async def _worker(self, messages: List[Dict[str, str]]) -> None:
        """Background worker to process messages asynchronously.
        
        Args:
            messages: The current message history
        """
        while True:
            try:
                # Get the next user input from the queue
                user_input = await self.message_queue.get()
                
                if user_input == "__TERMINATE__":
                    break
                    
                # Add user message to history
                messages.append({"role": "user", "content": user_input})
                
                # Get LLM response
                llm_response = await self.llm_client.get_response(messages)
                logging.info("\nAssistant: %s", llm_response)

                # Process the response and execute any tools
                result = await self.process_llm_response(llm_response)
                
                # If we got a result from a tool, get a final response
                if result != llm_response:
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "system", "content": result})
                    
                    final_response = await self.llm_client.get_response(messages)
                    logging.info("\nFinal response: %s", final_response)
                    messages.append({"role": "assistant", "content": final_response})
                    
                    # Put the final response in the queue
                    await self.response_queue.put(final_response)
                else:
                    # Add the LLM response to history
                    messages.append({"role": "assistant", "content": llm_response})
                    
                    # Put the response in the queue
                    await self.response_queue.put(llm_response)
                
                # Mark this task as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in worker: {e}")
                await self.response_queue.put(f"An error occurred: {str(e)}")
                self.message_queue.task_done()
    
    async def start_model_refresh(self) -> None:
        """Start a periodic task to refresh models for all providers."""
        if self._model_refresh_task and not self._model_refresh_task.done():
            self._model_refresh_task.cancel()
            
        self._model_refresh_task = asyncio.create_task(self._model_refresh_loop())
        
    async def _model_refresh_loop(self) -> None:
        """Periodically refresh the model lists for all providers."""
        while True:
            try:
                # Wait for the refresh interval
                await asyncio.sleep(self._model_refresh_interval)
                
                logging.info("Refreshing model lists for all providers...")
                
                # Create fetch tasks for all providers
                fetch_tasks = []
                for provider_name in self.llm_client.PROVIDER_CONFIGS.keys():
                    # Only try to fetch if we have an API key (except for Ollama)
                    if provider_name == "ollama" or self.llm_client.config.get_api_key(provider_name) is not None:
                        task = asyncio.create_task(self.llm_client._fetch_provider_models(provider_name))
                        fetch_tasks.append(task)
                
                # Wait for all fetches to complete (don't raise exceptions)
                if fetch_tasks:
                    await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    
                logging.info("Model refresh complete")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in model refresh loop: {e}")
    
    async def start(self) -> None:
        """Main chat session handler with improved async flow."""
        try:
            # Initialize all servers in parallel
            init_tasks = []
            for server in self.servers:
                init_tasks.append(asyncio.create_task(server.initialize()))
            
            # Initialize LLM client in parallel
            llm_init_task = asyncio.create_task(self.llm_client.initialize())
            
            # Wait for all initializations to complete
            await asyncio.gather(*init_tasks, llm_init_task)
            
            # Start periodic model refreshing
            await self.start_model_refresh()
            
            # Collect all tools from all servers
            all_tools = []
            tool_tasks = []
            for server in self.servers:
                task = asyncio.create_task(server.list_tools())
                tool_tasks.append(task)
            
            # Wait for all tool lists and combine them
            tool_results = await asyncio.gather(*tool_tasks)
            for tools in tool_results:
                all_tools.extend(tools)
            
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]
            
            # Show current LLM provider and model at startup
            print(f"\nCurrent LLM: {self.llm_client.provider.upper()} - {self.llm_client.model}")
            print("Type '/llm' to see all available providers and models")
            print("Type '/help' for more commands\n")

            # Start the worker task
            self._worker_task = asyncio.create_task(self._worker(messages.copy()))

            while True:
                try:
                    # Get user input
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("You: ").strip()
                    )
                    
                    if user_input.lower() in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break
                    
                    # Check for provider switch commands
                    if user_input.startswith("/switch "):
                        parts = user_input.split()
                        if len(parts) >= 3:
                            provider = parts[1].lower()
                            model = parts[2]
                            
                            try:
                                await self.llm_client.change_provider(provider, model)
                                print(f"\nSwitched to {provider.upper()} with model {self.llm_client.model}")
                                continue
                            except ValueError as e:
                                print(f"\nError: {str(e)}")
                                continue
                        else:
                            print("\nUsage: /switch <provider> <model>")
                            print("Example: /switch openai gpt-4o")
                            print("Type '/llm' to see available providers and models")
                            continue
                    
                    elif user_input.lower() == "/llm":
                        self.display_llm_list()
                        continue
                    
                    elif user_input.lower() == "/refresh":
                        print("\nRefreshing model lists for all providers...")
                        refresh_tasks = []
                        for provider_name in self.llm_client.PROVIDER_CONFIGS.keys():
                            if provider_name == "ollama" or self.llm_client.config.get_api_key(provider_name) is not None:
                                task = asyncio.create_task(
                                    self.llm_client._fetch_provider_models(provider_name)
                                )
                                refresh_tasks.append(task)
                        
                        if refresh_tasks:
                            await asyncio.gather(*refresh_tasks, return_exceptions=True)
                        
                        print("Model refresh complete")
                        self.display_llm_list()
                        continue
                    
                    elif user_input.lower() == "/help":
                        print("\n===== Available Commands =====")
                        print("  /llm                      - Show available LLM providers and models")
                        print("  /switch <provider> <model> - Switch to a different LLM")
                        print("  /refresh                  - Refresh model lists for all providers")
                        print("  /help                     - Show this help message")
                        print("  quit or exit              - Exit the chat")
                        continue

                    # Put the user input in the queue for processing
                    await self.message_queue.put(user_input)
                    
                    # Wait for the response
                    response = await self.response_queue.get()
                    print(f"Assistant: {response}")
                    self.response_queue.task_done()

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break
        
        finally:
            # Signal the worker to terminate
            if self._worker_task and not self._worker_task.done():
                await self.message_queue.put("__TERMINATE__")
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel model refresh task
            if self._model_refresh_task and not self._model_refresh_task.done():
                self._model_refresh_task.cancel()
                try:
                    await self._model_refresh_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up all resources
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    
    # Create servers with the global config
    servers = [Server(name, srv_config, config) for name, srv_config in server_config['mcpServers'].items()]
    
    # Create and initialize LLM client
    llm_client = LLMClient(config)
    
    # Create and start chat session
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())