"""
Web Automation Model for LM Arena

Integration with Manus automation agent for web-based tasks.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import aiohttp

from .agent import GenerationRequest, GenerationResponse, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class WebAutomationTask:
    """Web automation task specification"""
    url: str
    action: str
    selector: Optional[str] = None
    value: Optional[str] = None
    wait_for: Optional[str] = None
    screenshot: bool = False
    extract_text: bool = True


class WebAutomationModel:
    """Web automation model that integrates with Manus agent"""

    def __init__(self, manus_api_url: str = "http://localhost:8000"):
        self.name = "web-automation"
        self.provider = ModelProvider.MANUS
        self.manus_api_url = manus_api_url
        self.session = None
        # Add capabilities attribute for monitoring
        from .agent import ModelCapabilities, ModelConfig
        self.capabilities = ModelCapabilities(
            max_context_length=4096,
            supports_streaming=True,
            supports_functions=True,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
            supported_languages=["en", "zh", "es", "fr", "de"]
        )
        # Add config attribute for monitoring
        self.config = ModelConfig(
            name=self.name,
            provider=self.provider.value,
            model_id=self.name,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_requests_per_minute=60
        )

    async def initialize(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def validate_connection(self) -> bool:
        """Validate connection to Manus automation agent"""
        try:
            await self.initialize()
            async with self.session.get(f"{self.manus_api_url}/health", timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Failed to connect to Manus agent: {e}")
            return False

    async def execute_automation_task(self, task: WebAutomationTask) -> Dict[str, Any]:
        """Execute web automation task via Manus agent"""
        try:
            await self.initialize()

            payload = {
                "url": task.url,
                "action": task.action,
                "selector": task.selector,
                "value": task.value,
                "wait_for": task.wait_for,
                "screenshot": task.screenshot,
                "extract_text": task.extract_text
            }

            async with self.session.post(
                f"{self.manus_api_url}/automation",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Manus API error: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Web automation task failed: {e}")
            return {"error": str(e), "success": False}

    def parse_automation_request(self, prompt: str) -> Optional[WebAutomationTask]:
        """Parse user prompt into automation task"""
        prompt_lower = prompt.lower()

        # Simple keyword-based parsing
        if "search" in prompt_lower or "google" in prompt_lower:
            query = self._extract_search_query(prompt)
            if query:
                return WebAutomationTask(
                    url="https://www.google.com",
                    action="search",
                    selector="input[name='q']",
                    value=query,
                    wait_for="div#search"
                )

        elif "weather" in prompt_lower and "london" in prompt_lower:
            return WebAutomationTask(
                url="https://www.weather.com",
                action="navigate",
                wait_for="section[data-testid='CurrentConditions']"
            )

        elif "scrape" in prompt_lower or "extract" in prompt_lower:
            url = self._extract_url(prompt)
            if url:
                return WebAutomationTask(
                    url=url,
                    action="extract_content",
                    extract_text=True,
                    screenshot=True
                )

        return None

    def _extract_search_query(self, prompt: str) -> Optional[str]:
        """Extract search query from prompt"""
        import re

        # Look for patterns like "search for X" or "google X"
        patterns = [
            r"search for ([^.!?]+)",
            r"google ([^.!?]+)",
            r"search (.+)",
            r"find ([^.!?]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_url(self, prompt: str) -> Optional[str]:
        """Extract URL from prompt"""
        import re

        # Simple URL pattern matching
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, prompt)
        return match.group(0) if match else None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response using web automation"""
        start_time = time.time()

        try:
            # Parse automation task from prompt
            task = self.parse_automation_request(request.prompt)

            if not task:
                return GenerationResponse(
                    content="I couldn't parse a web automation task from your request. Try asking me to search for something, check weather, or scrape a website.",
                    model_name=self.name,
                    usage={"total_tokens": 0},
                    generation_time=time.time() - start_time,
                    metadata={"automation_task": None}
                )

            # Execute automation task
            result = await self.execute_automation_task(task)
            generation_time = time.time() - start_time

            if result.get("success", False):
                content = f"✅ **Web Automation Completed Successfully**\n\n"
                content += f"**Action:** {task.action}\n"
                content += f"**URL:** {task.url}\n\n"

                if "extracted_text" in result:
                    content += f"**Extracted Content:**\n{result['extracted_text']}\n\n"

                if "screenshot" in result:
                    content += f"**Screenshot:** Captured successfully\n"

                if "data" in result:
                    content += f"**Data:** {json.dumps(result['data'], indent=2)}\n"

                return GenerationResponse(
                    content=content,
                    model_name=self.name,
                    usage={"total_tokens": len(content.split())},
                    generation_time=generation_time,
                    metadata={
                        "automation_task": task.__dict__,
                        "automation_result": result
                    }
                )
            else:
                error_msg = result.get("error", "Unknown automation error")
                return GenerationResponse(
                    content=f"❌ **Web Automation Failed**\n\n**Error:** {error_msg}",
                    model_name=self.name,
                    usage={"total_tokens": 0},
                    generation_time=generation_time,
                    metadata={
                        "automation_task": task.__dict__,
                        "error": error_msg
                    }
                )

        except Exception as e:
            logger.error(f"Web automation generation failed: {e}")
            return GenerationResponse(
                content=f"❌ **Web Automation Error**\n\n{str(e)}",
                model_name=self.name,
                usage={"total_tokens": 0},
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class MockWebAutomationModel(WebAutomationModel):
    """Mock web automation model for testing when Manus agent is not available"""

    def __init__(self):
        super().__init__(manus_api_url="mock://localhost")
        # Add capabilities attribute for monitoring
        from .agent import ModelCapabilities, ModelConfig
        self.capabilities = ModelCapabilities(
            max_context_length=4096,
            supports_streaming=True,
            supports_functions=True,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
            supported_languages=["en", "zh", "es", "fr", "de"]
        )
        # Add config attribute for monitoring
        self.config = ModelConfig(
            name=self.name,
            provider=self.provider.value,
            model_id=self.name,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_requests_per_minute=60
        )

    async def execute_automation_task(self, task: WebAutomationTask) -> Dict[str, Any]:
        """Mock execution of automation task"""
        await asyncio.sleep(1)  # Simulate work

        if task.action == "search":
            return {
                "success": True,
                "extracted_text": f"Mock search results for: {task.value}",
                "data": {"query": task.value, "results_count": 10}
            }

        elif "weather" in task.url:
            return {
                "success": True,
                "extracted_text": "Current weather in London: 15°C, partly cloudy",
                "data": {"temperature": "15°C", "condition": "partly cloudy"}
            }

        elif task.action == "extract_content":
            return {
                "success": True,
                "extracted_text": f"Mock extracted content from {task.url}",
                "screenshot": "mock_screenshot.png"
            }

        else:
            return {
                "success": True,
                "extracted_text": f"Mock automation completed for action: {task.action}",
                "data": {"action": task.action, "url": task.url}
            }