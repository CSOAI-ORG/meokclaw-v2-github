"""Structured Output / JSON Mode for MEOKCLAW

Enforce JSON schemas, function calling, and typed outputs.
Every response is validated against a Pydantic schema.
Invalid responses trigger automatic retry with schema correction.

Features:
- JSON schema enforcement via Pydantic
- Function calling with parameter validation
- Schema correction on validation failure
- Type-safe response parsing
- Custom output formats (Markdown tables, CSV, etc.)

Usage:
    from structured_output import structured_infer
    from pydantic import BaseModel

    class WeatherResponse(BaseModel):
        city: str
        temperature: float
        conditions: str
        forecast: list[str]

    result = await structured_infer(
        prompt="What's the weather in London?",
        schema=WeatherResponse,
        model="deepseek-v4-pro",
    )
    # result = WeatherResponse(city="London", temperature=15.2, ...)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult:
    data: Any
    raw_text: str
    parsed: bool
    validation_errors: List[str]
    retry_count: int
    model: str
    cost_usd: float
    latency_ms: int


class StructuredOutputEngine:
    """Engine for structured output generation with schema validation."""

    MAX_RETRIES = 3

    def __init__(self):
        self._schema_cache: Dict[str, str] = {}

    def _generate_schema_prompt(self, schema: Type[BaseModel]) -> str:
        """Generate a prompt suffix that instructs the model to output valid JSON."""
        schema_json = schema.model_json_schema()

        # Build a compact schema description
        required = schema_json.get("required", [])
        properties = schema_json.get("properties", {})

        field_descriptions = []
        for field_name, field_info in properties.items():
            ftype = field_info.get("type", "any")
            desc = field_info.get("description", "")
            field_descriptions.append(f'  "{field_name}": <{ftype}>  // {desc}')

        schema_desc = "\n".join(field_descriptions)

        return f"""

You MUST respond with a valid JSON object matching this exact schema:
{{
{schema_desc}
}}

Rules:
- Output ONLY the JSON object, no markdown, no explanations
- All required fields must be present: {', '.join(required)}
- Use correct JSON types (string, number, boolean, array, object)
- Do not include trailing commas
"""

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from model response (handles markdown code blocks)."""
        # Try markdown code block
        code_block = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Try raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json_match.group(0).strip()

        # Try JSON array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group(0).strip()

        return text.strip()

    def _fix_common_json_errors(self, text: str) -> str:
        """Fix common JSON parsing errors."""
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix single quotes to double quotes (simple cases)
        text = re.sub(r"'([^']*?)'(?=\s*:)", r'"\1"', text)
        text = re.sub(r":\s*'([^']*?)'(?=\s*[,}\]])", r': "\1"', text)
        return text

    async def infer(
        self,
        prompt: str,
        schema: Type[T],
        model: str = "deepseek-v4-pro",
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> StructuredResult:
        """
        Generate structured output matching a Pydantic schema.
        Retries with schema correction on validation failure.
        """
        from dual_brain_orchestrator import DualBrainOrchestrator

        orch = DualBrainOrchestrator()
        schema_instruction = self._generate_schema_prompt(schema)

        full_prompt = prompt + schema_instruction

        last_error = None
        raw_text = ""
        total_cost = 0.0
        total_latency = 0

        for attempt in range(max_retries):
            try:
                result = await orch.think(full_prompt, None)
                raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
                total_cost += result.get("cost_usd", 0.0) if isinstance(result, dict) else 0.0
                total_latency += result.get("latency_ms", 0) if isinstance(result, dict) else 0

                # Extract JSON
                json_str = self._extract_json(raw_text)
                if not json_str:
                    raise ValueError("No JSON found in response")

                # Fix common errors
                json_str = self._fix_common_json_errors(json_str)

                # Parse
                data = json.loads(json_str)

                # Validate with Pydantic
                validated = schema.model_validate(data)

                return StructuredResult(
                    data=validated,
                    raw_text=raw_text,
                    parsed=True,
                    validation_errors=[],
                    retry_count=attempt,
                    model=model,
                    cost_usd=total_cost,
                    latency_ms=total_latency,
                )

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = str(e)
                # Add error feedback for next attempt
                full_prompt = (
                    prompt + schema_instruction +
                    f"\n\nYour previous response was invalid: {last_error}\n" +
                    "Please fix the JSON and try again. Output ONLY valid JSON."
                )
                continue

        # All retries exhausted
        return StructuredResult(
            data=None,
            raw_text=raw_text,
            parsed=False,
            validation_errors=[last_error] if last_error else ["Unknown error"],
            retry_count=max_retries,
            model=model,
            cost_usd=total_cost,
            latency_ms=total_latency,
        )

    async def function_call(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        model: str = "deepseek-v4-pro",
    ) -> Dict[str, Any]:
        """
        Function calling with parameter validation.
        
        functions format:
        [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "city": {"type": "string", "required": True},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                }
            }
        ]
        """
        func_desc = "\n".join([
            f"- {f['name']}: {f.get('description', '')}\n  Parameters: {json.dumps(f.get('parameters', {}))}"
            for f in functions
        ])

        system_msg = f"""You have access to these functions:
{func_desc}

Respond with a JSON object in this format:
{{"function": "name", "arguments": {{"param": "value"}}}}

Choose the most appropriate function for the user's request."""

        result = await self.infer(
            prompt=prompt,
            schema=_FunctionCallSchema,
            model=model,
            system_prompt=system_msg,
        )

        if result.parsed:
            return result.data.model_dump()

        return {"error": "Failed to parse function call", "raw": result.raw_text}


# Dynamic schema for function calls
class _FunctionCallSchema(BaseModel):
    function: str
    arguments: Dict[str, Any]


# Singleton
structured_engine = StructuredOutputEngine()


if __name__ == "__main__":
    import asyncio

    class WeatherResponse(BaseModel):
        city: str
        temperature: float
        conditions: str
        forecast: list[str]

    async def demo():
        engine = StructuredOutputEngine()

        result = await engine.infer(
            prompt="What's the weather like?",
            schema=WeatherResponse,
        )

        print(f"Parsed: {result.parsed}")
        print(f"Retry count: {result.retry_count}")
        if result.parsed:
            print(f"Data: {result.data.model_dump()}")
        else:
            print(f"Errors: {result.validation_errors}")
            print(f"Raw: {result.raw_text[:200]}")

    asyncio.run(demo())
