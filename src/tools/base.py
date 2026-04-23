"""
tools/base.py — 工具基类

目标：
  - 每个工具类自己定义 name / description / parameters / execute
  - Registry 直接持有 Tool 实例
  - Executor 统一基于 schema 做参数 cast / validate
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _resolve_json_schema_type(raw_type: Any) -> str | None:
    if isinstance(raw_type, list):
        return next((item for item in raw_type if item != "null"), None)
    return raw_type if isinstance(raw_type, str) else None


def _subpath(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _validate_json_schema_value(
    val: Any,
    schema: dict[str, Any],
    path: str = "",
) -> list[str]:
    raw_type = schema.get("type")
    nullable = (
        isinstance(raw_type, list) and "null" in raw_type
    ) or schema.get("nullable", False)
    resolved_type = _resolve_json_schema_type(raw_type)
    label = path or "parameter"

    if nullable and val is None:
        return []

    if resolved_type == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
        return [f"{label} should be integer"]
    if resolved_type == "number" and (
        not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
    ):
        return [f"{label} should be number"]
    if (
        resolved_type in _JSON_TYPE_MAP
        and resolved_type not in ("integer", "number")
        and not isinstance(val, _JSON_TYPE_MAP[resolved_type])
    ):
        return [f"{label} should be {resolved_type}"]

    errors: list[str] = []

    if "enum" in schema and val not in schema["enum"]:
        errors.append(f"{label} must be one of {schema['enum']}")

    if resolved_type in ("integer", "number"):
        if "minimum" in schema and val < schema["minimum"]:
            errors.append(f"{label} must be >= {schema['minimum']}")
        if "maximum" in schema and val > schema["maximum"]:
            errors.append(f"{label} must be <= {schema['maximum']}")

    if resolved_type == "string":
        if "minLength" in schema and len(val) < schema["minLength"]:
            errors.append(f"{label} must be at least {schema['minLength']} chars")
        if "maxLength" in schema and len(val) > schema["maxLength"]:
            errors.append(f"{label} must be at most {schema['maxLength']} chars")

    if resolved_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in val:
                errors.append(f"missing required {_subpath(path, key)}")

        additional_allowed = schema.get("additionalProperties", False) is True
        for key, item in val.items():
            if key in props:
                errors.extend(
                    _validate_json_schema_value(item, props[key], _subpath(path, key))
                )
            elif not additional_allowed:
                errors.append(f"unexpected {_subpath(path, key)}")

    if resolved_type == "array":
        if "minItems" in schema and len(val) < schema["minItems"]:
            errors.append(f"{label} must have at least {schema['minItems']} items")
        if "maxItems" in schema and len(val) > schema["maxItems"]:
            errors.append(f"{label} must be at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if item_schema:
            prefix = f"{path}[{{}}]" if path else "[{}]"
            for index, item in enumerate(val):
                errors.extend(
                    _validate_json_schema_value(item, item_schema, prefix.format(index))
                )

    return errors


class Tool(ABC):
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述。"""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema 风格参数定义。"""

    @property
    def read_only(self) -> bool:
        """是否不会修改用户数据或外部状态。"""
        return False

    @property
    def exclusive(self) -> bool:
        """是否需要独占执行。通常用于浏览器、单设备控制等场景。"""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """
        是否允许与其他工具调用并发执行。

        注意：这和 read_only 不是一个概念。只读工具也可能依赖共享连接、
        全局状态或非线程安全/非协程安全资源，因此默认保守为 False，
        需要具体工具显式声明。
        """
        return False

    @property
    def allow_spill(self) -> bool:
        return True

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具。"""

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        return {
            key: self._cast_value(value, props[key]) if key in props else value
            for key, value in obj.items()
        }

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        resolved_type = _resolve_json_schema_type(schema.get("type"))

        if resolved_type == "boolean" and isinstance(val, bool):
            return val
        if resolved_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if (
            resolved_type in _JSON_TYPE_MAP
            and resolved_type not in ("boolean", "integer", "array", "object")
            and isinstance(val, _JSON_TYPE_MAP[resolved_type])
        ):
            return val

        if isinstance(val, str) and resolved_type in ("integer", "number"):
            try:
                return int(val) if resolved_type == "integer" else float(val)
            except ValueError:
                return val

        if resolved_type == "string":
            return val if val is None else str(val)

        if resolved_type == "boolean" and isinstance(val, str):
            lowered = val.lower()
            if lowered in self._BOOL_TRUE:
                return True
            if lowered in self._BOOL_FALSE:
                return False
            return val

        if resolved_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in val] if item_schema else val

        if resolved_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return [f"schema must be object type, got {schema.get('type')!r}"]
        return _validate_json_schema_value(params, {**schema, "type": "object"})
