from enum import Enum
from typing import Dict, Type, Union, cast, TypeVar
from pydantic import BaseModel

# Create type vars for better type hints
T = TypeVar('T', bound=Union[BaseModel, Enum])

class ApiModelRegistry:
    """Registry for both Pydantic models and Enums that need OpenAPI schemas"""
    _schemas: Dict[str, Union[Type[BaseModel], Type[Enum]]] = {}
    
    @classmethod
    def register(cls, schema_type: T) -> T:
        """Register a model or enum for OpenAPI schema generation"""
        cls._schemas[schema_type.__name__] = schema_type

        # Use cast to preserve the original type
        return cast(T, schema_type)
    
    @classmethod
    def get_schema(cls, name: str) -> dict:
        """Get OpenAPI schema for a registered type"""
        schema_type = cls._schemas.get(name)
        if not schema_type:
            raise ValueError(f"Schema {name} not registered")
            
        if issubclass(schema_type, BaseModel):
            # make it openapi compatible
            return schema_type.model_json_schema(ref_template="#/components/schemas/{model}")
        elif issubclass(schema_type, Enum):
            # Handle Enum schemas differently
            return {
                "type": "string",
                "enum": [e.value for e in schema_type],
                "title": schema_type.__name__,
                "description": schema_type.__doc__ or ""
            }
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    @classmethod
    def get_all_schemas(cls) -> Dict[str, dict]:
        """Get all registered OpenAPI schemas"""
        return {name: cls.get_schema(name) for name in cls._schemas}
