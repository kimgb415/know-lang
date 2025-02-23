import pytest
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from knowlang.api import ApiModelRegistry

# Define models but don't register them yet
class SampleStatus(str, Enum):
    """Test status enum"""
    ACTIVE = "active"
    INACTIVE = "inactive"

class SampleSubModel(BaseModel):
    """Test sub-model"""
    name: str
    count: int

class SampleModel(BaseModel):
    """Test model"""
    id: int
    status: SampleStatus
    items: List[SampleSubModel]
    description: Optional[str] = None

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test"""
    ApiModelRegistry._schemas = {}
    yield
    ApiModelRegistry._schemas = {}

def test_register_enum():
    """Test registering an enum"""
    # Register the enum inside the test
    registered_enum = ApiModelRegistry.register(SampleStatus)
    
    # Check if enum was registered
    assert "SampleStatus" in ApiModelRegistry._schemas
    
    # Check if type information is preserved
    assert registered_enum.ACTIVE == "active"
    assert isinstance(registered_enum.ACTIVE, registered_enum)

def test_register_model():
    """Test registering a Pydantic model"""
    # Register models inside the test
    ApiModelRegistry.register(SampleStatus)
    ApiModelRegistry.register(SampleSubModel)
    registered_model = ApiModelRegistry.register(SampleModel)
    
    # Check if models were registered
    assert "SampleModel" in ApiModelRegistry._schemas
    assert "SampleSubModel" in ApiModelRegistry._schemas
    
    # Verify we can still create model instances
    sub_item = SampleSubModel(name="test", count=1)
    model = registered_model(
        id=1,
        status=SampleStatus.ACTIVE,
        items=[sub_item]
    )
    assert model.id == 1
    assert model.status == SampleStatus.ACTIVE

def test_get_schema_enum():
    """Test getting schema for enum"""
    # Register the enum
    ApiModelRegistry.register(SampleStatus)
    
    schema = ApiModelRegistry.get_schema("SampleStatus")
    
    assert schema["type"] == "string"
    assert schema["enum"] == ["active", "inactive"]
    assert schema["title"] == "SampleStatus"
    assert schema["description"] == "Test status enum"

def test_get_schema_model():
    """Test getting schema for Pydantic model"""
    # Register all required models
    ApiModelRegistry.register(SampleStatus)
    ApiModelRegistry.register(SampleSubModel)
    ApiModelRegistry.register(SampleModel)
    
    schema = ApiModelRegistry.get_schema("SampleModel")
    
    # Check basic schema structure
    assert schema["type"] == "object"
    assert "properties" in schema
    
    # Check properties
    props = schema["properties"]
    assert "id" in props
    assert "status" in props
    assert "items" in props
    assert "description" in props
    
    # Check ref format
    assert schema["properties"]["status"]["$ref"] == "#/components/schemas/SampleStatus"

def test_get_schema_not_found():
    """Test getting schema for unregistered type"""
    with pytest.raises(ValueError, match="Schema NotFound not registered"):
        ApiModelRegistry.get_schema("NotFound")

def test_get_all_schemas():
    """Test getting all schemas"""
    # Register all models
    ApiModelRegistry.register(SampleStatus)
    ApiModelRegistry.register(SampleSubModel)
    ApiModelRegistry.register(SampleModel)
    
    schemas = ApiModelRegistry.get_all_schemas()
    
    # Check if all models are included
    assert "SampleStatus" in schemas
    assert "SampleModel" in schemas
    assert "SampleSubModel" in schemas
    
    # Check schema content for one model
    assert schemas["SampleStatus"]["type"] == "string"
    assert schemas["SampleStatus"]["enum"] == ["active", "inactive"]