"""
Comprehensive unit tests for TypedDictUtils.filter_dict_by_typeddict()

Tests cover:
- Basic filtering functionality
- TypedDict inheritance
- Required vs optional fields
- Strict validation mode
- Edge cases
- Error handling
"""

from typing import TypedDict

import pytest
from typed_dict_utils import TypedDictUtils

# ============================================================================
# TEST FIXTURES & TYPEDDICTS
# ============================================================================


class SimpleParams(TypedDict, total=False):
    """Basic TypedDict with all optional fields."""

    name: str
    age: int
    email: str


class RequiredParams(TypedDict):
    """TypedDict with all required fields."""

    api_key: str
    token: str


class MixedParams(RequiredParams, total=False):
    """TypedDict with both required and optional fields."""

    timeout: int
    retries: int


class BaseConfig(TypedDict):
    """Base TypedDict with required fields."""

    id: int
    created_at: str


class ExtendedConfig(BaseConfig, total=False):
    """Extended TypedDict inheriting from BaseConfig."""

    name: str
    description: str


class DeepInheritance(ExtendedConfig):
    """TypedDict with deep inheritance chain."""

    extra_field: str


class NotATypeDict:
    """Regular class, not a TypedDict."""


# ============================================================================
# BASIC FILTERING TESTS
# ============================================================================


class TestBasicFiltering:
    """Test basic dictionary filtering functionality."""

    def test_simple_filtering_removes_unknown_keys(self):
        """Test that unknown keys are removed."""
        data = {"name": "Alice", "age": 30, "phone": "555-1234", "extra": "ignored"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result == {"name": "Alice", "age": 30}
        assert "phone" not in result
        assert "extra" not in result

    def test_filtering_keeps_all_valid_keys(self):
        """Test that all valid keys are preserved."""
        data = {"name": "Bob", "age": 25, "email": "bob@example.com"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result == data

    def test_filtering_partial_keys(self):
        """Test filtering with only some keys present."""
        data = {"name": "Charlie", "phone": "555-9999"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result == {"name": "Charlie"}
        assert "phone" not in result

    def test_filtering_empty_dict(self):
        """Test filtering an empty dictionary."""
        data = {}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result == {}

    def test_filtering_all_extra_keys(self):
        """Test filtering when all keys are extra."""
        data = {"a": 1, "b": 2, "c": 3}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result == {}

    def test_filtering_preserves_values(self):
        """Test that values are preserved correctly."""
        data = {"name": "Alice", "age": 30, "email": "alice@example.com", "extra": "removed"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["email"] == "alice@example.com"

    def test_filtering_preserves_value_types(self):
        """Test that value types are preserved."""
        data = {
            "name": "Alice",
            "age": 30,
            "extra": "removed",
            "extra_int": 999,
        }
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert isinstance(result["name"], str)
        assert isinstance(result["age"], int)
        assert result["age"] == 30


# ============================================================================
# INHERITANCE TESTS
# ============================================================================


class TestInheritance:
    """Test TypedDict inheritance handling."""

    def test_inheritance_includes_parent_keys(self):
        """Test that parent keys are included."""
        data = {
            "id": 1,
            "created_at": "2024-01-01",
            "name": "Test",
            "unknown": "removed",
        }
        result = TypedDictUtils.filter_dict_by_typeddict(data, ExtendedConfig)

        assert result == {
            "id": 1,
            "created_at": "2024-01-01",
            "name": "Test",
        }

    def test_deep_inheritance_chain(self):
        """Test with deep inheritance chain."""
        data = {
            "id": 1,
            "created_at": "2024-01-01",
            "name": "Test",
            "description": "A test",
            "extra_field": "value",
            "unknown": "removed",
        }
        result = TypedDictUtils.filter_dict_by_typeddict(data, DeepInheritance)

        assert result == {
            "id": 1,
            "created_at": "2024-01-01",
            "name": "Test",
            "description": "A test",
            "extra_field": "value",
        }

    def test_inheritance_parent_required_fields(self):
        """Test that parent required fields are recognized."""
        # ExtendedConfig inherits from BaseConfig which requires id and created_at
        data = {"id": 1, "created_at": "2024-01-01"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, ExtendedConfig)

        assert result == {"id": 1, "created_at": "2024-01-01"}

    def test_inheritance_mixed_required_optional(self):
        """Test inheritance with mixed required/optional fields."""
        data = {
            "api_key": "secret",
            "token": "xyz",
            "timeout": 30,
            "retries": 3,
            "debug": True,
        }
        result = TypedDictUtils.filter_dict_by_typeddict(data, MixedParams)

        assert result == {
            "api_key": "secret",
            "token": "xyz",
            "timeout": 30,
            "retries": 3,
        }


# ============================================================================
# REQUIRED VS OPTIONAL FIELD TESTS
# ============================================================================


class TestRequiredVsOptional:
    """Test handling of required vs optional fields."""

    def test_required_fields_detected(self):
        """Test that required fields are detected."""
        # RequiredParams has total=True (default), so all fields are required
        data = {"api_key": "abc123", "token": "xyz"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, RequiredParams)

        assert result == {"api_key": "abc123", "token": "xyz"}

    def test_optional_fields_detected(self):
        """Test that optional fields are detected."""
        # SimpleParams has total=False, so all fields are optional
        data = {"name": "Alice"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SimpleParams)

        assert result == {"name": "Alice"}

    def test_mixed_required_present(self):
        """Test mixed fields when all required are present."""
        data = {
            "api_key": "abc123",
            "token": "xyz",
            "timeout": 30,
        }
        result = TypedDictUtils.filter_dict_by_typeddict(data, MixedParams)

        assert result == {
            "api_key": "abc123",
            "token": "xyz",
            "timeout": 30,
        }

    def test_mixed_only_required_present(self):
        """Test mixed fields with only required fields."""
        data = {"api_key": "abc123", "token": "xyz"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, MixedParams)

        assert result == {"api_key": "abc123", "token": "xyz"}


# ============================================================================
# STRICT MODE TESTS
# ============================================================================


class TestStrictMode:
    """Test strict validation mode."""

    def test_strict_mode_raises_on_missing_required_keys(self):
        """Test that strict mode raises KeyError for missing required keys."""
        data = {"api_key": "abc123"}  # Missing token
        with pytest.raises(KeyError):
            TypedDictUtils.filter_dict_by_typeddict(data, RequiredParams, strict=True)

    def test_strict_mode_raises_with_descriptive_message(self):
        """Test that error message includes missing keys."""
        data = {"api_key": "abc123"}
        with pytest.raises(KeyError) as exc_info:
            TypedDictUtils.filter_dict_by_typeddict(data, RequiredParams, strict=True)

        error_message = str(exc_info.value)
        assert "token" in error_message

    def test_strict_mode_succeeds_with_all_required_keys(self):
        """Test that strict mode succeeds when all required keys present."""
        data = {"api_key": "abc123", "token": "xyz", "extra": "removed"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, RequiredParams, strict=True)

        assert result == {"api_key": "abc123", "token": "xyz"}

    def test_strict_mode_allows_missing_optional_keys(self):
        """Test that strict mode allows missing optional keys."""
        data = {"api_key": "abc123", "token": "xyz"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, MixedParams, strict=True)

        assert result == {"api_key": "abc123", "token": "xyz"}

    def test_strict_mode_multiple_missing_keys(self):
        """Test error when multiple required keys are missing."""
        data = {}
        with pytest.raises(KeyError) as exc_info:
            TypedDictUtils.filter_dict_by_typeddict(data, RequiredParams, strict=True)

        error_message = str(exc_info.value)
        assert "api_key" in error_message or "token" in error_message

    def test_non_strict_mode_allows_missing_required_keys(self):
        """Test that non-strict mode allows missing required keys."""
        data = {"api_key": "abc123"}  # Missing token
        result = TypedDictUtils.filter_dict_by_typeddict(data, RequiredParams, strict=False)

        assert result == {"api_key": "abc123"}


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and validation."""

    def test_non_dict_input_raises_typeerror(self):
        """Test that non-dict input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TypedDictUtils.filter_dict_by_typeddict([1, 2, 3], SimpleParams)

        assert "Expected dict" in str(exc_info.value)
        assert "list" in str(exc_info.value)

    def test_non_typeddict_class_raises_typeerror(self):
        """Test that non-TypedDict class raises TypeError."""
        data = {"a": 1, "b": 2}
        with pytest.raises(TypeError) as exc_info:
            TypedDictUtils.filter_dict_by_typeddict(data, NotATypeDict)

        assert "not a TypedDict" in str(exc_info.value)

    def test_none_input_raises_typeerror(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError):
            TypedDictUtils.filter_dict_by_typeddict(None, SimpleParams)

    def test_string_input_raises_typeerror(self):
        """Test that string input raises TypeError."""
        with pytest.raises(TypeError):
            TypedDictUtils.filter_dict_by_typeddict("not a dict", SimpleParams)

    def test_invalid_typeddict_class(self):
        """Test error handling for invalid TypedDict."""
        data = {"name": "Alice"}

        class FakeTypedDict(dict):
            """Looks like TypedDict but isn't."""

        with pytest.raises(TypeError):
            TypedDictUtils.filter_dict_by_typeddict(data, FakeTypedDict)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_typeddict(self):
        """Test filtering with empty TypedDict."""

        class EmptyParams(TypedDict, total=False):
            pass

        data = {"a": 1, "b": 2}
        result = TypedDictUtils.filter_dict_by_typeddict(data, EmptyParams)

        assert result == {}

    def test_single_key_typeddict(self):
        """Test TypedDict with single key."""

        class SingleKeyParams(TypedDict, total=False):
            name: str

        data = {"name": "Alice", "extra": "removed"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, SingleKeyParams)

        assert result == {"name": "Alice"}

    def test_many_keys_typeddict(self):
        """Test TypedDict with many keys."""

        class ManyKeysParams(TypedDict, total=False):
            a: int
            b: int
            c: int
            d: int
            e: int

        data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}
        result = TypedDictUtils.filter_dict_by_typeddict(data, ManyKeysParams)

        assert len(result) == 5
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    def test_special_value_types(self):
        """Test filtering with special value types."""

        class SpecialParams(TypedDict, total=False):
            none_val: str
            list_val: list
            dict_val: dict

        data = {
            "none_val": None,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
            "extra": "removed",
        }
        result = TypedDictUtils.filter_dict_by_typeddict(data, SpecialParams)

        assert result["none_val"] is None
        assert result["list_val"] == [1, 2, 3]
        assert result["dict_val"] == {"nested": "value"}

    def test_unicode_keys(self):
        """Test filtering with unicode keys."""

        class UnicodeParams(TypedDict, total=False):
            name_en: str
            name_es: str

        data = {"name_en": "Alice", "name_es": "Alicia", "extra": "removed"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, UnicodeParams)

        assert result == {"name_en": "Alice", "name_es": "Alicia"}

    def test_numeric_string_values(self):
        """Test that numeric strings are preserved."""

        class ParamsWithStrings(TypedDict, total=False):
            id_str: str
            age_str: str

        data = {"id_str": "12345", "age_str": "30", "extra": "removed"}
        result = TypedDictUtils.filter_dict_by_typeddict(data, ParamsWithStrings)

        assert result["id_str"] == "12345"
        assert isinstance(result["id_str"], str)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test real-world integration scenarios."""

    def test_kwargs_cleaning(self):
        """Test using filter to clean function kwargs."""

        class ProcessOptions(TypedDict, total=False):
            batch_size: int
            num_workers: int
            shuffle: bool

        user_kwargs = {
            "batch_size": 32,
            "num_workers": 4,
            "shuffle": True,
            "debug": True,  # Not in TypedDict
            "verbose": False,  # Not in TypedDict
        }

        clean_kwargs = TypedDictUtils.filter_dict_by_typeddict(user_kwargs, ProcessOptions)

        # Can safely pass to function expecting ProcessOptions
        assert clean_kwargs == {"batch_size": 32, "num_workers": 4, "shuffle": True}

    def test_configuration_loading(self):
        """Test loading and validating configuration."""

        class AppConfig(TypedDict):
            database_url: str
            api_key: str

        class ExtendedAppConfig(AppConfig, total=False):
            debug: bool
            log_level: str

        config_file = {
            "database_url": "postgresql://localhost",
            "api_key": "secret123",
            "debug": True,
            "log_level": "INFO",
            "unknown_setting": "ignored",
        }

        # Validate required fields are present
        validated = TypedDictUtils.filter_dict_by_typeddict(config_file, ExtendedAppConfig, strict=True)

        assert validated["database_url"] == "postgresql://localhost"
        assert validated["api_key"] == "secret123"
        assert validated["debug"] is True

    def test_api_response_cleaning(self):
        """Test cleaning API response data."""

        class APIResponse(TypedDict, total=False):
            user_id: int
            username: str
            email: str

        api_response = {
            "user_id": 123,
            "username": "alice",
            "email": "alice@example.com",
            "internal_field": "ignored",
            "debug_info": "ignored",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        cleaned = TypedDictUtils.filter_dict_by_typeddict(api_response, APIResponse)

        assert cleaned == {"user_id": 123, "username": "alice", "email": "alice@example.com"}

    def test_chained_filtering(self):
        """Test filtering with multiple TypedDicts."""

        class BasicParams(TypedDict, total=False):
            name: str

        class ExtendedParams(BasicParams, total=False):
            email: str

        data = {"name": "Alice", "email": "alice@example.com", "extra": "removed"}

        # Filter by basic first
        basic = TypedDictUtils.filter_dict_by_typeddict(data, BasicParams)
        assert basic == {"name": "Alice"}

        # Filter by extended
        extended = TypedDictUtils.filter_dict_by_typeddict(data, ExtendedParams)
        assert extended == {"name": "Alice", "email": "alice@example.com"}


# ============================================================================
# ALIAS TESTS
# ============================================================================


class TestAliases:
    """Test convenience function aliases."""

    def test_drop_unknown_keys_alias(self):
        """Test drop_unknown_keys alias."""
        from typed_dict_utils import drop_unknown_keys

        data = {"name": "Alice", "extra": "removed"}
        result = drop_unknown_keys(data, SimpleParams)

        assert result == {"name": "Alice"}

    def test_validate_dict_keys_alias(self):
        """Test validate_dict_keys alias."""
        from typed_dict_utils import validate_dict_keys

        data = {"name": "Alice", "extra": "removed"}
        result = validate_dict_keys(data, SimpleParams)

        assert result == {"name": "Alice"}

    def test_validate_required_keys_alias(self):
        """Test validate_required_keys alias."""
        from typed_dict_utils import validate_required_keys

        data = {"api_key": "abc", "token": "xyz"}
        result = validate_required_keys(data, RequiredParams)

        assert result == {"api_key": "abc", "token": "xyz"}


# ============================================================================
# PERFORMANCE TESTS (optional, for benchmarking)
# ============================================================================


class TestPerformance:
    """Performance and stress tests."""

    def test_large_dict_filtering(self):
        """Test filtering large dictionaries."""

        class LargeParams(TypedDict, total=False):
            key_0: int
            key_1: int
            key_2: int
            key_3: int
            key_4: int

        # Create large data with many extra keys
        data = {f"key_{i}": i for i in range(100)}

        result = TypedDictUtils.filter_dict_by_typeddict(data, LargeParams)

        assert len(result) == 5
        assert all(k.startswith("key_") for k in result.keys())

    def test_deeply_nested_values(self):
        """Test filtering with deeply nested values."""

        class NestedParams(TypedDict, total=False):
            data: dict

        nested_data = {"data": {"level1": {"level2": {"level3": {"value": "deep"}}}}, "extra": "removed"}

        result = TypedDictUtils.filter_dict_by_typeddict(nested_data, NestedParams)

        assert result["data"]["level1"]["level2"]["level3"]["value"] == "deep"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
