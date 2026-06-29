class DataLayerError(Exception):
    """Base exception for all data layer failures."""


class ProviderFetchError(DataLayerError):
    """Raised when an upstream OpenBB provider call fails or times out."""


class DataValidationError(DataLayerError):
    """Raised when upstream data fails Pydantic validation.

    The analyst layer must never see this data — it is caught at the
    data_layer boundary and either retried against a fallback provider
    or surfaced as a hard failure, never silently coerced.
    """
