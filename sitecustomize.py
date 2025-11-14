import warnings

try:
    # pydantic v2 - import the warnings module and resolve attributes at runtime
    import pydantic.warnings as pydantic_warnings  # type: ignore

    PydanticUserWarning = getattr(pydantic_warnings, "PydanticUserWarning", UserWarning)
    UnsupportedFieldAttributeWarning = getattr(pydantic_warnings, "UnsupportedFieldAttributeWarning", Warning)

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
    # belt-and-suspenders in case subclasses vary
    warnings.filterwarnings(
        "ignore",
        category=PydanticUserWarning,
        message=r".*UnsupportedFieldAttributeWarning.*",
    )
except Exception:
    pass

# also quiet the MPS pin_memory message
warnings.filterwarnings(
    "ignore", message=r".*pin_memory.*", category=UserWarning, module=r"torch\.utils\.data\.dataloader"
)
