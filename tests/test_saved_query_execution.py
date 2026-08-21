import pytest
from pydantic import ValidationError

from app import SavedQueryRequest, app


def test_saved_query_execution_route_is_registered() -> None:
    routes = {route.path for route in app.routes}

    assert "/query/execute-saved" in routes


def test_saved_query_request_bounds_result_size() -> None:
    request = SavedQueryRequest(sql="SELECT 1", row_limit=500)

    assert request.row_limit == 500
    with pytest.raises(ValidationError):
        SavedQueryRequest(sql="SELECT 1", row_limit=2001)
