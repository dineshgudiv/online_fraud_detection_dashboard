"""Pagination helpers for list endpoints."""

from __future__ import annotations

from typing import Iterable, List, TypeVar

from fastapi import Query

from core.schemas import Page, PaginationParams


T = TypeVar("T")


def pagination_params(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
) -> PaginationParams:
    return PaginationParams(page=page, page_size=page_size)


def paginate_list(items: Iterable[T], params: PaginationParams) -> Page[T]:
    items_list: List[T] = list(items)
    total = len(items_list)
    start = (params.page - 1) * params.page_size
    end = start + params.page_size
    sliced = items_list[start:end]
    return Page(page=params.page, page_size=params.page_size, total=total, items=sliced)


def paginate_query(query, params: PaginationParams):
    total = query.count()
    items = query.offset((params.page - 1) * params.page_size).limit(params.page_size).all()
    return total, items


