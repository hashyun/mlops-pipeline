#!/usr/bin/python
# -*- coding:utf-8 -*-

try:
    from . import text_processor
except Exception:  # pragma: no cover - optional dependency
    text_processor = None

try:
    from .sentiment_analysis import get_sentiment
except Exception:  # pragma: no cover - optional dependency
    def get_sentiment(*args, **kwargs):
        raise ImportError('sentiment analysis dependencies are missing')

__all__ = ['text_processor', 'get_sentiment']
