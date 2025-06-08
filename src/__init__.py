#!/usr/bin/python
# -*- coding:utf-8 -*-

try:
    from .modules.reddit import get_post_data
except Exception:  # pragma: no cover - optional dependency
    get_post_data = None

try:
    from .utils.text_processor import clean_text
except Exception:  # pragma: no cover - optional dependency
    clean_text = None

try:
    from .utils.sentiment_analysis import get_sentiment
except Exception:  # pragma: no cover - optional dependency
    get_sentiment = None

__all__ = ['get_post_data', 'clean_text', 'get_sentiment']
