from query_cache import (
    cache_size,
    clear_cache,
    get_cached_result,
    make_cache_key,
    set_cached_result,
)


def test_cache_key_is_stable():
    key1 = make_cache_key("What is property insurance?", 3, "gemini-2.0-flash")
    key2 = make_cache_key("What is property insurance?", 3, "gemini-2.0-flash")

    assert key1 == key2


def test_cache_key_changes_when_settings_change():
    key1 = make_cache_key("What is property insurance?", 3, "gemini-2.0-flash")
    key2 = make_cache_key("What is property insurance?", 5, "gemini-2.0-flash")

    assert key1 != key2


def test_cache_store_and_retrieve():
    clear_cache()

    key = make_cache_key("Test question", 3, "gemini-2.0-flash")
    result = {"answer": "Test answer", "sources": []}

    set_cached_result(key, result)

    assert get_cached_result(key) == result
    assert cache_size() == 1

    clear_cache()
    assert cache_size() == 0
