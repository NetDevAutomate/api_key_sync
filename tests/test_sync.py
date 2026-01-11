from api_key_sync.models import APIKey
from api_key_sync.sync import SyncEngine


class MockStore:
    def __init__(self, data: dict[str, str] | None = None):
        self.data = data or {}
        self.put_calls: list[APIKey] = []
        self.delete_calls: list[str] = []

    def get(self, name: str) -> str | None:
        return self.data.get(name)

    def put(self, key: APIKey) -> bool:
        self.put_calls.append(key)
        self.data[key.name] = key.value
        return True

    def delete(self, name: str) -> bool:
        self.delete_calls.append(name)
        self.data.pop(name, None)
        return True

    def list_keys(self, key_names: list[str]) -> dict[str, str]:
        return {k: v for k, v in self.data.items() if k in key_names}

    def list_all_keys(self) -> dict[str, str]:
        return dict(self.data)


class TestSyncEngine:
    def test_sync_new_key(self):
        source = MockStore({"TEST_API_KEY": "value_a"})
        target = MockStore()
        engine = SyncEngine(source, target, patterns=["_API"])

        result = engine.sync()

        assert result.synced == ["TEST_API_KEY"]
        assert target.data["TEST_API_KEY"] == "value_a"

    def test_sync_updated_key(self):
        source = MockStore({"TEST_API_KEY": "new_value"})
        target = MockStore({"TEST_API_KEY": "old_value"})
        engine = SyncEngine(source, target, patterns=["_API"])

        result = engine.sync()

        assert result.synced == ["TEST_API_KEY"]
        assert target.data["TEST_API_KEY"] == "new_value"

    def test_sync_unchanged_key(self):
        source = MockStore({"TEST_API_KEY": "same"})
        target = MockStore({"TEST_API_KEY": "same"})
        engine = SyncEngine(source, target, patterns=["_API"])

        result = engine.sync()

        assert result.skipped == ["TEST_API_KEY"]
        assert result.synced == []

    def test_sync_deletion(self):
        # Source has one key, target has a different key that should be deleted
        source = MockStore({"OTHER_API_KEY": "keep_me"})
        target = MockStore({"TEST_API_KEY": "orphan", "OTHER_API_KEY": "keep_me"})
        engine = SyncEngine(source, target, patterns=["_API"])

        result = engine.sync(sync_deletions=True)

        assert result.deleted == ["TEST_API_KEY"]
        assert "TEST_API_KEY" not in target.data
        assert "OTHER_API_KEY" in target.data  # Keep existing key from source

    def test_sync_no_deletion_by_default(self):
        source = MockStore()
        target = MockStore({"TEST_API_KEY": "orphan"})
        engine = SyncEngine(source, target, patterns=["_API"])

        result = engine.sync(sync_deletions=False)

        assert result.deleted == []
        assert "TEST_API_KEY" in target.data

    def test_dry_run_no_changes(self):
        source = MockStore({"TEST_API_KEY": "value"})
        target = MockStore()
        engine = SyncEngine(source, target, patterns=["_API"])

        result = engine.sync(dry_run=True)

        assert result.synced == ["TEST_API_KEY"]
        assert "TEST_API_KEY" not in target.data  # No actual change
        assert len(target.put_calls) == 0

    def test_sync_filters_by_pattern(self):
        source = MockStore(
            {
                "TEST_API_KEY": "value1",
                "OTHER_TOKEN": "value2",
                "RANDOM_VAR": "value3",
            }
        )
        target = MockStore()
        engine = SyncEngine(source, target, patterns=["_API", "_TOKEN"])

        result = engine.sync()

        assert set(result.synced) == {"OTHER_TOKEN", "TEST_API_KEY"}
        assert "RANDOM_VAR" not in target.data

    def test_sync_case_sensitive(self):
        source = MockStore(
            {
                "TEST_api_key": "value1",  # Lowercase _api
                "OTHER_API_KEY": "value2",  # Uppercase _API
            }
        )
        target = MockStore()
        engine = SyncEngine(source, target, patterns=["_API"], case_sensitive=True)

        result = engine.sync()

        # Only uppercase matches
        assert result.synced == ["OTHER_API_KEY"]
        assert "TEST_api_key" not in target.data

    def test_sync_case_insensitive(self):
        source = MockStore(
            {
                "TEST_api_key": "value1",  # Lowercase _api
                "OTHER_API_KEY": "value2",  # Uppercase _API
            }
        )
        target = MockStore()
        engine = SyncEngine(source, target, patterns=["_API"], case_sensitive=False)

        result = engine.sync()

        # Both match
        assert set(result.synced) == {"OTHER_API_KEY", "TEST_api_key"}
