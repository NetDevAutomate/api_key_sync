import pytest
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


class TestSyncEngine:
    def test_sync_new_key(self):
        source = MockStore({"KEY_A": "value_a"})
        target = MockStore()
        engine = SyncEngine(source, target, ["KEY_A"])
        
        result = engine.sync()
        
        assert result.synced == ["KEY_A"]
        assert target.data["KEY_A"] == "value_a"
    
    def test_sync_updated_key(self):
        source = MockStore({"KEY_A": "new_value"})
        target = MockStore({"KEY_A": "old_value"})
        engine = SyncEngine(source, target, ["KEY_A"])
        
        result = engine.sync()
        
        assert result.synced == ["KEY_A"]
        assert target.data["KEY_A"] == "new_value"
    
    def test_sync_unchanged_key(self):
        source = MockStore({"KEY_A": "same"})
        target = MockStore({"KEY_A": "same"})
        engine = SyncEngine(source, target, ["KEY_A"])
        
        result = engine.sync()
        
        assert result.skipped == ["KEY_A"]
        assert result.synced == []
    
    def test_sync_deletion(self):
        source = MockStore()
        target = MockStore({"KEY_A": "orphan"})
        engine = SyncEngine(source, target, ["KEY_A"])
        
        result = engine.sync(sync_deletions=True)
        
        assert result.deleted == ["KEY_A"]
        assert "KEY_A" not in target.data
    
    def test_sync_no_deletion_by_default(self):
        source = MockStore()
        target = MockStore({"KEY_A": "orphan"})
        engine = SyncEngine(source, target, ["KEY_A"])
        
        result = engine.sync(sync_deletions=False)
        
        assert result.deleted == []
        assert "KEY_A" in target.data
    
    def test_dry_run_no_changes(self):
        source = MockStore({"KEY_A": "value"})
        target = MockStore()
        engine = SyncEngine(source, target, ["KEY_A"])
        
        result = engine.sync(dry_run=True)
        
        assert result.synced == ["KEY_A"]
        assert "KEY_A" not in target.data  # No actual change
        assert len(target.put_calls) == 0
