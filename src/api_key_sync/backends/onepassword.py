import json
import subprocess
from ..models import APIKey


class OnePasswordStore:
    def __init__(self, vault: str = "API_KEYS"):
        self.vault = vault
    
    def _run(self, args: list[str], input_data: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["op"] + args,
            capture_output=True,
            text=True,
            input=input_data,
            check=True
        )
    
    def get(self, name: str) -> str | None:
        try:
            result = self._run(["item", "list", "--vault", self.vault, "--format", "json"])
            items = json.loads(result.stdout)
            item_id = next((i["id"] for i in items if i["title"] == name), None)
            if not item_id:
                return None
            
            result = self._run(["item", "get", item_id, "--reveal", "--format", "json", "--vault", self.vault])
            data = json.loads(result.stdout)
            for field in data.get("fields", []):
                if field.get("label") == "credential" and field.get("value"):
                    return field["value"]
            return None
        except (subprocess.CalledProcessError, json.JSONDecodeError, StopIteration):
            return None
    
    def put(self, key: APIKey) -> bool:
        try:
            check = subprocess.run(
                ["op", "item", "get", key.name, "--vault", self.vault],
                capture_output=True, text=True
            )
            if check.returncode == 0:
                self._run(["item", "edit", key.name, "--vault", self.vault, f"credential={key.value}"])
            else:
                self._run([
                    "item", "create",
                    "--category", "API Credential",
                    "--title", key.name,
                    "--vault", self.vault,
                    f"credential={key.value}"
                ])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def delete(self, name: str) -> bool:
        try:
            self._run(["item", "delete", name, "--vault", self.vault])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def list_keys(self, key_names: list[str]) -> dict[str, str]:
        try:
            result = self._run(["item", "list", "--vault", self.vault, "--format", "json"])
            items = json.loads(result.stdout)
            name_set = set(key_names)
            matching = [i for i in items if i["title"] in name_set]
            
            if not matching:
                return {}
            
            # Pipe filtered items to op item get -
            filtered_json = json.dumps(matching)
            result = self._run(
                ["item", "get", "-", "--reveal", "--format", "json", "--vault", self.vault],
                input_data=filtered_json
            )
            
            # op returns newline-delimited JSON objects, wrap in array
            raw = result.stdout.strip()
            if raw.startswith("["):
                data = json.loads(raw)
            else:
                # Convert }{ or }\n{ to },{
                import re
                wrapped = "[" + re.sub(r'\}\s*\{', '},{', raw) + "]"
                data = json.loads(wrapped)
            
            if isinstance(data, dict):
                data = [data]
            
            keys = {}
            for item in data:
                title = item.get("title")
                for field in item.get("fields", []):
                    if field.get("label") == "credential":
                        val = field.get("value")
                        if val:  # Skip null/empty
                            keys[title] = val
                            break
            return keys
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return {}
