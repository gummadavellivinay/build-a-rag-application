import json
from datetime import datetime, timezone
from urllib.parse import urlparse
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field, AwareDatetime
from .models import Evidence


class ListingStatus(BaseModel):
    listing_id: str
    status: str = Field(max_length=100)
    asking_price: float = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    updated_at: AwareDatetime
    tenant: str


class Inventory:
    def __init__(self, settings, tenant):
        self.settings, self.tenant = settings, tenant

    def lookup(self, listing_id: str) -> Evidence:
        from .config import safe_id

        safe_id(listing_id)
        base = self.settings.inventory_base_url
        if base:
            parsed = urlparse(base)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.query:
                raise ValueError("Inventory base must be an operator-configured HTTPS origin/path")
            # The user/LLM never supplies a URL. Redirects cannot escape the configured endpoint.
            with httpx.Client(timeout=5, follow_redirects=False) as client:
                response = client.get(
                    f"{base.rstrip('/')}/listings/{listing_id}",
                    headers={
                        "Authorization": f"Bearer {self.settings.inventory_api_token.get_secret_value()}",
                        "X-Tenant-ID": self.tenant,
                    },
                )
                response.raise_for_status()
                data = response.json()
            source = f"inventory:{listing_id}"
        else:
            rows = json.loads((self.settings.data_dir / "inventory.json").read_text())
            data = next(
                (r for r in rows if r["listing_id"] == listing_id and r["tenant"] == self.tenant), None
            )
            if data is None:
                raise LookupError("Listing not found in inventory")
            source = f"fixture-inventory:{listing_id}"
        status = ListingStatus.model_validate(data)
        if status.tenant != self.tenant or status.listing_id != listing_id:
            raise ValueError("Tool identity mismatch")
        age = (datetime.now(timezone.utc) - status.updated_at).total_seconds()
        if base and (age > 900 or age < -60):
            raise ValueError("Inventory response timestamp is stale or invalid")
        label = "LIVE service response" if base else "SYNTHETIC FIXTURE, not live market data"
        return Evidence(
            id=f"tool:{listing_id}", source=source, kind="tool", text=label + "\n" + status.model_dump_json()
        )

    def langchain_tool(self):
        @tool
        def listing_status(listing_id: str) -> dict:
            """Read asking price and availability for one authorized, resolved listing ID."""
            return self.lookup(listing_id).model_dump()

        return listing_status
