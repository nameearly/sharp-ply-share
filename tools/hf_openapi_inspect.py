import json
import os
import argparse
from typing import Any, Dict, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_ref(doc: Dict[str, Any], ref: str) -> Any:
    # Only supports internal refs like "#/components/schemas/Foo"
    if not ref.startswith("#/"):
        return {"$ref": ref}
    cur: Any = doc
    for part in ref[2:].split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return {"$ref": ref}
    return cur


def _schema_brief(schema: Optional[Dict[str, Any]]) -> Any:
    if schema is None:
        return None
    if "$ref" in schema:
        return {"$ref": schema["$ref"]}
    out: Dict[str, Any] = {}
    if "type" in schema:
        out["type"] = schema.get("type")
    if "format" in schema:
        out["format"] = schema.get("format")
    if "enum" in schema:
        out["enum"] = schema.get("enum")
    if "items" in schema and isinstance(schema.get("items"), dict):
        out["items"] = _schema_brief(schema.get("items"))
    return out or {"type": "schema"}


def _print_endpoint(doc: Dict[str, Any], path: str, raw: bool) -> None:
    paths = doc.get("paths", {})
    item = paths.get(path)
    print("\n===", path, "===")
    if not isinstance(item, dict):
        print("MISSING")
        return

    if raw:
        print(json.dumps(item, ensure_ascii=False, indent=2)[:20000])
        return

    for method in sorted(item.keys()):
        if method.startswith("x-"):
            continue
        op = item[method]
        print("\n--", method.upper(), "--")
        if isinstance(op, dict):
            if op.get("operationId"):
                print("operationId:", op.get("operationId"))
            if op.get("summary"):
                print("summary:", op.get("summary"))
            if op.get("description"):
                print("description:", (op.get("description") or "").strip()[:300])
            print("security:", op.get("security"))

            params = op.get("parameters", []) or []
            if params:
                print("parameters:")
                for prm in params:
                    if "$ref" in prm:
                        prm = _resolve_ref(doc, prm["$ref"])
                    if not isinstance(prm, dict):
                        continue
                    name = prm.get("name")
                    pin = prm.get("in")
                    req = prm.get("required")
                    sch = prm.get("schema")
                    print(" -", name, "in=", pin, "required=", req, "schema=", _schema_brief(sch))

            rb = op.get("requestBody")
            if rb:
                if "$ref" in rb:
                    rb = _resolve_ref(doc, rb["$ref"])
                print("requestBody:")
                if isinstance(rb, dict):
                    print(" required:", rb.get("required"))
                    content = rb.get("content", {}) or {}
                    for ctype, ct in content.items():
                        if not isinstance(ct, dict):
                            continue
                        sch = ct.get("schema")
                        print(" -", ctype, _schema_brief(sch))

            responses = op.get("responses", {}) or {}
            if responses:
                print("responses:")
                for code in sorted(responses.keys()):
                    r = responses[code]
                    if "$ref" in r:
                        r = _resolve_ref(doc, r["$ref"])
                    if not isinstance(r, dict):
                        continue
                    desc = (r.get("description") or "").strip()
                    print(" -", code, desc[:200])
                    content = r.get("content") or {}
                    # Only show JSON-like responses
                    for ctype, ct in content.items():
                        if ctype not in ("application/json", "application/problem+json"):
                            continue
                        sch = (ct or {}).get("schema") if isinstance(ct, dict) else None
                        print("   ", ctype, _schema_brief(sch))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="Dump raw path item JSON for selected endpoints")
    args = parser.parse_args()

    default_path = os.path.join(os.getcwd(), ".session", "hf-openapi.json")
    openapi_path = os.environ.get("HF_OPENAPI_PATH", default_path)

    doc = _load_json(openapi_path)

    # Focused endpoints for org membership & roles
    endpoints = [
        "/api/organizations/{name}/members",
        "/api/organizations/{name}/members/{username}/role",
    ]

    for ep in endpoints:
        _print_endpoint(doc, ep, raw=bool(args.raw))


if __name__ == "__main__":
    main()
