import os
import shutil
import struct
import subprocess


def maybe_export_from_ply(
    ply_path: str,
    *,
    enabled: bool,
    tool: str,
    gsbox_bin: str,
    gsbox_spz_quality: int,
    gsbox_spz_version: int,
    gsconverter_bin: str,
    gsconverter_compression_level: int,
    debug_fn,
) -> str | None:
    try:
        if not enabled:
            return None
        src = os.path.abspath(str(ply_path))
        if not os.path.isfile(src):
            return None

        base, _ = os.path.splitext(src)
        out = base + ".spz"
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out

        tool = (tool or "").strip().lower() or "3dgsconverter"

        timeout_s = None
        try:
            raw = str(os.getenv("SPZ_TIMEOUT_SECS", "") or "").strip()
            if raw:
                timeout_s = float(raw)
        except Exception:
            timeout_s = None
        if timeout_s is None:
            try:
                if tool in ("3dgsconverter", "gsconverter", "gsconv"):
                    timeout_s = float(str(os.getenv("SPZ_TIMEOUT_SECS_CUDA", "900") or "900").strip())
                else:
                    timeout_s = float(str(os.getenv("SPZ_TIMEOUT_SECS_CPU", "1800") or "1800").strip())
            except Exception:
                timeout_s = 900.0
        try:
            timeout_s = float(timeout_s)
        except Exception:
            timeout_s = 900.0
        if timeout_s <= 0:
            timeout_s = None

        def _print(msg: str):
            try:
                if debug_fn:
                    debug_fn(msg)
            except Exception:
                return

        def _gsbox_resolve_bin() -> str:
            try:
                cand = str(gsbox_bin or "").strip() or "gsbox"
                if os.path.isfile(cand):
                    return cand
                which = shutil.which(cand)
                if which:
                    return which
                local = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools",
                    "gsbox",
                    "gsbox.exe",
                )
                if os.path.isfile(local):
                    return local
                return cand
            except Exception:
                return str(gsbox_bin or "gsbox")

        def _gsconverter_resolve_bin() -> str:
            try:
                cand = str(gsconverter_bin or "").strip() or "3dgsconverter"
                if os.path.isfile(cand):
                    return cand
                which = shutil.which(cand)
                if which:
                    return which
                return cand
            except Exception:
                return str(gsconverter_bin or "3dgsconverter")

        def _ply_has_non_vertex_elements(path: str) -> bool:
            try:
                path = os.path.abspath(str(path))
                if not os.path.isfile(path):
                    return False
                with open(path, "rb") as f:
                    seen_vertex = False
                    while True:
                        raw = f.readline()
                        if not raw:
                            return False
                        try:
                            line = raw.decode("ascii", errors="ignore").strip("\r\n")
                        except Exception:
                            line = ""
                        low = line.strip().lower()
                        if low == "end_header":
                            break
                        if not low.startswith("element "):
                            continue
                        parts = low.split()
                        if len(parts) < 3:
                            continue
                        name = parts[1]
                        if name == "vertex":
                            seen_vertex = True
                            continue
                        if seen_vertex:
                            return True
                return False
            except Exception:
                return False

        def _ply_make_vertex_only_binary_little_endian(in_path: str) -> str | None:
            try:
                in_path = os.path.abspath(str(in_path))
                if not os.path.isfile(in_path):
                    return None
                obase, _ = os.path.splitext(in_path)
                out_path = obase + ".vertexonly.binary.ply"
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                    return out_path

                type_map = {
                    "char": ("b", 1),
                    "int8": ("b", 1),
                    "uchar": ("B", 1),
                    "uint8": ("B", 1),
                    "short": ("h", 2),
                    "int16": ("h", 2),
                    "ushort": ("H", 2),
                    "uint16": ("H", 2),
                    "int": ("i", 4),
                    "int32": ("i", 4),
                    "uint": ("I", 4),
                    "uint32": ("I", 4),
                    "float": ("f", 4),
                    "float32": ("f", 4),
                    "double": ("d", 8),
                    "float64": ("d", 8),
                }

                with open(in_path, "rb") as rf:
                    fmt = None
                    vertex_count = None
                    in_vertex = False
                    vertex_props: list[tuple[str, str]] = []

                    while True:
                        raw = rf.readline()
                        if not raw:
                            return None
                        try:
                            line = raw.decode("ascii", errors="ignore").strip("\r\n")
                        except Exception:
                            line = ""

                        low = line.strip().lower()
                        if low.startswith("format "):
                            parts = low.split()
                            if len(parts) >= 3:
                                fmt = parts[1]
                        elif low.startswith("element "):
                            parts = low.split()
                            if len(parts) >= 3:
                                name = parts[1]
                                count_s = parts[2]
                                if name == "vertex":
                                    try:
                                        vertex_count = int(count_s)
                                    except Exception:
                                        vertex_count = None
                                    in_vertex = True
                                else:
                                    in_vertex = False
                        elif low.startswith("property ") and in_vertex:
                            parts = low.split()
                            if len(parts) >= 3:
                                if parts[1] == "list":
                                    return None
                                ptype = parts[1]
                                pname = parts[2]
                                if ptype not in type_map:
                                    return None
                                vertex_props.append((ptype, pname))
                        elif low == "end_header":
                            break

                    if not fmt or vertex_count is None or vertex_count <= 0 or not vertex_props:
                        return None

                    out_header = []
                    out_header.append("ply")
                    out_header.append("format binary_little_endian 1.0")
                    out_header.append("comment vertex-only rewritten for gsbox")
                    out_header.append(f"element vertex {int(vertex_count)}")
                    for ptype, pname in vertex_props:
                        out_header.append(f"property {ptype} {pname}")
                    out_header.append("end_header")
                    out_header_blob = ("\n".join(out_header) + "\n").encode("ascii")

                    if fmt == "binary_little_endian":
                        bytes_per_vertex = 0
                        for ptype, _ in vertex_props:
                            _, sz = type_map[ptype]
                            bytes_per_vertex += int(sz)
                        need = int(vertex_count) * int(bytes_per_vertex)
                        data = rf.read(need)
                        if len(data) != need:
                            return None
                        with open(out_path, "wb") as wf:
                            wf.write(out_header_blob)
                            wf.write(data)
                        return out_path

                    if fmt == "ascii":
                        pack_fmt = "<" + "".join(type_map[ptype][0] for ptype, _ in vertex_props)
                        with open(out_path, "wb") as wf:
                            wf.write(out_header_blob)
                            for _ in range(int(vertex_count)):
                                line_b = rf.readline()
                                if not line_b:
                                    return None
                                s = line_b.decode("ascii", errors="ignore").strip()
                                parts = [p for p in s.split() if p]
                                if len(parts) < len(vertex_props):
                                    return None
                                vals = []
                                for (ptype, _), tok in zip(vertex_props, parts):
                                    code = type_map[ptype][0]
                                    if code in ("f", "d"):
                                        vals.append(float(tok))
                                    else:
                                        vals.append(int(float(tok)))
                                wf.write(struct.pack(pack_fmt, *vals))
                        return out_path

                    return None
            except Exception:
                return None

        if tool == "gsbox":
            resolved = _gsbox_resolve_bin()
            try:
                if not (resolved and (os.path.isfile(resolved) or shutil.which(resolved))):
                    _print(
                        "SPZ: 未找到 gsbox，可通过以下任一方式解决：\n"
                        "  1) 设置环境变量 GSBOX_BIN 指向 gsbox.exe\n"
                        "  2) 将 gsbox.exe 放到 PATH\n"
                        "  3) 放到本项目 tools/gsbox/gsbox.exe（当前已支持自动查找）\n"
                        "  4) 或改用 SPZ_TOOL=3dgsconverter\n"
                        "参考：https://github.com/gotoeasy/gsbox/releases"
                    )
                    return None
            except Exception:
                return None

            _print(f"SPZ: gsbox={resolved}")

            def _gsbox_cmd(in_ply: str) -> list[str]:
                cmd = [resolved, "p2z", "-i", in_ply, "-o", out]
                if int(gsbox_spz_quality) > 0:
                    cmd += ["-q", str(int(gsbox_spz_quality))]
                if int(gsbox_spz_version) > 0:
                    cmd += ["-ov", str(int(gsbox_spz_version))]
                return cmd

            src_for_gsbox = src
            try:
                if _ply_has_non_vertex_elements(src):
                    tmp = _ply_make_vertex_only_binary_little_endian(src)
                    if tmp:
                        _print(f"SPZ: PLY 含非 vertex 元素（如 extrinsic/intrinsic），将使用 vertex-only 重写后交给 gsbox | ply={os.path.basename(src)}")
                        src_for_gsbox = tmp
            except Exception:
                src_for_gsbox = src

            try:
                p = subprocess.run(
                    _gsbox_cmd(src_for_gsbox),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as e:
                _print(f"SPZ: gsbox 超时（跳过） | timeout_s={int(timeout_s or 0)} | err={str(e)}")
                return None
            if p.returncode != 0 or (not os.path.isfile(out)) or os.path.getsize(out) <= 0:
                tmp = _ply_make_vertex_only_binary_little_endian(src)
                if tmp:
                    _print(f"SPZ: gsbox 初次转换失败，将重写 vertex-only PLY 后重试 | ply={os.path.basename(src)}")
                    try:
                        p2 = subprocess.run(
                            _gsbox_cmd(tmp),
                            capture_output=True,
                            text=True,
                            timeout=timeout_s,
                        )
                    except subprocess.TimeoutExpired as e:
                        _print(f"SPZ: gsbox 超时（跳过） | timeout_s={int(timeout_s or 0)} | err={str(e)}")
                        return None
                    if p2.returncode != 0:
                        msg = ((p2.stderr or "") + "\n" + (p2.stdout or "")).strip()
                        _print(
                            "SPZ: gsbox 二次转换仍失败（vertex-only 重写已执行）。建议：\n"
                            "  - 检查 PLY 是否损坏/截断\n"
                            "  - 或改用 SPZ_TOOL=3dgsconverter\n"
                            f"err={msg[:400]}"
                        )
                        raise RuntimeError(msg[:800])
                else:
                    msg = ((p.stderr or "") + "\n" + (p.stdout or "")).strip()
                    _print(
                        "SPZ: gsbox 转换失败且无法重写 vertex-only PLY。建议：\n"
                        "  - 检查 PLY header/format\n"
                        "  - 或改用 SPZ_TOOL=3dgsconverter\n"
                        f"err={msg[:400]}"
                    )
                    raise RuntimeError(msg[:800])

        elif tool in ("3dgsconverter", "gsconverter", "gsconv"):
            resolved = _gsconverter_resolve_bin()
            try:
                if not (resolved and (os.path.isfile(resolved) or shutil.which(resolved))):
                    _print(
                        "SPZ: 未找到 3dgsconverter，可通过以下任一方式解决：\n"
                        "  1) pip install git+https://github.com/francescofugazzi/3dgsconverter.git\n"
                        "  2) 设置环境变量 GSCONVERTER_BIN 指向 3dgsconverter/gsconverter\n"
                        "  3) 或改用 SPZ_TOOL=gsbox\n"
                    )
                    _print("SPZ: 将自动回退到 gsbox")
                    tool = "gsbox"
                else:
                    _print(f"SPZ: 3dgsconverter={resolved}")
                    cmd = [
                        str(resolved),
                        "-i",
                        src,
                        "-o",
                        out,
                        "-f",
                        "spz",
                        "--compression_level",
                        str(int(gsconverter_compression_level)),
                        "--rgb",
                        "--force",
                    ]
                    subprocess.run(cmd, check=True, timeout=timeout_s)
            except subprocess.TimeoutExpired as e:
                _print(f"SPZ: 3dgsconverter 超时（跳过） | timeout_s={int(timeout_s or 0)} | err={str(e)}")
                return None
            except Exception as e:
                _print(f"SPZ: 3dgsconverter 转换失败，将自动回退到 gsbox | err={str(e)}")
                tool = "gsbox"

            if tool == "gsbox":
                resolved = _gsbox_resolve_bin()
                try:
                    if not (resolved and (os.path.isfile(resolved) or shutil.which(resolved))):
                        return None
                except Exception:
                    return None

                _print(f"SPZ: gsbox={resolved}")

                def _gsbox_cmd(in_ply: str) -> list[str]:
                    cmd = [resolved, "p2z", "-i", in_ply, "-o", out]
                    if int(gsbox_spz_quality) > 0:
                        cmd += ["-q", str(int(gsbox_spz_quality))]
                    if int(gsbox_spz_version) > 0:
                        cmd += ["-ov", str(int(gsbox_spz_version))]
                    return cmd

                src_for_gsbox = src
                try:
                    if _ply_has_non_vertex_elements(src):
                        tmp = _ply_make_vertex_only_binary_little_endian(src)
                        if tmp:
                            _print(f"SPZ: PLY 含非 vertex 元素（如 extrinsic/intrinsic），将使用 vertex-only 重写后交给 gsbox | ply={os.path.basename(src)}")
                            src_for_gsbox = tmp
                except Exception:
                    src_for_gsbox = src

                try:
                    p = subprocess.run(
                        _gsbox_cmd(src_for_gsbox),
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                except subprocess.TimeoutExpired as e:
                    _print(f"SPZ: gsbox 超时（跳过） | timeout_s={int(timeout_s or 0)} | err={str(e)}")
                    return None
                if p.returncode != 0 or (not os.path.isfile(out)) or os.path.getsize(out) <= 0:
                    tmp = _ply_make_vertex_only_binary_little_endian(src)
                    if tmp:
                        _print(f"SPZ: gsbox 初次转换失败，将重写 vertex-only PLY 后重试 | ply={os.path.basename(src)}")
                        p2 = subprocess.run(
                            _gsbox_cmd(tmp),
                            capture_output=True,
                            text=True,
                            timeout=timeout_s,
                        )
                        if p2.returncode != 0:
                            msg = ((p2.stderr or "") + "\n" + (p2.stdout or "")).strip()
                            _print(
                                "SPZ: gsbox 二次转换仍失败（vertex-only 重写已执行）。建议：\n"
                                "  - 检查 PLY 是否损坏/截断\n"
                                "  - 或改用 SPZ_TOOL=3dgsconverter\n"
                                f"err={msg[:400]}"
                            )
                            raise RuntimeError(msg[:800])
                    else:
                        msg = ((p.stderr or "") + "\n" + (p.stdout or "")).strip()
                        _print(
                            "SPZ: gsbox 转换失败且无法重写 vertex-only PLY。建议：\n"
                            "  - 检查 PLY header/format\n"
                            "  - 或改用 SPZ_TOOL=3dgsconverter\n"
                            f"err={msg[:400]}"
                        )
                        raise RuntimeError(msg[:800])
        else:
            _print(f"SPZ: 未知 SPZ_TOOL={tool}，跳过")
            return None

        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        return None
    except Exception as e:
        try:
            if debug_fn:
                debug_fn(f"SPZ: 生成失败（可忽略） | err={str(e)}")
        except Exception:
            pass
        return None
