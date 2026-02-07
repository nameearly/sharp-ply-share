import asyncio
import sys
import os
import random
import re
from datetime import datetime
import playwright
from playwright.async_api import async_playwright

async def register_unsplash_app(app_name: str, headless: bool = False) -> str | None:
    """
    Registers a new Unsplash app and returns the Access Key.
    """
    async with async_playwright() as p:
        try:
            print(f"[DIAG] playwright={getattr(playwright, '__version__', 'unknown')}")
        except Exception:
            pass

        pw_channel = str(os.getenv("PLAYWRIGHT_CHANNEL", "") or "").strip() or None
        if pw_channel:
            try:
                print(f"[DIAG] channel={pw_channel}")
            except Exception:
                pass

        user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".unsplash_session")
        try:
            os.makedirs(os.path.dirname(user_data_dir), exist_ok=True)
        except Exception:
            pass
            
        browser_for_version = None
        try:
            # 增加更多真实的浏览器启动参数，减少被判定为机器人的概率
            context = await p.chromium.launch_persistent_context(
                user_data_dir, 
                headless=headless,
                channel=pw_channel,
                viewport={'width': 1366, 'height': 768},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="en-US",
                timezone_id="Asia/Shanghai",
                ignore_https_errors=True
            )
            try:
                browser_for_version = context.browser
            except Exception:
                browser_for_version = None
        except Exception as e:
            print(f"无法启动 Chrome/Chromium: {e}. 尝试标准模式...")
            browser = await p.chromium.launch(headless=headless, channel=pw_channel)
            browser_for_version = browser
            context = await browser.new_context(viewport={'width': 1366, 'height': 768})

        try:
            if browser_for_version is not None:
                print(f"[DIAG] browser={browser_for_version.browser_type.name} version={browser_for_version.version}")
        except Exception:
            pass

        page = await context.new_page()
        # 设置全局超时
        page.set_default_timeout(45000)

        async def _safe_dump(prefix: str):
            try:
                if page.is_closed():
                    return
            except Exception:
                return

            try:
                await page.screenshot(path=f"{prefix}.png")
            except Exception:
                pass

            try:
                with open(f"{prefix}.html", "w", encoding="utf-8") as f:
                    f.write(await page.content())
            except Exception:
                pass

        async def _safe_close_context():
            try:
                await context.close()
            except Exception:
                pass

        try:
            print("正在访问 Unsplash 开发者中心...")
            # 进一步降低对网络加载的敏感度
            await page.goto("https://unsplash.com/oauth/applications", wait_until="commit", timeout=60000)
            await page.wait_for_load_state("domcontentloaded")

            dashboard_heading = page.get_by_role("heading", name=re.compile(r"Your applications", re.I))
            try:
                await dashboard_heading.wait_for(state="visible", timeout=20000)
            except Exception:
                pass
            
            # 登录检测：如果 URL 包含 login 或者有 Log in 链接
            login_link = page.get_by_role("link", name=re.compile(r"Log in|Login", re.I))
            if "login" in page.url or await login_link.count() > 0:
                if headless:
                    print("错误: 需要登录。请先手动运行一次脚本进行登录：python scripts/register_unsplash_app.py")
                    await _safe_close_context()
                    return None
                print("检测到未登录，请在浏览器中完成登录并回到应用管理页面...")
                # 阻塞直到跳转回 applications
                await page.wait_for_url("**/oauth/applications", timeout=300000)
                await page.wait_for_load_state("domcontentloaded")

                try:
                    await dashboard_heading.wait_for(state="visible", timeout=30000)
                except Exception:
                    pass

            # 步骤 1: 检查是否能直接创建，或者需要点击按钮
            print("正在准备创建应用...")
            
            # 这种情况下通常是由于已经通过条款直接跳转到创建页，或者在列表页
            if "/oauth/applications/new" in page.url:
                pass
            else:
                # 尝试点击按钮
                new_app_btn = page.get_by_role("link", name=re.compile(r"New Application", re.I))
                if await new_app_btn.count() > 0:
                    await new_app_btn.first.click()
                else:
                    await page.goto("https://unsplash.com/oauth/applications/new", wait_until="domcontentloaded")

            await page.wait_for_load_state("domcontentloaded")

            # 步骤 2: 处理条款页
            # 如果出现复选框，说明在条款页
            legacy_terms_form = page.locator("form.js-accept-api-terms-form")
            legacy_terms_handled = False
            try:
                if await legacy_terms_form.count() > 0:
                    print("正在接受条款...")
                    required_cbs = legacy_terms_form.locator("input.js-accept-api-required[type='checkbox']")
                    try:
                        await required_cbs.first.wait_for(state="attached", timeout=15000)
                    except Exception:
                        pass

                    rc = 0
                    try:
                        rc = await required_cbs.count()
                    except Exception:
                        rc = 0

                    for i in range(rc):
                        cb = required_cbs.nth(i)
                        try:
                            if not await cb.is_checked():
                                await cb.check()
                        except Exception:
                            try:
                                if not await cb.is_checked():
                                    await cb.click(force=True)
                            except Exception:
                                continue

                    try:
                        await page.evaluate(
                            """() => {
                                document.querySelectorAll('form.js-accept-api-terms-form input.js-accept-api-required[type="checkbox"]').forEach(cb => {
                                    cb.dispatchEvent(new Event('change', { bubbles: true }));
                                });
                            }"""
                        )
                    except Exception:
                        pass

                    async def _force_open_legacy_modal():
                        try:
                            await page.evaluate(
                                """() => {
                                    const modal = document.getElementById('createApplication');
                                    if (!modal) return;

                                    const jq = window.jQuery;
                                    if (jq && typeof jq.fn === 'object' && typeof jq.fn.modal === 'function') {
                                        try { jq(modal).modal('show'); return; } catch (e) {}
                                    }

                                    modal.style.display = 'block';
                                    modal.classList.add('in');
                                    modal.classList.add('show');
                                    modal.setAttribute('aria-hidden', 'false');

                                    document.body.classList.add('modal-open');
                                    if (!document.querySelector('.modal-backdrop')) {
                                        const bd = document.createElement('div');
                                        bd.className = 'modal-backdrop fade in show';
                                        document.body.appendChild(bd);
                                    }
                                }"""
                            )
                        except Exception:
                            return

                    legacy_accept_btn = page.locator("button.js-accept-api-terms-button")
                    if await legacy_accept_btn.count() == 0:
                        legacy_accept_btn = page.locator("button:has-text('Accept terms')")

                    try:
                        await legacy_accept_btn.first.wait_for(state="visible", timeout=30000)
                    except Exception:
                        pass

                    try:
                        btn_handle = await legacy_accept_btn.first.element_handle()
                        if btn_handle:
                            await page.wait_for_function(
                                "(el) => !el.disabled && el.getAttribute('aria-disabled') !== 'true'",
                                arg=btn_handle,
                                timeout=30000,
                            )
                    except Exception:
                        pass

                    await legacy_accept_btn.first.click()

                    legacy_modal = page.locator("#createApplication")
                    try:
                        await legacy_modal.first.wait_for(state="attached", timeout=30000)
                    except Exception:
                        pass

                    async def _wait_legacy_modal_visible(timeout_ms: int) -> bool:
                        try:
                            modal_handle = await legacy_modal.first.element_handle()
                            if not modal_handle:
                                return False
                            await page.wait_for_function(
                                "(el) => { const s = window.getComputedStyle(el); const cls = el.classList; return (cls.contains('in') || cls.contains('show') || (s && s.display !== 'none' && s.visibility !== 'hidden')) && el.getAttribute('aria-hidden') !== 'true'; }",
                                arg=modal_handle,
                                timeout=timeout_ms,
                            )
                            return True
                        except Exception:
                            return False

                    if not await _wait_legacy_modal_visible(10000):
                        try:
                            await legacy_accept_btn.first.click(force=True)
                        except Exception:
                            pass

                    if not await _wait_legacy_modal_visible(10000):
                        await _force_open_legacy_modal()

                    if not await _wait_legacy_modal_visible(10000):
                        try:
                            await page.evaluate(
                                """() => {
                                    const btn = document.querySelector('button.js-accept-api-terms-button') || Array.from(document.querySelectorAll('button')).find(b => (b.textContent || '').trim() === 'Accept terms');
                                    if (btn) btn.click();
                                }"""
                            )
                        except Exception:
                            pass

                    if not await _wait_legacy_modal_visible(10000):
                        await _force_open_legacy_modal()

                    if not await _wait_legacy_modal_visible(15000):
                        raise RuntimeError("Legacy terms modal did not open")

                    name_probe = page.locator("#doorkeeper_application_name")
                    try:
                        await name_probe.wait_for(state="visible", timeout=60000)
                    except Exception:
                        pass

                    legacy_terms_handled = True
            except Exception:
                pass

            terms_accept_btn = page.get_by_role("button", name=re.compile(r"Accept\s*terms|I\s*agree|Agree|Accept", re.I))
            if await terms_accept_btn.count() == 0:
                terms_accept_btn = page.locator("button:has-text('Accept')").first

            on_terms_page = False
            try:
                if await terms_accept_btn.count() > 0:
                    on_terms_page = True
            except Exception:
                on_terms_page = False

            if on_terms_page and not legacy_terms_handled:
                print("正在接受条款...")
                terms_checkboxes = page.get_by_role("checkbox")
                try:
                    await terms_checkboxes.first.wait_for(state="visible", timeout=15000)
                except Exception:
                    pass

                cb_count = 0
                try:
                    cb_count = await terms_checkboxes.count()
                except Exception:
                    cb_count = 0

                if cb_count > 0:
                    for i in range(cb_count):
                        cb = terms_checkboxes.nth(i)
                        try:
                            await cb.wait_for(state="visible", timeout=15000)
                            if not await cb.is_checked():
                                await cb.click()
                        except Exception:
                            continue
                else:
                    raw_cbs = page.locator("input[type='checkbox']")
                    try:
                        await raw_cbs.first.wait_for(state="attached", timeout=15000)
                    except Exception:
                        pass
                    raw_count = 0
                    try:
                        raw_count = await raw_cbs.count()
                    except Exception:
                        raw_count = 0
                    for i in range(raw_count):
                        cb = raw_cbs.nth(i)
                        try:
                            if not await cb.is_checked():
                                await cb.check()
                        except Exception:
                            try:
                                if not await cb.is_checked():
                                    await cb.click(force=True)
                            except Exception:
                                continue

                await terms_accept_btn.first.wait_for(state="visible", timeout=30000)
                await terms_accept_btn.first.wait_for(state="attached", timeout=30000)
                await terms_accept_btn.first.wait_for(state="visible", timeout=30000)

                try:
                    btn_handle = await terms_accept_btn.first.element_handle()
                    if btn_handle:
                        await page.wait_for_function(
                            "(el) => !el.disabled && el.getAttribute('aria-disabled') !== 'true'",
                            arg=btn_handle,
                            timeout=30000,
                        )
                except Exception:
                    pass
                await terms_accept_btn.first.click()

            app_info_heading = page.get_by_role("heading", name=re.compile(r"Application information", re.I))
            try:
                await app_info_heading.wait_for(state="visible", timeout=30000)
            except Exception:
                pass

            # 步骤 3: 填写应用信息
            print("填写应用信息...")
            name_input = page.get_by_label(re.compile(r"Application name", re.I))
            if await name_input.count() == 0:
                name_input = page.get_by_role("textbox", name=re.compile(r"Application name", re.I))

            desc_input = page.get_by_label(re.compile(r"Description", re.I))
            if await desc_input.count() == 0:
                desc_input = page.get_by_role("textbox", name=re.compile(r"Description", re.I))
            
            # 如果没找到，可能是页面还没渲染完或结构又跳了，强制进入 /new 确保万无一失
            try:
                await name_input.wait_for(state="visible", timeout=30000)
            except Exception:
                try:
                    name_locator = page.locator("#doorkeeper_application_name")
                    if await name_locator.count() > 0:
                        await page.evaluate(
                            """() => {
                                const modal = document.getElementById('createApplication');
                                if (!modal) return;
                                const jq = window.jQuery;
                                if (jq && typeof jq.fn === 'object' && typeof jq.fn.modal === 'function') {
                                    try { jq(modal).modal('show'); return; } catch (e) {}
                                }
                                modal.style.display = 'block';
                                modal.classList.add('in');
                                modal.classList.add('show');
                                modal.setAttribute('aria-hidden', 'false');
                                document.body.classList.add('modal-open');
                                if (!document.querySelector('.modal-backdrop')) {
                                    const bd = document.createElement('div');
                                    bd.className = 'modal-backdrop fade in show';
                                    document.body.appendChild(bd);
                                }
                            }"""
                        )
                except Exception:
                    pass

                try:
                    legacy_modal = page.locator("#createApplication")
                    if await legacy_modal.count() > 0:
                        await page.evaluate(
                            """() => {
                                const btn = document.querySelector('button.js-accept-api-terms-button') || Array.from(document.querySelectorAll('button')).find(b => (b.textContent || '').trim() === 'Accept terms');
                                if (btn) btn.click();
                            }"""
                        )
                except Exception:
                    pass

                try:
                    await name_input.wait_for(state="visible", timeout=30000)
                except Exception:
                    print("未发现输入框，尝试强制重定向到新建应用表单...")
                    await page.goto("https://unsplash.com/oauth/applications/new", wait_until="domcontentloaded")
                    await name_input.wait_for(state="visible", timeout=30000)

            try:
                await name_input.wait_for(state="visible", timeout=30000)
            except Exception:
                pass

            await name_input.fill(app_name)
            await desc_input.fill("Dataset pipeline coordination for sharp-ply-share project.")
            
            # 步骤 4: 提交创建
            print("提交创建并等待详情页...")
            create_btn = page.get_by_role("button", name=re.compile(r"Create application", re.I))
            if await create_btn.count() == 0:
                create_btn = page.locator("input[type='submit'][value='Create application']")
            # 提交后可能需要较长时间处理
            await create_btn.wait_for(state="visible", timeout=30000)

            try:
                await page.wait_for_function(
                    "(el) => !el.disabled",
                    arg=await create_btn.element_handle(),
                    timeout=30000,
                )
            except Exception:
                pass

            await create_btn.click()

            access_key_probe = page.get_by_label(re.compile(r"^Access Key$", re.I))
            if await access_key_probe.count() == 0:
                access_key_probe = page.get_by_role("textbox", name=re.compile(r"^Access Key$", re.I))

            try:
                await page.wait_for_url(re.compile(r"https://unsplash\\.com/oauth/applications/\\d+"), timeout=60000)
            except Exception:
                pass

            try:
                await access_key_probe.first.wait_for(state="attached", timeout=60000)
            except Exception:
                pass

            # 步骤 5: 获取 Access Key (最稳健的解析方式)
            print("正在提取 Access Key...")
            access_key = None
            try:
                access_key_input = page.get_by_label(re.compile(r"^Access Key$", re.I))
                if await access_key_input.count() == 0:
                    access_key_input = page.get_by_role("textbox", name=re.compile(r"^Access Key$", re.I))
                await access_key_input.first.wait_for(state="attached", timeout=60000)
                await access_key_input.first.wait_for(state="visible", timeout=60000)

                try:
                    el = await access_key_input.first.element_handle()
                    if el is not None:
                        await page.wait_for_function(
                            "(el) => (el.value || '').trim().length > 20",
                            arg=el,
                            timeout=60000,
                        )
                except Exception:
                    pass

                access_key = (await access_key_input.first.input_value()).strip()
            except Exception:
                access_key = None

            if not access_key:
                try:
                    access_key = await page.evaluate("""() => {
                        const labelNodes = Array.from(document.querySelectorAll('label'));
                        for (const l of labelNodes) {
                            const t = (l.textContent || '').trim().toLowerCase();
                            if (t === 'access key') {
                                const id = l.getAttribute('for');
                                if (id) {
                                    const el = document.getElementById(id);
                                    if (el && el.value) return (el.value || '').trim();
                                }
                            }
                        }

                        const inputs = Array.from(document.querySelectorAll('input'));
                        for (const i of inputs) {
                            const v = (i.value || '').trim();
                            if (v.length > 20 && !v.includes(' ') && v !== '✓') {
                                return v;
                            }
                        }

                        return null;
                    }""")
                except Exception:
                    access_key = None

            if access_key:
                print(f"Access Key: {access_key}")
                with open("unsplash_keys_log.txt", "a") as log:
                    log.write(f"{datetime.now()}: {app_name} -> {access_key}\n")
                await context.close()
                return access_key
            else:
                await _safe_dump("reg_error_final")
                print("未能解析 Access Key，已截图 reg_error_final.png")
                await _safe_close_context()
                return None

        except Exception as e:
            print(f"自动化注册流程失败: {e}")
            await _safe_dump("process_exception")
            await _safe_close_context()
            return None

if __name__ == "__main__":
    name = f"sharp-ply-share-{random.randint(100, 999)}"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    asyncio.run(register_unsplash_app(name))
