import asyncio
import sys
import os
import random
import re
from datetime import datetime
from playwright.async_api import async_playwright

async def register_unsplash_app(app_name: str, headless: bool = False) -> str | None:
    """
    Registers a new Unsplash app and returns the Access Key.
    """
    async with async_playwright() as p:
        user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".unsplash_session")
        try:
            os.makedirs(os.path.dirname(user_data_dir), exist_ok=True)
        except Exception:
            pass
            
        try:
            # 增加更多真实的浏览器启动参数，减少被判定为机器人的概率
            context = await p.chromium.launch_persistent_context(
                user_data_dir, 
                headless=headless,
                viewport={'width': 1366, 'height': 768},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                ignore_https_errors=True
            )
        except Exception as e:
            print(f"无法启动 Chrome/Chromium: {e}. 尝试标准模式...")
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(viewport={'width': 1366, 'height': 768})

        page = await context.new_page()
        # 设置全局超时
        page.set_default_timeout(45000)

        try:
            print("正在访问 Unsplash 开发者中心...")
            # 进一步降低对网络加载的敏感度
            await page.goto("https://unsplash.com/oauth/applications", wait_until="commit", timeout=60000)
            await asyncio.sleep(2)
            
            # 登录检测：如果 URL 包含 login 或者有 Log in 链接
            login_link = page.get_by_role("link", name=re.compile(r"Log in|Login", re.I))
            if "login" in page.url or await login_link.count() > 0:
                if headless:
                    print("错误: 需要登录。请先手动运行一次脚本进行登录：python scripts/register_unsplash_app.py")
                    await context.close()
                    return None
                print("检测到未登录，请在浏览器中完成登录并回到应用管理页面...")
                # 阻塞直到跳转回 applications
                await page.wait_for_url("**/oauth/applications", timeout=300000)
                await asyncio.sleep(2)

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

            # 步骤 2: 处理条款页
            # 如果出现复选框，说明在条款页
            if await page.get_by_role("checkbox").count() > 0:
                print("正在接受条款...")
                await page.evaluate("""() => {
                    document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                        if (!cb.checked) {
                            cb.click();
                            cb.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                    });
                }""")
                
                accept_btn = page.get_by_role("button", name=re.compile(r"Accept terms", re.I))
                if await accept_btn.count() > 0:
                    await accept_btn.first.click()
                    # 关键：等待 Modal 弹出或页面内容切换，不需要网络空闲，因为 Unsplash 用的是局部渲染
                    await asyncio.sleep(3)

            # 步骤 3: 填写应用信息
            print("填写应用信息...")
            # 显式寻找输入框，即使在 Modal 中 get_by_role 也是有效的
            name_input = page.get_by_role("textbox", name=re.compile(r"Application name", re.I))
            desc_input = page.get_by_role("textbox", name=re.compile(r"Description", re.I))
            
            # 如果没找到，可能是页面还没渲染完或结构又跳了，强制进入 /new 确保万无一失
            try:
                await name_input.wait_for(state="visible", timeout=10000)
            except Exception:
                print("未发现输入框，尝试强制重定向到新建应用表单...")
                await page.goto("https://unsplash.com/oauth/applications/new", wait_until="domcontentloaded")
                await name_input.wait_for(state="visible", timeout=10000)

            await name_input.fill(app_name)
            await desc_input.fill("Dataset pipeline coordination for sharp-ply-share project.")
            
            # 步骤 4: 提交创建
            print("提交创建并等待详情页...")
            create_btn = page.get_by_role("button", name=re.compile(r"Create application", re.I))
            # 提交后可能需要较长时间处理
            await create_btn.click()

            # 步骤 5: 获取 Access Key (最稳健的解析方式)
            print("正在提取 Access Key...")
            # 创建成功后会跳转，或者 Modal 切换。我们等待 access_key 元素出现
            key_selector = "#access_key"
            try:
                await page.wait_for_selector(key_selector, state="attached", timeout=30000)
            except Exception:
                print("未检测到标准 Access Key 元素，尝试全文正则提取...")

            # 解析逻辑
            access_key = await page.evaluate("""() => {
                // 优先取 ID 匹配的
                const el = document.querySelector('#access_key');
                if (el && el.value && el.value.length > 20 && el.value !== '✓') return el.value;
                
                // 其次取所有只读输入框中看起来像 Key 的
                const inputs = Array.from(document.querySelectorAll('input[readonly]'));
                for (const i of inputs) {
                    const v = i.value.trim();
                    if (v.length > 20 && !v.includes(' ') && v !== '✓') return v;
                }
                return null;
            }""")

            if access_key:
                print(f"应用注册成功！Key: {access_key}")
                with open("unsplash_keys_log.txt", "a") as log:
                    log.write(f"{datetime.now()}: {app_name} -> {access_key}\n")
                await asyncio.sleep(2)
                await context.close()
                return access_key
            else:
                await page.screenshot(path="reg_error_final.png")
                print("未能解析 Access Key，已截图 reg_error_final.png")
                await context.close()
                return None

        except Exception as e:
            print(f"自动化注册流程失败: {e}")
            await page.screenshot(path="process_exception.png")
            await context.close()
            return None

if __name__ == "__main__":
    name = f"sharp-ply-share-{random.randint(100, 999)}"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    asyncio.run(register_unsplash_app(name))
