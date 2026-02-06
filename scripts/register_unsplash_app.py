import asyncio
import sys
import os
import random
from playwright.async_api import async_playwright

async def register_unsplash_app(app_name: str, headless: bool = False) -> str | None:
    """
    Registers a new Unsplash app and returns the Access Key.
    """
    async with async_playwright() as p:
        # Use persistent context to reuse login session
        user_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".unsplash_session")
        try:
            os.makedirs(os.path.dirname(user_data_dir), exist_ok=True)
        except Exception:
            pass
            
        context = await p.chromium.launch_persistent_context(user_data_dir, headless=headless)
        page = await context.new_page()

        try:
            print(f"正在访问 Unsplash 开发者页面...")
            await page.goto("https://unsplash.com/oauth/applications", timeout=60000)

            # Check for login
            if await page.locator("text=Login").is_visible() or "login" in page.url:
                if headless:
                    print("错误: 在无头模式下需要登录。请先手动运行一次脚本进行登录。")
                    await context.close()
                    return None
                print("请在浏览器中完成登录...")
                await page.wait_for_url("**/oauth/applications", timeout=300000)

            # Check if we can create more (max 5 in demo mode usually)
            apps_count = await page.locator(".js-application-row").count()
            if apps_count >= 10: # Just a safety limit
                print(f"应用数量已达上限 ({apps_count})，跳过注册。")
                await context.close()
                return None

            print("点击 'New Application'...")
            new_app_button = page.locator("text=New Application")
            if await new_app_button.is_visible():
                await new_app_button.click()
            else:
                await page.goto("https://unsplash.com/oauth/applications/new")

            print("接受条款...")
            # Wait for checkboxes
            await page.wait_for_selector("input[type='checkbox']", timeout=30000)
            checkboxes = page.locator("input[type='checkbox']")
            count = await checkboxes.count()
            for i in range(count):
                try:
                    await checkboxes.nth(i).check()
                except Exception:
                    pass
            
            await page.click("text=Accept terms")

            print("填写应用信息...")
            await page.fill("input[name='application[name]']", app_name)
            await page.fill("textarea[name='application[description]']", "one image to one ply, not high-quality, just for fun")
            
            print("提交创建...")
            await page.click("text=Create application")

            # Wait for Redirect to details page
            await page.wait_for_load_state("networkidle")
            
            # Find the Access Key
            print("正在获取 Access Key...")
            # Access key is usually in a code/input field under "Keys" section
            # Looking for a field that looks like a 43-character hex string
            keys_section = page.locator("text=Access Key")
            await keys_section.wait_for(timeout=30000)
            
            # Access key is in an input field right after the label "Access Key"
            access_key_input = page.locator("label:has-text('Access Key') + div input, div:has-text('Access Key') + div input, input[readonly]")
            # There might be multiple (Access Key, Secret Key). Access Key is first.
            access_key = await access_key_input.first.input_value()
            
            if access_key and len(access_key) > 20:
                print(f"应用 '{app_name}' 创建成功！Access Key: {access_key}")
                await asyncio.sleep(2)
                await context.close()
                return access_key
            else:
                print("未能自动获取 Access Key，请检查页面。")
                await asyncio.sleep(5)
                await context.close()
                return None
        except Exception as e:
            print(f"自动化注册失败: {str(e)}")
            await asyncio.sleep(5)
            await context.close()
            return None

if __name__ == "__main__":
    name = f"sharp-ply-share-{random.randint(100, 999)}"
    if len(sys.argv) > 1:
        name = sys.argv[1]
    asyncio.run(register_unsplash_app(name))
