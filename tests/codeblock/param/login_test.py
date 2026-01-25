import pandas as pd
import os
import asyncio
from playwright.async_api import async_playwright, Page, expect

csv_path = r"/Users/niwatorimostiqo/Desktop/Coding/Mitou-Fukuoka/tests/csv_s/login_test.csv"
df = pd.read_csv(csv_path)

print(f"Running {len(df)} test cases from CSV")

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False,slow_mo=50)

        for index, row in df.iterrows():
            print(f"\\n=== Test Case {index + 1}/{len(df)} ===")
            print(f"Input values: {dict(row)}")
            context = await browser.new_context()
            page = await context.new_page()        
            try:

                await page.goto("http://localhost:5173/")
                await page.wait_for_load_state("networkidle")

                email_value = str(row["email"]) if pd.notna(row["email"]) else ""
                password_value = str(row["password"]) if pd.notna(row["password"]) else ""
                expected_result = str(row["expected_result"]) if pd.notna(row["expected_result"]) else ""

                await page.fill("#login-email", email_value)
                await page.fill("#login-password", password_value)

                await page.click("button[type='submit']")
                await page.wait_for_load_state("networkidle")

                # Extract the actual result from the designated element
                result_element = page.locator("#login-result")
                await result_element.wait_for(state="visible")
                actual_result = await result_element.text_content()

                # Flexible assertion
                assert expected_result.lower() in actual_result.lower() or actual_result.lower() in expected_result.lower(), \
                    f"Assertion failed: Expected '{expected_result}' to be in or contain '{actual_result}'"
            except Exception as e:
                print(f"Test case {index + 1} FAILED: {e}")
            else:
                print(f"Test case {index + 1} PASSED")
            finally:
                await context.close()
        await browser.close()             

if __name__ == "__main__":
    asyncio.run(main())
    print("\nAll tests completed!")
