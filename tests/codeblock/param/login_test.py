import pandas as pd
import os
import asyncio
from playwright.async_api import async_playwright, Page, expect

csv_path = r"/Users/niwatorimostiqo/Desktop/Coding/Mitou-Fukuoka/tests/csv_s/login_test.csv"
df = pd.read_csv(csv_path)

print(f"Running {len(df)} test cases from CSV")

async def main():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)

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

                                            await page.click("button.submit-btn")
                                            await page.wait_for_load_state("networkidle")

                                            actual_result_element = page.locator("[ref=e38]")
                                            actual_result_text = await actual_result_element.text_content()
                                            actual_result_text = actual_result_text.strip() if actual_result_text else ""

                                            print(f"Expected: '{expected_result}'")
                                            print(f"Actual: '{actual_result_text}'")

                                            assert expected_result.lower() in actual_result_text.lower() or actual_result_text.lower() in expected_result.lower(), \
                                                f"Assertion failed: Expected '{expected_result}' to be in or contain '{actual_result_text}'"
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
